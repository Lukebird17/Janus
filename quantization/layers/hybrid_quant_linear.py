# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
混合量化线性层

支持的优化方法（可组合）：
1. SmoothQuant - 激活/权重平滑
2. SVD 分解 - 低秩近似
3. 异常值提取（Outlier Extraction）- 使用稀疏矩阵存储异常值
4. Weight-Activation 量化

🔥 新的叠加顺序（用户要求）：
原始 W → [SmoothQuant] → W' = W × s
       → [SVD] → W' ≈ U·Σ·V^T + R_svd
       → [异常值提取] → R_svd = W_outlier + R_final (从 SVD 残差中提取)
       → [量化] → Quant(R_final)

前向传播重建：
W = U·Σ·V^T + W_outlier + Dequant(R_final)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HybridQuantLinear(nn.Module):
    """
    混合量化线性层
    
    支持：异常值提取 + SmoothQuant + SVD + GPTQ + 量化的组合
    """
    
    # 🔥 类变量：缓存完整激活值数据（所有层实例共享，避免重复加载）
    _cached_full_activations = None
    _activation_load_attempted = False
    _activation_config = None  # 🔥 新增：存储激活文件配置，用于延迟加载
    
    @classmethod
    def set_activation_config(cls, config):
        """
        设置激活数据配置（推荐方式）
        
        这个方法只存储配置，不立即加载数据。
        数据会在第一次需要时自动加载（延迟加载）。
        
        Args:
            config: 包含激活文件路径的配置字典，支持以下格式：
                - {'stage0_full_activation_file': '/path/to/file.pt'}
                - {'gptq': {'full_activation_file': '/path/to/file.pt'}}
                - 或者直接传入文件路径字符串
        
        Example:
            # 方式 1：传入配置字典
            HybridQuantLinear.set_activation_config({
                'stage0_full_activation_file': '/path/to/activations.pt'
            })
            
            # 方式 2：直接传入路径
            HybridQuantLinear.set_activation_config('/path/to/activations.pt')
            
            # 之后创建的所有层会自动使用这个配置
            layer = HybridQuantLinear(..., use_gptq=True)
            layer.prepare_weight(layer_name='...')  # 自动加载激活数据
        """
        from pathlib import Path
        
        if isinstance(config, str):
            # 如果直接传入路径字符串，转换为字典格式
            config_path = config
            config = {'stage0_full_activation_file': config}
            
            # 验证文件是否存在
            if not Path(config_path).exists():
                print(f"⚠️  [HybridQuantLinear] Warning: Activation file not found: {config_path}")
                print(f"    GPTQ quantization will fail unless file is provided later")
        
        cls._activation_config = config
        print(f"✓ [HybridQuantLinear] Activation config set, data will be loaded on first use")
        
        # 重置加载尝试标志，以便可以重新尝试加载
        cls._activation_load_attempted = False
    
    @classmethod
    def set_full_activations(cls, full_activations_dict):
        """设置完整激活值数据（由外部调用一次）"""
        cls._cached_full_activations = full_activations_dict
        cls._activation_load_attempted = True
    
    @classmethod
    def clear_full_activations_cache(cls, verbose=False):
        """
        清理激活数据缓存（量化完成后调用以释放显存）
        
        ⚠️  注意：调用后将无法再使用GPTQ量化新的层
        """
        if cls._cached_full_activations is not None:
            if verbose:
                num_layers = len(cls._cached_full_activations) if cls._cached_full_activations else 0
                print(f"  [Memory] Clearing activation cache ({num_layers} layers)...")
            
            del cls._cached_full_activations
            cls._cached_full_activations = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if verbose:
                print(f"  [Memory] Activation cache cleared")
    
    @classmethod
    def load_full_activations_from_config(cls, config, verbose=True):
        """
        从配置文件自动加载完整激活值（用于 GPTQ）
        
        支持多种配置方式（按优先级）：
        1. config['stage0_full_activation_file']  # Stage 特定
        2. config['gptq']['full_activation_file'] # 通用配置
        3. 自动查找默认路径
        """
        if cls._activation_load_attempted:
            return cls._cached_full_activations
        
        cls._activation_load_attempted = True
        
        import torch
        from pathlib import Path
        
        activation_file = None
        
        # 优先级 1: stage0_full_activation_file
        if 'stage0_full_activation_file' in config and config['stage0_full_activation_file']:
            activation_file = config['stage0_full_activation_file']
        
        # 优先级 2: gptq.full_activation_file
        elif 'gptq' in config and 'full_activation_file' in config['gptq']:
            activation_file = config['gptq']['full_activation_file']
        
        # 优先级 3: 自动查找
        else:
            default_paths = [
                'quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt',
                'quantization_outputs/stage0_full_activation/stage0_full_activation_100samples_latest.pt',
                './quantization_outputs/stage0_full_activation/stage0_full_activation_50samples_latest.pt',
            ]
            for path in default_paths:
                if Path(path).exists():
                    activation_file = path
                    if verbose:
                        print(f"⚠️  Auto-detected GPTQ activation file: {path}")
                    break
        
        # 没找到
        if not activation_file:
            if verbose:
                print("\n⚠️  No GPTQ activation data found, GPTQ will fall back to RTN")
            return None
        
        # 加载
        try:
            if verbose:
                print(f"\n🔥 Loading GPTQ activation data from {activation_file}...")
            
            activation_path = Path(activation_file)
            if not activation_path.exists():
                if verbose:
                    print(f"⚠️  File not found: {activation_file}")
                return None
            
            cls._cached_full_activations = torch.load(activation_path, map_location='cpu')
            
            if verbose:
                num_layers = len(cls._cached_full_activations)
                if num_layers > 0:
                    sample_layer = list(cls._cached_full_activations.keys())[0]
                    sample_shape = cls._cached_full_activations[sample_layer].shape
                    file_size_mb = activation_path.stat().st_size / (1024 * 1024)
                    
                    print(f"✓ Loaded GPTQ activation data: {num_layers} layers")
                    print(f"  Sample shape: {sample_shape}")
                    print(f"  File size: {file_size_mb:.1f} MB")
            
            return cls._cached_full_activations
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load: {e}")
            return None
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        # 量化配置
        weight_bit: int = 8,
        act_bit: int = 8,
        quant_percentile: float = 0.999999,  # 量化百分位数 (99.9999%)
        act_unsigned: bool = False,  # 激活是否使用 unsigned 量化（INT4: 0-15 vs -8-7）
        # 异常值提取配置
        use_sparse: bool = False,
        sparse_ratio: float = 0.0001,  # 异常值比例
        sparse_threshold: Optional[float] = None,  # 阈值（如果指定，优先使用）
        # SmoothQuant 配置
        use_smoothquant: bool = False,
        smoothquant_alpha: float = 0.5,
        # SVD 配置
        use_svd: bool = False,
        svd_rank: int = 32,  # SVD 秩
        # RTN 分块量化配置 (Block Quantization - 控制量化粒度)
        use_block_quant: bool = False,  # 是否启用权重的分块量化 (RTN only)
        use_block_quant_act: bool = False,  # 是否启用激活的分块量化 (RTN only)
        block_size_weight: int = 128,  # RTN 权重分块大小（量化粒度）
        block_size_act: int = 128,  # RTN 激活分块大小（量化粒度）
        # GPTQ 配置
        use_gptq: bool = False,  # 是否使用 GPTQ 量化
        gptq_group_size: int = 64,  # GPTQ group size (per-group quantization)
        gptq_damp_percentage: float = 0.01,  # GPTQ damping 百分比
        gptq_block_size: int = 128,  # GPTQ block size (算法优化)
        gptq_num_inv_tries: int = 250,  # Hessian 逆矩阵计算尝试次数
        gptq_hessian_block_size: int = 512,  # Hessian 矩阵计算分块大小
        # AWQ 配置
        use_awq: bool = False,
        awq_alpha: float = 0.5,
        awq_n_grid: int = 20,
        # 其他
        device=None,
        dtype=None,
    ):
        super().__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # 量化配置
        self.weight_bit = weight_bit
        self.act_bit = act_bit
        self.quant_percentile = quant_percentile  # 百分位数
        self.act_unsigned = act_unsigned  # 激活 unsigned 量化
        
        # 优化方法开关
        self.use_sparse = use_sparse
        self.use_smoothquant = use_smoothquant
        self.use_svd = use_svd
        self.use_block_quant = use_block_quant  # 权重的分块量化开关
        self.use_block_quant_act = use_block_quant_act  # 激活的分块量化开关
        
        # 异常值提取配置
        self.sparse_ratio = sparse_ratio
        self.sparse_threshold = sparse_threshold
        
        # SmoothQuant 配置
        self.smoothquant_alpha = smoothquant_alpha
        
        # AWQ 配置
        self.use_awq = use_awq
        self.awq_alpha = awq_alpha
        self.awq_n_grid = awq_n_grid
        
        # SVD 配置
        self.svd_rank = svd_rank
        
        # RTN 分块量化配置
        self.block_size_weight = block_size_weight
        self.block_size_act = block_size_act
        
        # GPTQ 配置
        self.use_gptq = use_gptq
        self.gptq_group_size = gptq_group_size
        self.gptq_damp_percentage = gptq_damp_percentage
        self.gptq_block_size = gptq_block_size
        self.gptq_num_inv_tries = gptq_num_inv_tries
        self.gptq_hessian_block_size = gptq_hessian_block_size
        
        # 🔥 性能优化选项：缓存反量化权重（牺牲显存换速度）
        # 警告：启用缓存会让显存占用增加约 50%
        # 建议：推理时启用以显著加速（避免每次重建权重）
        # 🔥 默认禁用权重缓存以节省显存
        # 缓存会存储完整的 BF16 权重，完全抵消量化的显存节省
        # 🔥 权重矩阵在推理时不常变化，缓存可以大幅加速
        # 优化阶段会频繁推理，启用缓存能减少重复计算 SVD 和反量化
        self.cache_dequant_weight = True  # 🔥 启用缓存，优化推理速度
        
        # 原始权重（训练时使用）
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # 异常值存储（如果启用）- 🔥 使用稀疏格式节省显存
        if self.use_sparse:
            # 🔥 不再使用密集的 outlier_weight，改用稀疏格式：
            # self.register_buffer('outlier_weight', None)  # [out_features, in_features] 太占显存
            self.register_buffer('outlier_indices', None)  # [num_outliers, 2] 索引
            self.register_buffer('outlier_values', None)   # [num_outliers] 值
            self.register_buffer('outlier_shape', None)    # [2] 原始形状
            # 🔥 不保存 weight_residual，它只是中间变量，不需要持久化
            # self.register_buffer('weight_residual', None)  # 浪费显存
        
        # SmoothQuant / AWQ 缩放因子（如果启用）
        if self.use_smoothquant or self.use_awq:
            self.register_buffer('smooth_scales', None)  # [in_features] per-channel scales
        
        # SVD 分解（如果启用）
        if self.use_svd:
            self.register_buffer('svd_U', None)  # [out_features, rank]
            self.register_buffer('svd_S', None)  # [rank]
            self.register_buffer('svd_V', None)  # [rank, in_features]
            # 🔥 不保存 svd_residual，它只是中间变量，不需要持久化
            # self.register_buffer('svd_residual', None)  # 浪费显存
        
        # 量化参数
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        self.register_buffer('act_scale', None)
        self.register_buffer('act_zero_point', None)
        
        # 量化后的权重
        self.register_buffer('quantized_weight', None)
        
        # 🔥 缓存反量化后的权重（推理时使用，避免每次重建）
        self.register_buffer('_cached_dequant_weight', None)
        self._weight_cache_dtype = None
        
        # 🔥 缓存SVD权重矩阵（在prepare_weight时计算，forward时直接使用）
        self.register_buffer('_cached_svd_weight', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def extract_outliers(self, weight: torch.Tensor, activation_stats: Optional[torch.Tensor] = None, verbose: bool = False):
        """
        提取权重异常值，使用密集矩阵存储
        
        🔥 显存优化版本：避免创建完整的importance矩阵
        
        Args:
            weight: 权重矩阵 [out_features, in_features]
            activation_stats: 激活值统计 [in_features]（用于加权判断）
            verbose: 是否输出详细信息
        
        Returns:
            outlier_mask: 异常值 mask
            outlier_values: 异常值
            residual: 残差权重
        """
        # 🔥 显存优化：直接在weight上操作，不创建额外的importance矩阵
        if activation_stats is not None:
            # 重要性 = |w| × |x|
            # 🔥 优化：使用in-place操作减少显存
            importance = weight.abs() * activation_stats.unsqueeze(0)
        else:
            importance = weight.abs()
        
        # 确定阈值
        if self.sparse_threshold is not None:
            # 🔥 sparse_threshold表示保留的outlier比例（例如0.01表示保留最大的1%）
            # 使用percentile而不是相对于最大值，这样更直观
            importance_flat = importance.flatten().float()
            
            # 计算percentile阈值：保留最大的sparse_threshold比例的权重
            percentile_value = 1.0 - self.sparse_threshold  # 0.01 -> 0.99 (99th percentile)
            
            if importance_flat.numel() > 1000000:
                # 大权重矩阵：随机采样估计
                indices = torch.randperm(importance_flat.numel(), device=importance_flat.device)[:1000000]
                threshold = torch.quantile(importance_flat[indices], percentile_value)
            else:
                threshold = torch.quantile(importance_flat, percentile_value)
            
            del importance_flat
        else:
            # 使用百分位数（随机采样策略）
            importance_flat = importance.flatten().float()
            
            # 🔥 使用随机采样而非固定步长（避免引入偏差）
            if importance_flat.numel() > 1000000:  # 超过 100 万个元素才采样
                # 随机采样 100 万个元素（足够估计百分位数）
                indices = torch.randperm(importance_flat.numel(), device=importance_flat.device)[:1000000]
                threshold = torch.quantile(importance_flat[indices], 1 - self.sparse_ratio)
            else:
                threshold = torch.quantile(importance_flat, 1 - self.sparse_ratio)
            
            # 🔥 及时释放不需要的tensor
            del importance_flat
        
        # 提取异常值
        outlier_mask = importance > threshold
        
        # 🔥 及时释放importance
        del importance
        
        outlier_values = weight[outlier_mask]  # [nnz]
        
        # 🔥 显存优化：in-place修改而非clone
        residual = weight.clone()
        residual[outlier_mask] = 0.0
        
        if verbose:
            num_outliers = outlier_mask.sum().item()
            print(f"  [Outliers] Extracted {num_outliers} outliers "
                  f"({num_outliers / weight.numel() * 100:.2f}%)")
        
        return outlier_mask, outlier_values, residual
    
    def compute_smoothquant_scales(
        self, 
        activation_max: torch.Tensor,  # [in_features]
        weight_max: torch.Tensor,  # [in_features]
    ) -> torch.Tensor:
        """
        计算 SmoothQuant 缩放因子
        
        s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
        
        Args:
            activation_max: 每个 channel 的激活最大值
            weight_max: 每个 channel 的权重最大值（跨 output channel）
        
        Returns:
            scales: [in_features] 缩放因子
        """
        alpha = self.smoothquant_alpha
        
        # 避免除零
        activation_max = torch.clamp(activation_max, min=1e-5)
        weight_max = torch.clamp(weight_max, min=1e-5)
        
        # s = (max_act)^α / (max_weight)^(1-α)
        # 🔥 类型修复：确保使用 bfloat16 进行计算和返回
        activation_max = activation_max.to(torch.bfloat16)
        weight_max = weight_max.to(torch.bfloat16)
        scales = (activation_max ** alpha) / (weight_max ** (1 - alpha)).clamp(min=1e-5)
        
        print(f"  [SmoothQuant] α={alpha:.2f}, scale range: "
              f"[{scales.min().item():.6f}, {scales.max().item():.6f}]")
        
        # 🔥 确保返回 bfloat16
        return scales.to(torch.bfloat16)
    
    def apply_smoothquant(self, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        应用 SmoothQuant 到权重
        
        Ŵ = diag(s) @ W
        
        Args:
            weight: [out_features, in_features]
            scales: [in_features]
        
        Returns:
            smoothed_weight: [out_features, in_features]
        """
        # W' = W @ diag(s) = W * s
        smoothed = weight * scales.unsqueeze(0)
        return smoothed
    
    def compute_awq_scales(
        self,
        weight: torch.Tensor,
        activation_data: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        AWQ (Activation-Aware Weight Quantization) 缩放因子计算

        与 SmoothQuant 互斥，通过 grid search 找最优 α 使量化误差最小。
        s = mean(|X|)^α，显著通道权重被放大，量化后误差更小。

        Args:
            weight: [out_features, in_features]
            activation_data: [num_samples, in_features]
            verbose: 是否输出详细信息

        Returns:
            best_scales: [in_features] 最优缩放因子
        """
        device = weight.device
        dtype = weight.dtype

        act_data = activation_data.to(device=device, dtype=torch.float32)
        w = weight.float()

        act_mean = act_data.abs().mean(dim=0)  # [in_features]
        act_mean = torch.clamp(act_mean, min=1e-5)

        n_grid = self.awq_n_grid
        best_error = float('inf')
        best_scales = torch.ones(w.shape[1], device=device, dtype=torch.float32)
        best_alpha = 0.0

        org_out = (act_data @ w.t())  # [num_samples, out_features]

        for ratio in range(n_grid + 1):
            alpha = ratio / n_grid
            scales = act_mean.pow(alpha).clamp(min=1e-4)

            scaled_w = w * scales.unsqueeze(0)
            scaled_act = act_data / scales.unsqueeze(0)

            q_w, q_s, q_z = self.quantize_tensor(
                scaled_w.to(dtype), self.weight_bit,
                per_channel=True, channel_dim=0,
                percentile=self.quant_percentile,
                use_block=self.use_block_quant,
                block_size=self.block_size_weight,
                unsigned=False,
            )
            dq_w = self.dequantize_tensor(
                q_w.float(), q_s.float(), q_z.float(),
                is_blocked=(self.use_block_quant and len(q_s.shape) == 2),
                block_size=self.block_size_weight,
            ).float()

            q_out = scaled_act @ dq_w.t()
            error = (org_out - q_out).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_scales = scales.clone()
                best_alpha = alpha

            del q_w, q_s, q_z, dq_w, q_out, scaled_w, scaled_act

        if verbose:
            print(f"  [AWQ] Grid search done: best α={best_alpha:.2f}, "
                  f"error={best_error:.6f}, "
                  f"scale range: [{best_scales.min().item():.4f}, {best_scales.max().item():.4f}]")

        del act_data, w, org_out
        return best_scales.to(dtype)

    def compute_svd(self, weight: torch.Tensor, rank: int, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对权重进行 SVD 分解（优化显存使用和速度）
        
        W ≈ U @ S @ V^T
        
        Args:
            weight: [out_features, in_features]
            rank: SVD 秩
            verbose: 是否输出详细信息
        
        Returns:
            U: [out_features, rank]
            S: [rank]
            V: [rank, in_features]
            residual: W - U @ S @ V^T
        """
        # 保存原始设备和数据类型
        original_device = weight.device
        original_dtype = weight.dtype
        

        out_features, in_features = weight.shape
        matrix_size = out_features * in_features
        weight_compute = weight.float()
        compute_device = weight.device

        
        # 使用 full_matrices=False 只计算必要的部分
        U, S, Vh = torch.linalg.svd(weight_compute, full_matrices=False)
        
        # 立即截断到指定秩，减少内存占用
        U_r = U[:, :rank].contiguous()
        S_r = S[:rank].contiguous()
        V_r = Vh[:rank, :].contiguous()
        
        # 清理中间结果
        del U, S, Vh, weight_compute
        
        # 移回原始设备并转换回原始数据类型
        U_r = U_r.to(device=original_device, dtype=original_dtype)
        S_r = S_r.to(device=original_device, dtype=original_dtype)
        V_r = V_r.to(device=original_device, dtype=original_dtype)
        
        # 计算残差（使用更节省显存的方式）
        reconstructed = (U_r * S_r.unsqueeze(0)) @ V_r
        residual = weight - reconstructed
        del reconstructed
        
        if verbose:
            # 只在需要时计算这些统计信息
            original_size = weight.numel()
            compressed_size = U_r.numel() + S_r.numel() + V_r.numel()
            compression_ratio = original_size / compressed_size
            relative_error = torch.norm(residual) / torch.norm(weight)
            print(f"  [SVD] rank={rank}, compression={compression_ratio:.2f}x, "
                  f"relative_error={relative_error.item():.6f}")
        
        return U_r, S_r, V_r, residual
    
    def quantize_tensor(
        self, 
        tensor: torch.Tensor, 
        n_bits: int,
        per_channel: bool = False,
        channel_dim: int = 0,
        percentile: float = 0.999999,  # 99.9999% 百分位数
        use_block: bool = False,  # 是否使用分块量化
        block_size: int = 128,  # 分块大小
        unsigned: bool = False,  # 是否使用 unsigned 量化
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        量化张量（支持对称/非对称量化，基于百分位数）
        
        使用 99.9999% 百分位数代替最大值，减少极端异常值的影响
        
        支持三种量化粒度：
        1. Per-tensor: 整个张量共享一个 scale
        2. Per-channel: 每个 channel 独立 scale
        3. Per-block (channel 内分块): 每个 channel 内的每个 block 独立 scale
        
        Args:
            tensor: 待量化张量
            n_bits: 位宽
            per_channel: 是否 per-channel 量化
            channel_dim: channel 维度
            percentile: 百分位数阈值 (default: 0.999999 = 99.9999%)
            use_block: 是否使用分块量化 (仅在 per_channel=True 时有效)
            block_size: 分块大小 (channel 内分块)
            unsigned: 是否使用 unsigned 量化 (True: 0到2^n-1, False: -2^(n-1)到2^(n-1)-1)
        
        Returns:
            quantized: 量化后的张量
            scale: 缩放因子 (per-block 时为 [C, num_blocks])
            zero_point: 零点（对称量化为0，非对称量化为计算值）
        """
        if per_channel:
            # 将 channel 维度移到第一维
            if channel_dim != 0:
                tensor = tensor.transpose(0, channel_dim)
            
            shape = tensor.shape
            n_channels = shape[0]
            
            # 🔥 分块量化：在 channel 内进一步分块
            if use_block and block_size > 0:
                # Per-block (within channel): 每个 channel 内的每个 block 独立 scale
                tensor_flat = tensor.reshape(shape[0], -1)  # [C, N]
                n_elements = tensor_flat.shape[1]
                
                # 计算分块数量，向上取整
                num_blocks = (n_elements + block_size - 1) // block_size
                # print("block_size",block_size)
                # Pad 到整数个 blocks
                pad_size = num_blocks * block_size - n_elements
                if pad_size > 0:
                    tensor_flat = F.pad(tensor_flat, (0, pad_size), mode='constant', value=0)
                
                # Reshape 成 [C, num_blocks, block_size]
                tensor_blocked = tensor_flat.reshape(n_channels, num_blocks, block_size)
                del tensor_flat  # 🔥 及时释放显存
                
                abs_tensor = tensor_blocked.abs().float()
                
                # 计算每个 block 的百分位数 [C, num_blocks]
                device_orig = abs_tensor.device
                
                if block_size > 10000:  # 如果 block 很大才采样
                    # 对每个 block 采样
                    sample_size = min(10000, block_size)
                    indices = torch.randperm(block_size, device=abs_tensor.device)[:sample_size]
                    sampled = abs_tensor[:, :, indices]
                    del indices  # 🔥 及时释放显存
                    max_val = torch.quantile(sampled, percentile, dim=2)  # [C, num_blocks]
                    del sampled  # 🔥 及时释放显存
                else:
                    max_val = torch.quantile(abs_tensor, percentile, dim=2)  # [C, num_blocks]
                
                del abs_tensor  # 🔥 及时释放显存
                
                if max_val.device != device_orig:
                    max_val = max_val.to(device_orig)
                max_val = torch.clamp(max_val, min=1e-5)
                
                # 计算量化范围
                if unsigned:
                    qmin = 0
                    qmax = 2 ** n_bits - 1
                    # 计算 zero_point（针对每个block）
                    min_val = tensor_blocked.min(dim=2)[0]  # [C, num_blocks]
                    scale = (max_val - min_val) / qmax
                    scale = torch.clamp(scale, min=1e-5)
                    zero_point_val = -min_val / scale
                else:
                    qmin = -(2 ** (n_bits - 1))
                    qmax = 2 ** (n_bits - 1) - 1
                    scale = max_val / qmax  # [C, num_blocks]
                    zero_point_val = torch.zeros_like(scale)
                
                del max_val  # 🔥 及时释放显存
                
                # 量化每个 block
                scale_expanded = scale.unsqueeze(2)  # [C, num_blocks, 1]
                if unsigned:
                    zero_expanded = zero_point_val.unsqueeze(2)  # [C, num_blocks, 1]
                    quantized_blocked = torch.clamp(
                        torch.round(tensor_blocked / scale_expanded + zero_expanded),
                        qmin,
                        qmax
                    )
                else:
                    quantized_blocked = torch.clamp(
                        torch.round(tensor_blocked / scale_expanded),
                        qmin,
                        qmax
                    )
                del tensor_blocked, scale_expanded  # 🔥 及时释放显存
                
                # 恢复形状
                quantized = quantized_blocked.reshape(n_channels, -1)  # [C, num_blocks*block_size]
                del quantized_blocked  # 🔥 及时释放显存
                
                if pad_size > 0:
                    quantized = quantized[:, :-pad_size]  # 去除 padding
                quantized = quantized.reshape(shape)
                
                # scale 和 zero_point 保持 [C, num_blocks] 形状
                # 注意：这会改变 scale 的形状，需要在 dequantize 时相应处理
                if unsigned:
                    zero_point = zero_point_val  # [C, num_blocks]
                else:
                    zero_point = torch.zeros_like(scale)  # [C, num_blocks]
                
            else:
                # Per-channel (不分块): 每个 channel 独立计算 scale
                tensor_flat = tensor.reshape(shape[0], -1)  # [C, *]
                
                # 计算每个 channel 的百分位数（而非最大值）
                abs_tensor = tensor_flat.abs().float()
                
                device_orig = abs_tensor.device
                
                # 使用向量化的 quantile 计算（PyTorch 1.7+ 支持）
                n_channels, n_elements = abs_tensor.shape
                if n_elements > 1000000:  # 超过 100 万个元素才采样
                    indices = torch.randperm(n_elements, device=abs_tensor.device)[:1000000]
                    sampled = abs_tensor[:, indices]
                    del indices  # 🔥 及时释放显存
                    max_val = torch.quantile(sampled, percentile, dim=1)
                    del sampled  # 🔥 及时释放显存
                else:
                    # 小张量直接在原设备计算（避免传输开销）
                    max_val = torch.quantile(abs_tensor, percentile, dim=1)
                
                del abs_tensor  # 🔥 及时释放显存
                
                # 确保在原设备上
                if max_val.device != device_orig:
                    max_val = max_val.to(device_orig)
                max_val = max_val.unsqueeze(1)  # [C, 1]
                max_val = torch.clamp(max_val, min=1e-5)
                
                # 计算量化范围和 scale
                if unsigned:
                    qmin = 0
                    qmax = 2 ** n_bits - 1
                    min_val = tensor_flat.min(dim=1)[0].unsqueeze(1)  # [C, 1]
                    scale = (max_val - min_val) / qmax  # [C, 1]
                    zero_point_val = -min_val / scale  # [C, 1]
                else:
                    qmin = -(2 ** (n_bits - 1))
                    qmax = 2 ** (n_bits - 1) - 1
                    scale = max_val / qmax  # [C, 1]
                    zero_point_val = 0
                
                del max_val  # 🔥 及时释放显存
                
                # 量化
                if unsigned:
                    quantized = torch.clamp(
                        torch.round(tensor_flat / scale + zero_point_val),
                        qmin,
                        qmax
                    )
                else:
                    quantized = torch.clamp(
                        torch.round(tensor_flat / scale),
                        qmin,
                        qmax
                    )
                del tensor_flat  # 🔥 及时释放显存
                
                # 恢复形状
                quantized = quantized.reshape(shape)
                scale = scale.squeeze(1)  # [C]
                if unsigned:
                    zero_point = zero_point_val.squeeze(1)  # [C]
                else:
                    zero_point = torch.zeros_like(scale)  # [C]
            
            if channel_dim != 0:
                quantized = quantized.transpose(0, channel_dim)
        else:
            # Per-tensor: 整个张量一个 scale（随机采样）
            # 使用百分位数代替最大值
            abs_tensor = tensor.abs().flatten()
            # 确保张量是 float 类型（quantile 要求 float 或 double）
            if abs_tensor.dtype not in [torch.float32, torch.float64]:
                abs_tensor = abs_tensor.float()
            
            if abs_tensor.numel() > 0:
                # 🔥 随机采样策略：减少采样数量以加速
                if abs_tensor.numel() > 1000000:  # 超过 100 万个元素才采样
                    # 随机采样 50 万个元素（足够估计百分位数）
                    indices = torch.randperm(abs_tensor.numel(), device=abs_tensor.device)[:1000000]
                    sampled = abs_tensor[indices]
                    del indices  # 🔥 及时释放显存
                    max_val = torch.quantile(sampled, percentile)
                    del sampled  # 🔥 及时释放显存
                else:
                    max_val = torch.quantile(abs_tensor, percentile)
            else:
                max_val = torch.tensor(1e-5, device=tensor.device)
            
            del abs_tensor  # 🔥 及时释放显存
            max_val = torch.clamp(max_val, min=1e-5)
            
            # 计算量化范围
            if unsigned:
                qmin = 0
                qmax = 2 ** n_bits - 1
                tensor_flat = tensor.flatten()
                min_val = tensor_flat.min()
                scale = (max_val - min_val) / qmax
                zero_point = -min_val / scale
                del tensor_flat
            else:
                qmin = -(2 ** (n_bits - 1))
                qmax = 2 ** (n_bits - 1) - 1
                scale = max_val / qmax
                zero_point = torch.tensor(0.0, device=tensor.device)
            
            del max_val  # 🔥 及时释放显存
            
            # 量化
            if unsigned:
                quantized = torch.clamp(
                    torch.round(tensor / scale + zero_point),
                    qmin,
                    qmax
                )
            else:
                quantized = torch.clamp(
                    torch.round(tensor / scale),
                    qmin,
                    qmax
                )
        
        # 确保 zero_point 与 scale 形状匹配
        if not isinstance(zero_point, torch.Tensor):
            zero_point = torch.zeros_like(scale)
        
        # 🔥 类型一致性修复：确保 scale 和 zero_point 使用与输入 tensor 相同的 dtype
        # 但保持足够的精度（使用 float32 或 bfloat16，而不是 int）
        target_dtype = tensor.dtype if tensor.dtype in [torch.float32, torch.bfloat16, torch.float16] else torch.bfloat16
        scale = scale.to(dtype=target_dtype)
        zero_point = zero_point.to(dtype=target_dtype)
        
        # quantized 保持 int 类型（int8 或 int4）
        return quantized, scale, zero_point
    
    def gptq_quantize_tensor(
        self,
        tensor: torch.Tensor,
        inputs: torch.Tensor,  # 校准激活数据
        n_bits: int = 8,
        channel_dim: int = 0,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用 GPTQ 算法量化张量
        
        GPTQ: 基于 Optimal Brain Quantization (OBQ)
        - 使用 Hessian 矩阵 (二阶信息) 指导量化
        - 逐列量化并补偿误差到未量化列
        - 比 RTN 更精确，但需要校准数据
        
        Args:
            tensor: 要量化的张量 [out_features, in_features]
            inputs: 校准激活数据 [batch, in_features]
            n_bits: 量化位数
            channel_dim: 通道维度 (0=按行/输出通道)
            verbose: 是否输出详细信息
        
        Returns:
            quantized: 量化后的张量
            scale: 缩放因子
            zero_point: 零点
        """
        if verbose:
            print(f"  [GPTQ] Starting GPTQ quantization...")
        
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype
        # print("using gptq")
        # 🔥 处理不同维度的inputs
        if inputs.dim() == 3:
            # [num_prompts, batch, in_features] -> [num_prompts*batch, in_features]
            inputs = inputs.reshape(-1, inputs.shape[-1])
            if verbose:
                print(f"  [GPTQ] Reshaped 3D inputs to 2D: {inputs.shape}")
        elif inputs.dim() != 2:
            raise ValueError(f"Expected inputs to be 2D or 3D, got {inputs.dim()}D")
        
        # 🔥 移除显存检查限制，完全按照配置执行 GPTQ
        
        # 🔥 修复：不转置，保持 [out_features, in_features] 顺序
        # Deep Compressor 通过 view_shape 处理，不需要转置
        # 🔥 类型修复：GPTQ 需要 float32 进行精确计算
        W = tensor.clone().float()  # [out_features, in_features], float32
        
        num_rows, num_cols = W.shape  # out_features, in_features
        
        # 🔥 移除层大小限制，完全按照配置执行 GPTQ
        if verbose:
            print(f"  [GPTQ] Layer size: {num_rows}x{num_cols}")
        
        # ========== Step 1: 计算 Hessian 矩阵 ==========
        # H = 2 * X^T @ X / num_samples
        # Hessian 是 [in_features, in_features]，对应权重的列（输入维度）
        if verbose:
            print(f"  [GPTQ] Computing Hessian matrix ({num_cols}x{num_cols})...")
            print(f"  [GPTQ] Activation data shape: {inputs.shape}")
        
        # 🔥 优化：增大 Hessian 批次大小以提升性能（如果显存允许）
        num_samples = inputs.shape[0]
        # 根据显存动态调整批次大小
        if device.type == 'cuda':
            free_memory = torch.cuda.mem_get_info(device.index)[0] / 1024**3  # GB
            # 如果显存充足，使用更大的批次
            if free_memory > 10:  # 超过 10GB 可用
                hessian_batch_size = min(128, num_samples)  # 增大批次
            elif free_memory > 5:  # 5-10GB
                hessian_batch_size = min(64, num_samples)
            else:
                hessian_batch_size = min(32, num_samples)  # 显存紧张时用小批次
        else:
            hessian_batch_size = min(64, num_samples)
        
        # 🔥 在CPU上计算Hessian（如果GPU显存不足）
        use_cpu_for_hessian = False
        if device.type == 'cuda':
            free_memory = torch.cuda.mem_get_info(device.index)[0] / 1024**3
            hessian_size_gb = num_cols * num_cols * 4 / 1024**3  # float32
            if free_memory < hessian_size_gb * 3:  # Hessian + H_inv + buffer
                use_cpu_for_hessian = True
                if verbose:
                    print(f"  [GPTQ] Using CPU for Hessian computation (GPU memory low)")
        
        H_device = torch.device('cpu') if use_cpu_for_hessian else device
        # 🔥 类型修复：Hessian 计算使用 float32 以获得精度，但最终 scales 会转换为 bfloat16
        H = torch.zeros((num_cols, num_cols), device=H_device, dtype=torch.float32)
        
        # 分批计算Hessian
        for i in range(0, num_samples, hessian_batch_size):
            batch = inputs[i:i+hessian_batch_size]
            # 🔥 类型修复：确保 batch 是 float32
            if batch.dtype != torch.float32:
                batch = batch.float()
            # 🔥 临时移到正确的设备
            if use_cpu_for_hessian:
                batch = batch.cpu()
            H += torch.matmul(batch.t(), batch) * (2.0 / num_samples)
            del batch  # 🔥 立即释放
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 🔥 清理inputs引用（不再需要）
        del inputs
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 处理 dead channels
        # H 是 [in_features, in_features]，W 是 [out_features, in_features]
        dead = H.diagonal() == 0
        H[dead, dead] = 1
        W[:, dead] = 0  # 🔥 修复：dead 对应的是 in_features（W 的列）
        
        # ========== Step 2: 计算scales (BEFORE sorting! - 关键修复) ==========
        # 🔥 必须在排序前计算scale，基于原始权重的group划分
        # 正确的维度：W = [out_features, in_features]
        # Group 划分在 in_features（列）维度上
        qmax = 2 ** (n_bits - 1) - 1
        group_size = self.gptq_group_size if hasattr(self, 'gptq_group_size') else num_cols
        
        if verbose:
            print(f"  [GPTQ] Computing scales BEFORE sorting...")
            print(f"  [GPTQ] W shape: {W.shape} = [out_features={num_rows}, in_features={num_cols}]")
            print(f"  [GPTQ] group_size: {group_size}, qmax: {qmax}")
        
        if group_size < num_cols:
            # 🔥 Per-group quantization: 在 in_features（列）维度上分组
            num_groups = (num_cols + group_size - 1) // group_size
            # 🔥 类型修复：GPTQ 内部计算使用 float32，最后会转换回原始 dtype
            scales = torch.zeros(num_rows, num_groups, device=device, dtype=torch.float32)
            
            if verbose:
                print(f"  [GPTQ] Per-group quantization: num_groups={num_groups}")
                print(f"  [GPTQ] scales shape: [{num_rows}, {num_groups}] = [out_features, num_groups]")
            
            # W 还未排序，是 [out_features, in_features_original]
            # 🔥 性能优化：向量化计算所有 groups 的 scales
            # 将 W reshape 成 [out_features, num_groups, group_size]
            W_padded = W
            if num_cols % group_size != 0:
                # 需要 padding 到整数个 groups
                pad_size = num_groups * group_size - num_cols
                W_padded = F.pad(W, (0, pad_size), mode='constant', value=0)
            
            W_grouped = W_padded.reshape(num_rows, num_groups, group_size)  # [out_features, num_groups, group_size]
            # 向量化计算每个 group 的最大值
            group_max = W_grouped.abs().max(dim=2)[0]  # [out_features, num_groups]
            group_max = torch.clamp(group_max, min=1e-5)
            scales = (group_max / qmax).float()  # [out_features, num_groups]
            
            del W_padded, W_grouped, group_max  # 清理临时变量
            
            if verbose:
                print(f"  [GPTQ] scales range: [{scales.min():.6f}, {scales.max():.6f}]")
        else:
            # Per-channel quantization: 每个输出通道一个 scale
            W_abs_max = W.abs().max(dim=1)[0]  # [out_features]
            scales = W_abs_max / qmax
            scales = torch.clamp(scales, min=1e-5)
            # 🔥 类型修复：GPTQ 内部计算使用 float32
            scales = scales.unsqueeze(1).float()  # [out_features, 1], float32
            
            if verbose:
                print(f"  [GPTQ] Per-channel quantization")
                print(f"  [GPTQ] scales shape: {scales.shape}")
        
        # ========== Step 3: 按重要性排序 (Actorder) ==========
        if verbose:
            print(f"  [GPTQ] Sorting by importance (actorder)...")
        
        # 🔥 标准GPTQ actorder: 按Hessian对角线元素排序
        # Hessian 是 [in_features, in_features]，对应权重的列
        importance = H.diagonal()
        perm = torch.argsort(importance, descending=True)
        
        # 🔥 排序 H 和 W 的列（in_features 维度）
        H_old = H
        W_old = W
        H = H[perm][:, perm]  # 排序 Hessian 的行和列
        W = W[:, perm]  # 🔥 修复：排序 W 的列（in_features）
        del H_old, W_old, importance  # 🔥 立即删除旧tensor
        
        inv_perm = torch.argsort(perm)
        
        if verbose:
            print(f"  [GPTQ] Permutation range: [{perm.min()}, {perm.max()}]")
        
        # ========== Step 4: Damping ==========
        H_diag = H.diagonal()
        H_diag_mean = H_diag.mean()
        damp = self.gptq_damp_percentage * H_diag_mean
        H[range(num_cols), range(num_cols)] += damp  # 🔥 修复：H 是 [num_cols, num_cols]
        del H_diag  # 🔥 清理
        
        # ========== Step 5: 计算 Hessian 逆矩阵 ==========
        if verbose:
            print(f"  [GPTQ] Computing Hessian inverse...")
        
        H_inv = None
        for attempt in range(self.gptq_num_inv_tries):
            try:
                # Cholesky 分解求逆
                L = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(L)
                del L  # 🔥 立即释放
                H_inv = torch.linalg.cholesky(H_inv, upper=True)
                break
            except RuntimeError as e:
                # 增加 damping 重试
                H[range(num_cols), range(num_cols)] += 0.001 * H_diag_mean  # 🔥 修复
                if attempt == self.gptq_num_inv_tries - 1:
                    # 🔥 移除自动回退，按照配置执行，失败时抛出异常
                    if verbose:
                        print(f"\n⚠️  [GPTQ] Hessian inversion failed after {self.gptq_num_inv_tries} tries")
                        print(f"   Error: {e}")
                    del H
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    # 抛出异常而不是回退
                    raise RuntimeError(f"GPTQ Hessian inversion failed after {self.gptq_num_inv_tries} tries: {e}")
        
        del H_diag_mean  # 🔥 清理
        
        # 🔥 如果在CPU上计算，移回GPU
        if use_cpu_for_hessian and device.type == 'cuda':
            if verbose:
                print(f"  [GPTQ] Moving H_inv back to GPU...")
            H_inv = H_inv.to(device)
        
        # 🔥 清理H（不再需要）
        del H
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # ========== Step 6: 逐块 GPTQ 量化 ==========
        # 🔥 动态调整block_size以节省显存
        effective_block_size = self.gptq_block_size
        if device.type == 'cuda':
            free_memory = torch.cuda.mem_get_info(device.index)[0] / 1024**3
            if free_memory < 1.0:  # 小于1GB可用
                effective_block_size = min(32, self.gptq_block_size)
                if verbose:
                    print(f"  [GPTQ] Low memory, reducing block size to {effective_block_size}")
        
        if verbose:
            print(f"  [GPTQ] Quantizing in blocks of {effective_block_size}...")
        
        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)
        
        # 🔥 修复：循环应该遍历列（in_features）
        for i_start in range(0, num_cols, effective_block_size):
            i_end = min(i_start + effective_block_size, num_cols)
            block_W = W[:, i_start:i_end].clone()  # [out_features, block_size]
            block_Q = Q[:, i_start:i_end]  # [out_features, block_size]
            block_Hinv = H_inv[i_start:i_end, i_start:i_end]  # [block_size, block_size]
            block_err = Err[:, i_start:i_end]  # [out_features, block_size]
            
            # 逐列量化
            for i in range(i_end - i_start):
                idx = i_start + i  # sorted后的列索引
                w_col = block_W[:, i]  # [out_features]
                h_inv_diag = block_Hinv[i, i]
                
                # 🔥 防止除零
                if h_inv_diag <= 0 or torch.isnan(h_inv_diag) or torch.isinf(h_inv_diag):
                    h_inv_diag = 1.0
                
                # 🔥 确定该列属于哪个 group (基于原始位置!)
                if scales.dim() == 2:  # per-group
                    # 🔥 性能优化：预先计算所有列的 group_idx，避免每次从 CPU 读取
                    if not hasattr(self, '_gptq_group_idx_cache'):
                        # 缓存 perm 和 group_idx 映射
                        perm_tensor = perm.to(device)  # 移到 GPU
                        self._gptq_group_idx_cache = (perm_tensor // group_size).to(device)  # [num_cols]
                    group_idx = self._gptq_group_idx_cache[idx]  # GPU 上的索引
                    scale = scales[:, group_idx]  # [out_features]
                else:  # per-channel
                    scale = scales.squeeze(1)  # [out_features]
                
                # 量化当前列
                q_col = torch.clamp(
                    torch.round(w_col / scale),
                    -qmax - 1,
                    qmax
                )
                block_Q[:, i] = q_col  # 🔥 修复：赋值到列
                
                # 反量化并计算误差
                dequant_col = q_col * scale
                err = (w_col - dequant_col) / h_inv_diag
                
                # 🔥 检查误差是否异常
                if torch.isnan(err).any() or torch.isinf(err).any():
                    err = torch.zeros_like(err)
                
                block_err[:, i] = err  # 🔥 修复：赋值到列
                
                # 补偿误差到剩余列
                if i < i_end - i_start - 1:
                    # err: [out_features], block_Hinv[i, i+1:]: [remaining_cols]
                    # block_W[:, i+1:]: [out_features, remaining_cols]
                    block_W[:, i+1:] -= err.unsqueeze(1) * block_Hinv[i, i+1:].unsqueeze(0)
            
            # 跨块误差传播
            if i_end < num_cols:
                # block_err: [out_features, block_size]
                # H_inv[i_start:i_end, i_end:]: [block_size, remaining_cols]
                # 需要: W[:, remaining_cols] -= block_err @ H_inv[block, remaining]
                W[:, i_end:] -= torch.matmul(block_err, H_inv[i_start:i_end, i_end:])
            
            # 🔥 清理块级临时变量
            del block_W, block_Hinv, block_err
            
            # 🔥 减少缓存清理频率以提升性能（每20个块清理一次）
            if (i_start // effective_block_size) % 20 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 🔥 清理中间矩阵
        del H_inv, W, Err
        
        # 强制垃圾回收
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # ========== Step 7: 恢复原始顺序 ==========
        # 🔥 创建g_idx: 对于每个原始in_features位置，记录它属于哪个group（基于原始位置）
        if scales.dim() == 2:  # per-group
            # 🔥 性能优化：向量化创建 g_idx
            g_idx = torch.arange(num_cols, dtype=torch.int32, device=device) // group_size
        else:
            g_idx = None
        
        # 🔥 清理缓存
        if hasattr(self, '_gptq_group_idx_cache'):
            del self._gptq_group_idx_cache
        
        # 恢复Q的列顺序
        Q_old = Q
        Q = Q[:, inv_perm].clone()  # 🔥 修复：恢复列的顺序
        del Q_old, inv_perm, perm  # 🔥 立即释放
        
        # 🔥 修复：不需要转置，Q 已经是正确的 [out_features, in_features] 形状
        
        if verbose:
            print(f"  [GPTQ] Final Q shape: {Q.shape} = [out_features={num_rows}, in_features={num_cols}]")
            print(f"  [GPTQ] Final scales shape: {scales.shape}")
            if g_idx is not None:
                print(f"  [GPTQ] g_idx shape: {g_idx.shape}, range: [{g_idx.min()}, {g_idx.max()}]")
            print(f"  [GPTQ] Q range: [{Q.min():.2f}, {Q.max():.2f}]")
            print(f"  [GPTQ] Quantization quality check:")
            # 重建权重检查误差
            if g_idx is not None:
                # Per-group with actorder: 使用g_idx
                dequant_test = torch.zeros_like(Q.float())
                for i in range(Q.shape[1]):  # 遍历每个in_feature（列）
                    group_idx = g_idx[i].item()
                    dequant_test[:, i] = Q[:, i].float() * scales[:, group_idx]
            elif scales.dim() == 2:
                # Per-group without actorder
                dequant_test = torch.zeros_like(Q.float())
                for g in range(scales.shape[1]):
                    g_start = g * group_size
                    g_end = min((g + 1) * group_size, Q.shape[1])
                    dequant_test[:, g_start:g_end] = Q[:, g_start:g_end].float() * scales[:, g].unsqueeze(1)
            else:
                # Per-channel: scales 是 [out_features, 1]
                dequant_test = Q.float() * scales  # Broadcasting: [out, in] * [out, 1]
            
            reconstruction_error = (dequant_test - tensor.float()).abs().mean()
            print(f"  [GPTQ] Mean reconstruction error: {reconstruction_error:.6f}")
            del dequant_test
        
        # 🔥 类型修复：Q 应该保持 int 类型（量化后的值），而不是转换为 float
        # scales 和 zero_point 应该使用 bfloat16（与模型其他部分一致）
        target_dtype = dtype if dtype in [torch.float32, torch.bfloat16, torch.float16] else torch.bfloat16
        # Q 保持 int 类型（在 dequantize 时会转换为 float）
        # scales 和 zero_point 使用目标 dtype
        scales = scales.to(dtype=target_dtype)
        zero_point = torch.zeros_like(scales)
        
        # 🔥 保存g_idx用于反量化（如果使用了per-group + actorder）
        if g_idx is not None:
            # 🔥 优化：使用int16并移到CPU节省显存
            # g_idx只在反量化时需要，移到CPU不影响性能
            self.gptq_g_idx = g_idx.to(torch.int16).cpu()
        else:
            self.gptq_g_idx = None
        
        # 🔥 最终清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if verbose:
            print(f"  [GPTQ] Quantization completed")
            if self.gptq_g_idx is not None:
                print(f"  [GPTQ] Saved g_idx for actorder dequantization")
        
        return Q, scales, zero_point
    
    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        is_blocked: bool = False,  # scale 是否是分块的
        block_size: int = 128,  # 分块大小
    ) -> torch.Tensor:
        """
        反量化
        
        Args:
            quantized: 量化后的张量 [out_features, in_features] (int 类型)
            scale: 缩放因子 (bfloat16 或 float32)
                - per-channel: [out_features]
                - per-block: [out_features, num_blocks]
            zero_point: 零点 (bfloat16 或 float32)
            is_blocked: scale 是否是分块的
            block_size: 分块大小
        
        Returns:
            dequantized: 反量化后的张量 (bfloat16)
        """
        # 🔥 设备兼容性修复：确保 scale 和 zero_point 与 quantized 在同一设备
        # 这在多GPU环境下（如使用 accelerate）非常重要
        device = quantized.device
        if scale.device != device:
            scale = scale.to(device)
        if zero_point.device != device:
            zero_point = zero_point.to(device)
        
        # 🔥 类型修复：确保 quantized 转换为 float 进行计算，scale 和 zero_point 保持原类型
        # 最终返回 bfloat16 以保持一致性
        if is_blocked and len(scale.shape) == 2:
            # 分块反量化：scale 是 [C, num_blocks]
            out_features, in_features = quantized.shape
            num_blocks = scale.shape[1]
            
            # Pad 到整数个 blocks
            n_elements = in_features
            expected_elements = num_blocks * block_size
            pad_size = expected_elements - n_elements
            
            if pad_size > 0:
                quantized = F.pad(quantized, (0, pad_size), mode='constant', value=0)
            
            # Reshape 成 [C, num_blocks, block_size]
            quantized_blocked = quantized.reshape(out_features, num_blocks, block_size)
            
            # Expand scale 和 zero_point
            if len(zero_point.shape) == 2:
                zero_point_expanded = zero_point.unsqueeze(2)  # [C, num_blocks, 1]
            else:
                zero_point_expanded = zero_point.unsqueeze(1).unsqueeze(2)  # [C, 1, 1]
            
            scale_expanded = scale.unsqueeze(2)  # [C, num_blocks, 1]
            
            # 反量化：确保 quantized 转换为 float 进行计算
            dequantized_blocked = (quantized_blocked.float() - zero_point_expanded) * scale_expanded
            del quantized_blocked, zero_point_expanded, scale_expanded  # 🔥 及时释放显存
            
            # 恢复形状
            dequantized = dequantized_blocked.reshape(out_features, -1)
            del dequantized_blocked  # 🔥 及时释放显存
            
            if pad_size > 0:
                dequantized = dequantized[:, :-pad_size]
            
            # 🔥 类型修复：确保返回 bfloat16
            return dequantized.to(torch.bfloat16)
        else:
            # 标准反量化（per-tensor 或 per-channel）
            # 🔥 类型修复：确保 quantized 转换为 float，最终返回 bfloat16
            dequantized = (quantized.float() - zero_point) * scale
            return dequantized.to(torch.bfloat16)
    
    def prepare_weight(
        self,
        activation_max: Optional[torch.Tensor] = None,
        activation_data: Optional[torch.Tensor] = None,  # 🔥 用于 GPTQ 的校准数据
        layer_name: Optional[str] = None,  # 🔥 层名（用于自动查找激活数据）
        verbose: bool = False,
    ):
        """
        准备权重：应用 SmoothQuant、SVD、异常值提取、量化
        
        🔥 新的执行顺序（用户要求）：
        1. SmoothQuant: W' = W * s (先 smooth 整个权重)
        2. SVD: W' = U @ S @ V + R_svd (先做低秩分解)
        3. Outlier: 从 R_svd 中提取异常值 (对 SVD 残差提取异常值)
           - 存储为稀疏格式：indices + values (节省显存)
           - 剩余权重：R_final = R_svd - W_outlier
        4. Quantize: Quant(R_final) 或 GPTQ
        
        前向传播重建顺序：
        W = U @ S @ V + W_outlier + Dequant(R_final)
        
        Args:
            activation_max: 激活值最大值统计 [in_features]（用于 SmoothQuant）
            activation_data: 完整激活数据 [num_samples, in_features]（用于 GPTQ）
            layer_name: 层名（如果提供，会自动从缓存中查找激活数据）
            verbose: 是否输出详细信息
        """
        # 🔥 直接使用传入的激活数据（从 Stage 实例变量传递）
        local_activation_max = activation_max
        local_activation_data = activation_data
        
        # 🔥 统一激活数据处理：如果有完整激活数据但没有 activation_max，自动计算
        # 这样 SmoothQuant、GPTQ 和 extract_outliers 可以共享同一份数据
        if local_activation_max is None and local_activation_data is not None:
            # 从完整激活数据计算 channel_max
            # activation_data shape: [num_samples, in_features]
            local_activation_max = local_activation_data.abs().max(dim=0)[0]  # [in_features]
            local_activation_max = local_activation_max.to(self.weight.device, dtype=self.weight.dtype)
            if verbose and self.use_smoothquant:
                print(f"  [SmoothQuant] Auto-computed activation_max from activation_data")
                print(f"    activation_max range: [{local_activation_max.min().item():.6f}, {local_activation_max.max().item():.6f}]")
        if self.use_gptq:
            has_activation_data = local_activation_data is not None
            print(f"[DEBUG] use_gptq=True, has_activation_data={has_activation_data}, layer_name={layer_name}")
        if verbose:
            print(f"\nPreparing weight for {self.__class__.__name__}:")
            print(f"  Config: sparse={self.use_sparse}, smooth={self.use_smoothquant}, "
                  f"svd={self.use_svd}, w_bit={self.weight_bit}, a_bit={self.act_bit}")
        
        current_weight = self.weight.data.clone()
        
        # Step 1: SmoothQuant（如果启用）- 🔥 必须先 smooth！
        if self.use_smoothquant:
            if local_activation_max is None:
                # 🔥 更友好的错误处理：自动禁用而不是崩溃
                if verbose:
                    print(f"  [Warning] SmoothQuant enabled but no activation_max provided.")
                    print(f"  [Warning] Disabling SmoothQuant for this layer.")
                self.use_smoothquant = False
            else:
                # 计算权重每个 input channel 的最大值
                weight_max = current_weight.abs().max(dim=0)[0]  # [in_features]
                
                # 计算缩放因子
                scales = self.compute_smoothquant_scales(local_activation_max, weight_max)
                # 🔥 类型修复：确保保存为 bfloat16
                self.smooth_scales = scales.to(torch.bfloat16)
                
                # 🔥 先对整个权重应用 smooth
                current_weight = self.apply_smoothquant(current_weight, scales)
                
                if verbose:
                    print(f"  [SmoothQuant] Applied scales to entire weight")
                
                # 🔥 立即清理不再需要的变量
                del weight_max, scales
        elif self.use_awq:
            if local_activation_data is not None:
                scales = self.compute_awq_scales(current_weight, local_activation_data, verbose=verbose)
                self.smooth_scales = scales.to(torch.bfloat16)
                current_weight = self.apply_smoothquant(current_weight, scales)
                if verbose:
                    print(f"  [AWQ] Applied scales to entire weight")
                del scales
            else:
                if verbose:
                    print(f"  [Warning] AWQ enabled but no activation_data provided, disabling.")
                self.use_awq = False
        
        # Step 2: SVD（如果启用）- 🔥 新顺序：先做 SVD
        if self.use_svd and self.svd_rank > 0:
            try:
                U, S, V, residual = self.compute_svd(current_weight, self.svd_rank, verbose=verbose)
                self.svd_U = U
                self.svd_S = S
                self.svd_V = V
                
                # 🔥 关键优化：直接计算并缓存SVD权重矩阵，避免forward时重复计算！
                # svd_weight = U @ diag(S) @ V^T = (U * S) @ V
                self._cached_svd_weight = ((U * S.unsqueeze(0)) @ V).to(torch.bfloat16).detach()
                if verbose:
                    print(f"  ✓ SVD weight matrix pre-computed and cached")
                
                # 🔥 SVD 残差将用于后续的 outlier 提取和量化
                current_weight = residual  # 后续操作在 SVD 残差上
                
                # 清理 SVD 相关的临时变量
                del U, S, V, residual
            except Exception as e:
                # SVD失败时的降级处理
                if verbose:
                    print(f"  [Warning] SVD failed: {e}")
                    print(f"  [Warning] Disabling SVD for this layer.")
                self.use_svd = False
                self._cached_svd_weight = None
        else:
            # 如果不启用SVD或rank为0，清空缓存
            self._cached_svd_weight = None
        
        # Step 3: 异常值提取（如果启用）- 🔥 新顺序：对 SVD 残差提取异常值
        if self.use_sparse:
            outlier_mask, outlier_values, residual = self.extract_outliers(
                current_weight,  # 🔥 这里是 SVD 残差（或 smooth 后的权重，如果没有 SVD）
                local_activation_max,
                verbose=verbose
            )
            
            # 🔥 使用稀疏格式存储异常值，节省显存
            # 只存储非零位置的索引和值
            outlier_indices = outlier_mask.nonzero(as_tuple=False)  # [num_outliers, 2]
            
            # 保存稀疏格式：indices 和 values
            # 🔥 类型修复：确保 outlier_values 使用 bfloat16
            self.register_buffer('outlier_indices', outlier_indices)
            self.register_buffer('outlier_values', outlier_values.to(torch.bfloat16))
            self.register_buffer('outlier_shape', torch.tensor(current_weight.shape))
            
            # 🔥 不再保存密集的 outlier_weight，节省显存
            # self.outlier_weight = None  # 标记为使用稀疏格式
            
            # 🔥 不保存 weight_residual，它只是中间变量，会在后续被量化
            # self.weight_residual = residual  # 不需要保存，浪费显存
            current_weight = residual  # 后续操作在残差上（去除 outlier 后）
            
            if verbose:
                num_outliers = outlier_indices.shape[0]
                total_elements = current_weight.numel()
                sparsity = 100 * (1 - num_outliers / total_elements)
                memory_saved = (total_elements - num_outliers) * 2  # BF16 = 2 bytes
                print(f"  [Sparse] {num_outliers}/{total_elements} outliers ({100-sparsity:.2f}%)")
                print(f"  [Sparse] Memory saved: {memory_saved / 1024 / 1024:.2f} MB")
                
                # 警告：异常值太多会影响性能
                if num_outliers > 500000:
                    print(f"  ⚠️  WARNING: Too many outliers ({num_outliers})!")
                    print(f"  ⚠️  This will increase memory usage during inference.")
                    print(f"  ⚠️  Consider increasing sparse_outlier_threshold or reducing sparse_ratio.")
            
            # 🔥 立即清理不再需要的变量
            del outlier_mask, outlier_values, residual, outlier_indices
        
        # Step 4: 量化（RTN 或 GPTQ）
        if verbose:
            if self.use_gptq:
                print(f"  [Quant] Using GPTQ quantization to {self.weight_bit}-bit...")
            elif self.use_block_quant:
                print(f"  [Quant] Quantizing weight to {self.weight_bit}-bit with block quantization (block_size={self.block_size_weight}, percentile={self.quant_percentile})...")
            else:
                print(f"  [Quant] Quantizing weight to {self.weight_bit}-bit (percentile={self.quant_percentile})...")
        
        # 使用 GPTQ 或 RTN
        if self.use_gptq:
            if local_activation_data is None:
                # 🔥 提供详细的错误信息，帮助用户诊断问题
                error_parts = ["GPTQ enabled but no activation_data available."]
                
                if layer_name is None:
                    error_parts.append("- layer_name is None: Cannot auto-load activation data without layer_name.")
                    error_parts.append("  Solution: Pass layer_name to prepare_weight()")
                elif self._activation_config is None and self._cached_full_activations is None:
                    error_parts.append("- No activation config set and no cached activations.")
                    error_parts.append("  Solution: Call HybridQuantLinear.set_activation_config(path) before quantization")
                elif self._cached_full_activations is not None and layer_name not in self._cached_full_activations:
                    error_parts.append(f"- layer_name '{layer_name}' not found in cached activations.")
                    error_parts.append(f"  Available layers: {len(self._cached_full_activations)} in cache")
                else:
                    error_parts.append("- Unknown reason")
                
                error_msg = "\n".join(error_parts)
                if verbose:
                    print(f"  [Error] {error_msg}")
                raise ValueError(error_msg)
            else:
                # 🔥 在调用GPTQ前主动清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
                    if verbose:
                        print(f"  [Quant] Free GPU memory before GPTQ: {free_mem:.2f}GB")
                # print("using gptq in prepare weight")
                try:
                    # 🔥 将激活数据移到正确的设备并确保类型一致（在GPTQ内部会处理）
                    # 激活数据需要与权重在同一设备，但保持 float32 用于 Hessian 计算
                    if local_activation_data.device != self.weight.device:
                        activation_data_gpu = local_activation_data.to(device=self.weight.device, dtype=torch.float32)
                    else:
                        # 确保是 float32（GPTQ 内部需要）
                        activation_data_gpu = local_activation_data.float() if local_activation_data.dtype != torch.float32 else local_activation_data
                    
                    # 使用 GPTQ 量化
                    quantized, scale, zero_point = self.gptq_quantize_tensor(
                        current_weight,
                        activation_data_gpu,
                        n_bits=self.weight_bit,
                        channel_dim=0,
                        verbose=verbose,
                    )
                    
                    # 🔥 立即清理GPU副本
                    if local_activation_data.device != self.weight.device:
                        del activation_data_gpu
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # 🔥 移除自动回退，按照配置执行，失败时抛出异常
                    if verbose or 'out of memory' in str(e).lower():
                        print(f"\n⚠️  GPTQ failed: {str(e)[:150]}")
                    # 清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 抛出异常而不是回退
                    raise RuntimeError(f"GPTQ quantization failed: {e}")
        else:
            # 使用 RTN 量化
            quantized, scale, zero_point = self.quantize_tensor(
                current_weight,
                self.weight_bit,
                per_channel=True,
                channel_dim=0,  # output channel
                percentile=self.quant_percentile,
                use_block=self.use_block_quant,  # 🔥 传递分块量化开关
                block_size=self.block_size_weight,  # 🔥 传递权重的分块大小
                unsigned=False,  # 权重使用对称量化
            )
        
        self.quantized_weight = quantized.to(torch.int8)
        # 🔥 类型修复：确保 scale 和 zero_point 使用 bfloat16 以保持一致性
        self.weight_scale = scale.to(torch.bfloat16)
        self.weight_zero_point = zero_point.to(torch.bfloat16)
        
        # 🔥 关键：删除原始权重以节省显存
        del self.weight
        self.weight = None  # 确保不会意外访问
        
        # 🔥 清理临时变量
        del current_weight, quantized, scale, zero_point
        
        # 🔥 清理临时加载的激活数据（无论成功还是失败都要清理）
        # 清理临时 GPU 副本（如果有）
        if 'activation_data_gpu' in locals():
            del activation_data_gpu
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 🔥 量化阶段不缓存反量化权重（节省显存）
        self._cached_dequant_weight = None
        # 🔥 注意：_cached_svd_weight 在 SVD 步骤中已经计算好了，这里不清空
        
        # 🔥 强制垃圾回收（针对大型权重矩阵）
        import gc
        gc.collect()
        
        if verbose:
            print(f"  ✓ Weight preparation completed")
            print(f"  ✓ Original weight deleted to save memory")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        新的执行顺序（匹配 prepare_weight 的改变）：
        1. [SmoothQuant] X' = X / s (对激活值应用 smooth)
        2. [量化] Quant(X') (可选的激活值量化)
        3. [重建权重] W = SVD + Outlier + Dequant(R_final)
           准备顺序：W' → SVD(W') = U·S·V^T + R_svd → Outlier(R_svd) = W_outlier + R_final → Quant(R_final)
           重建顺序：W = U·S·V^T + W_outlier + Dequant(R_final)
        4. [矩阵乘法] Y = X' @ W
        
        🔥 注意：
        - SVD 先加（因为 SVD 是对完整权重做的）
        - Outlier 后加（因为 outlier 是从 SVD 残差中提取的）
        - 最后加量化残差
        
        数学验证：
          W = U·S·V^T + W_outlier + Dequant(quantized)
          Y = (X / s) @ (W * s) = X @ W ✓
        
        🔥 所有输入输出强制使用 BFloat16
        🔥 显存优化：异常值矩阵永远不会被完整重建
        """
        # 🔥 强制输入为 BFloat16
        x = x.to(torch.bfloat16)
        
        # Step 1: SmoothQuant / AWQ 激活值
        if (self.use_smoothquant or self.use_awq) and self.smooth_scales is not None:
            # X' = X / s （逐元素除）
            # 🔥 确保 smooth_scales 在正确的设备上
            smooth_scales_bf16 = self.smooth_scales.to(device=x.device, dtype=torch.bfloat16)
            # print("smooth_scales_bf16",smooth_scales_bf16)
            x = x / smooth_scales_bf16.unsqueeze(0)
        
        # 🔥 关键修复：如果使用 SVD，先保存未量化的激活
        # SVD 部分需要高精度激活（smooth后但未量化）
        x_smooth_unquant = None
        if self.use_svd and self.svd_U is not None:
            x_smooth_unquant = x.clone()  # 保存用于 SVD 的高精度激活
        
        # Step 2: 量化激活值（推理时，支持块量化和 unsigned）
        # 🔥 注意：量化后的激活只用于残差部分（已量化的权重）
        if not self.training and self.act_bit < 16:
            # 🔥 DEBUG: 打印激活量化信息（仅第一次）
            if not hasattr(self, '_act_quant_logged'):
                print(f"  [DEBUG] Activation quantization ENABLED: act_bit={self.act_bit}, training={self.training}")
                self._act_quant_logged = True
            # 🔥 对 activation 应用块量化和 unsigned 量化（SVDQuant 推荐）
            x_quantized, act_scale, act_zero = self.quantize_tensor(
                x, 
                self.act_bit, 
                per_channel=True,
                channel_dim=0,  # activation 的 channel 维度
                percentile=self.quant_percentile,
                use_block=self.use_block_quant_act,  # 🔥 使用独立的激活块量化开关
                block_size=self.block_size_act,  # 🔥 使用激活的分块大小
                unsigned=self.act_unsigned,  # 🔥 SVDQuant: unsigned INT4 (0-15)
            )
            
            # 🔥 反量化时需要考虑块量化
            is_act_blocked = (self.use_block_quant_act and len(act_scale.shape) == 2)
            
            # 🔥 类型修复：确保 act_scale 和 act_zero 使用 bfloat16
            act_scale = act_scale.to(torch.bfloat16)
            act_zero = act_zero.to(torch.bfloat16)
            
            if is_act_blocked:
                # 分块量化：scale 是 [in_features, num_blocks]
                x = self.dequantize_tensor(
                    x_quantized.float(),
                    act_scale,  # [in_features, num_blocks]
                    act_zero if len(act_zero.shape) == 2 else act_zero.unsqueeze(1),
                    is_blocked=True,
                    block_size=self.block_size_act,
                )
            else:
                # 标准量化：scale 是 [in_features]
                x = self.dequantize_tensor(
                    x_quantized.float(),
                    act_scale.unsqueeze(1),  # [in_features, 1]
                    act_zero.unsqueeze(1),
                    is_blocked=False,
                    block_size=self.block_size_act,
                )
            
            # 🔥 dequantize_tensor 已经返回 bfloat16，这里不需要再次转换
            # x = x.to(torch.bfloat16)  # 已由 dequantize_tensor 保证
        else:
            # 🔥 DEBUG: A16 时跳过激活量化
            if not hasattr(self, '_act_skip_logged') and self.act_bit >= 16:
                print(f"  [DEBUG] Activation quantization SKIPPED: act_bit={self.act_bit}, using FP16")
                self._act_skip_logged = True
        
        # Step 3: 计算输出
        # 🔥 关键修复：SVD 和残差分开计算，使用不同精度的激活
        if self.quantized_weight is not None:
            # 🔥 Step 3.1: 先计算 SVD 低秩部分（使用未量化的激活）
            output = None
            if self.use_svd and self.svd_U is not None and x_smooth_unquant is not None:
                # 🔥 直接使用prepare_weight时预先计算好的SVD权重！
                if self._cached_svd_weight is not None:
                    svd_weight = self._cached_svd_weight.to(device=x.device)
                else:
                    # 训练模式或缓存失效，需要重建（不应该发生）
                    svd_U_bf16 = self.svd_U.to(device=x.device, dtype=torch.bfloat16)
                    svd_S_bf16 = self.svd_S.to(device=x.device, dtype=torch.bfloat16)
                    svd_V_bf16 = self.svd_V.to(device=x.device, dtype=torch.bfloat16)
                    svd_weight = ((svd_U_bf16 * svd_S_bf16.unsqueeze(0)) @ svd_V_bf16).to(torch.bfloat16)
                
                # 🔥 使用未量化的激活计算 SVD 部分
                output = F.linear(x_smooth_unquant, svd_weight, None)
                output = output.to(torch.bfloat16)
                
                # 注意：不delete svd_weight，因为它是缓存的引用
            
            # 🔥 Step 3.2: 计算残差部分（使用量化的激活）
            
            # 🔥 使用缓存避免每次重建权重（仅当启用缓存且推理时）
            if (self.cache_dequant_weight and not self.training and 
                self._cached_dequant_weight is not None and 
                self._weight_cache_dtype == torch.bfloat16):
                residual_weight = self._cached_dequant_weight
            else:
                # 反量化残差权重
                # 🔥 区分三种情况：
                # 1. GPTQ per-group: scales 是 [out_features, num_groups]
                # 2. RTN per-block: scales 是 [out_features, num_blocks]  
                # 3. RTN per-channel: scales 是 [out_features]
                
                # 🔥 设备兼容性修复：确保所有量化参数在正确的设备上
                # 这在多GPU环境（如 accelerate）下非常重要
                device = x.device
                weight_scale_bf16 = self.weight_scale.to(device=device, dtype=torch.bfloat16)
                weight_zero_point_bf16 = self.weight_zero_point.to(device=device, dtype=torch.bfloat16)
                quantized_weight = self.quantized_weight.to(device=device)
                
                if len(self.weight_scale.shape) == 2:
                    # 2D scales: 可能是 GPTQ per-group 或 RTN per-block
                    if self.use_gptq:
                        # 🔥 GPTQ per-group quantization with actorder
                        if hasattr(self, 'gptq_g_idx') and self.gptq_g_idx is not None:
                            # 使用g_idx进行反量化（actorder）
                            # 🔥 性能优化：向量化反量化，避免逐列循环
                            g_idx = self.gptq_g_idx.to(device=device, dtype=torch.int64)  # 移到GPU，转为int64
                            quantized_float = quantized_weight.float()  # [out_features, in_features]
                            
                            # 向量化：为每个列选择对应的 scale
                            # g_idx: [in_features] -> group indices
                            # weight_scale_bf16: [out_features, num_groups]
                            # 需要: residual_weight[:, i] = quantized_float[:, i] * weight_scale_bf16[:, g_idx[i]]
                            
                            # 使用 advanced indexing 向量化
                            num_groups = weight_scale_bf16.shape[1]
                            # 创建索引矩阵 [out_features, in_features]，每列的值是该列对应的 group_idx
                            group_indices = g_idx.unsqueeze(0).expand(quantized_float.shape[0], -1).long()  # [out_features, in_features], int64
                            
                            # 使用 gather 选择对应的 scales
                            # weight_scale_bf16: [out_features, num_groups]
                            # group_indices: [out_features, in_features]
                            selected_scales = torch.gather(
                                weight_scale_bf16.unsqueeze(2),  # [out_features, num_groups, 1]
                                dim=1,
                                index=group_indices.unsqueeze(2)  # [out_features, in_features, 1], int64
                            ).squeeze(2)  # [out_features, in_features]
                            
                            # 向量化反量化
                            residual_weight = (quantized_float * selected_scales).to(torch.bfloat16)
                            
                            # 清理临时变量
                            del quantized_float, group_indices, selected_scales, g_idx
                        else:
                            # 标准per-group反量化（无actorder）
                            residual_weight = self.dequantize_tensor(
                                quantized_weight.float(),
                                weight_scale_bf16,  # [out_features, num_groups]
                                weight_zero_point_bf16 if len(weight_zero_point_bf16.shape) == 2 else weight_zero_point_bf16.unsqueeze(1),
                                is_blocked=True,
                                block_size=self.gptq_group_size,  # 🔥 使用 GPTQ group size
                            )
                    else:
                        # RTN per-block quantization
                        residual_weight = self.dequantize_tensor(
                            quantized_weight.float(),
                            weight_scale_bf16,  # [out_features, num_blocks]
                            weight_zero_point_bf16 if len(weight_zero_point_bf16.shape) == 2 else weight_zero_point_bf16.unsqueeze(1),
                            is_blocked=True,
                            block_size=self.block_size_weight,  # 🔥 使用 RTN block size
                        )
                else:
                    # Per-channel quantization (RTN or GPTQ with full-channel groups)
                    residual_weight = self.dequantize_tensor(
                        quantized_weight.float(),
                        weight_scale_bf16.unsqueeze(1),  # [out_features, 1]
                        weight_zero_point_bf16.unsqueeze(1),
                        is_blocked=False,
                        block_size=self.block_size_weight,  # block_size 在这里不使用
                    )
                # 🔥 dequantize_tensor 已经返回 bfloat16，这里不需要再次转换
                # residual_weight = residual_weight.to(torch.bfloat16)  # 已由 dequantize_tensor 保证
                
                # 🔥 缓存反量化后的残差权重（不包含异常值）
                if self.cache_dequant_weight and not self.training:
                    self._cached_dequant_weight = residual_weight.detach()
                    self._weight_cache_dtype = torch.bfloat16
            
            # 🔥 计算残差部分：Y_residual = X_quant @ W_residual（使用量化的激活）
            residual_output = F.linear(x, residual_weight, None)
            residual_output = residual_output.to(torch.bfloat16)
            
            # 🔥 合并 SVD 和残差部分
            if output is not None:
                # 有 SVD: output = Y_svd + Y_residual
                output = output + residual_output
            else:
                # 无 SVD: output = Y_residual
                output = residual_output
            
            # 🔥 Step 3.3: 加上稀疏异常值的贡献（使用量化的激活）
            # 【关键优化】异常值用稀疏矩阵乘法，完全不重建密集矩阵
            # 注意：outlier 是从 SVD 残差中提取的，包含了 SVD 无法捕获的高频细节
            # 原理：Y_outlier = X_quant @ W_outlier
            # 其中 W_outlier 是稀疏的，只有少量非零元素
            if self.use_sparse and hasattr(self, 'outlier_indices') and self.outlier_indices is not None:
                # 🔥 性能优化核心思路：
                # 1. 避免重复的设备转换（检查是否已在正确设备）
                # 2. 使用scatter_add_的优化版本（预计算expand）
                # 3. 不使用torch.sparse（不支持bfloat16）
                
                device = x.device
                batch_size = x.shape[0]
                
                # 🔥 关键优化1：只在需要时才转移设备，避免重复to()
                if not hasattr(self, '_outlier_device') or self._outlier_device != device:
                    self._outlier_indices_cached = self.outlier_indices.to(device)
                    self._outlier_values_cached = self.outlier_values.to(device=device, dtype=torch.bfloat16)
                    self._outlier_device = device
                
                out_idx = self._outlier_indices_cached[:, 0]
                in_idx = self._outlier_indices_cached[:, 1]
                
                # 🔥 关键优化2：缓存expanded indices for常见的batch_size
                # 🔥 修复OOM：只缓存batch_size=1，避免缓存累积
                if batch_size == 1:
                    cache_key = '_outlier_idx_expanded_1'
                    if not hasattr(self, cache_key):
                        setattr(self, cache_key, out_idx.unsqueeze(0))
                    out_idx_expanded = getattr(self, cache_key)
                else:
                    # 其他batch_size不缓存，每次重新expand（避免内存累积）
                    out_idx_expanded = out_idx.unsqueeze(0).expand(batch_size, -1).contiguous()
                
                # 🔥 关键优化3：使用优化的向量化操作
                # index_select比直接索引稍快
                x_selected = torch.index_select(x, -1, in_idx)  # [batch, num_outliers]
                contributions = x_selected * self._outlier_values_cached.unsqueeze(0)  # [batch, num_outliers]
                
                # scatter_add_: 将contributions加到output的对应位置
                output.scatter_add_(dim=-1, index=out_idx_expanded, src=contributions)
                
                # 清理临时变量
                del x_selected, contributions
            
            # 🔥 Step 3.4: 加上 bias
            if self.bias is not None:
                # 🔥 设备兼容性修复：确保 bias 与 output 在同一设备
                bias = self.bias.to(device=output.device, dtype=torch.bfloat16)
                output = output + bias
        else:
            # 训练模式或未量化（直接使用原始权重）
            # 🔥 设备兼容性修复：确保 weight 和 bias 与输入 x 在同一设备
            weight = self.weight.to(device=x.device, dtype=torch.bfloat16)
            bias = self.bias.to(device=x.device, dtype=torch.bfloat16) if self.bias is not None else None
            output = F.linear(x, weight, bias)
        
        # 🔥 最终确保输出是 BFloat16
        output = output.to(torch.bfloat16)
        
        return output
    
    def clear_sparse_cache(self):
        """
        清理sparse outlier相关的缓存，用于配置切换时释放内存
        """
        if hasattr(self, '_outlier_indices_cached'):
            delattr(self, '_outlier_indices_cached')
        if hasattr(self, '_outlier_values_cached'):
            delattr(self, '_outlier_values_cached')
        if hasattr(self, '_outlier_device'):
            delattr(self, '_outlier_device')
        if hasattr(self, '_outlier_idx_expanded_1'):
            delattr(self, '_outlier_idx_expanded_1')
    
    def extra_repr(self) -> str:
        parts = [
            f'in_features={self.in_features}, out_features={self.out_features}',
            f'weight_bit={self.weight_bit}, act_bit={self.act_bit}',
            f'sparse={self.use_sparse}, smooth={self.use_smoothquant}, svd={self.use_svd}',
        ]
        if self.use_block_quant:
            parts.append(f'weight_block_quant=True, block_size={self.block_size_weight}')
        if self.use_block_quant_act:
            parts.append(f'act_block_quant=True, block_size={self.block_size_act}')
        return ', '.join(parts)

