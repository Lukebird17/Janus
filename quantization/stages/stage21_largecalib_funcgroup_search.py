"""
Stage 21 (Janus-Pro-7B): 大校准集 + W4A4 功能组级贪心 CKA 搜索

与 Stage 20 的核心差异:
  - Stage 20: 每个 decoder 层一种算法（层内所有 Linear 配置相同）
  - Stage 21: attn (Q/K/V/O) 和 mlp (gate/up/down) 分别选算法

Janus LLM backbone: LlamaForCausalLM (30 decoder layers)
层命名: language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj
        language_model.model.layers.{i}.mlp.{gate,up,down}_proj
"""

import os
import sys
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import yaml
import argparse

_current_dir = Path(__file__).resolve().parent
_quant_dir = _current_dir.parent

if str(_quant_dir) not in sys.path:
    sys.path.insert(0, str(_quant_dir))

from layers.hybrid_quant_linear import HybridQuantLinear
from utils.model_loader import JanusModelLoader

from stages.stage20_largecalib_search import (
    LinearCKA,
    LazyActivationProvider,
    LargeCalibrationLoader,
    build_stage20_pool,
    _W4_BASE,
)


FUNCTIONAL_GROUPS = [
    {
        'name': 'attn',
        'display': 'Attention (Q/K/V/O)',
        'suffixes': ['self_attn.q_proj', 'self_attn.k_proj',
                     'self_attn.v_proj', 'self_attn.o_proj'],
        'hook_target': 'self_attn',
    },
    {
        'name': 'mlp',
        'display': 'MLP (gate/up/down)',
        'suffixes': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
        'hook_target': 'mlp',
    },
]


def _s21_algo(name, desc, **overrides):
    cfg = dict(_W4_BASE)
    cfg.update(overrides)
    return {'name': name, 'description': desc, 'config': cfg}


def build_stage21_pool() -> Dict:
    pool = {}
    pool['gptq_w4a4'] = _s21_algo(
        'GPTQ W4A4', 'GPTQ group_size=64 W4A4',
        use_gptq=True, gptq_group_size=64,
    )
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'smooth_{tag}_gptq_w4a4'] = _s21_algo(
            f'Smooth(a={alpha})+GPTQ W4A4',
            f'SmoothQuant a={alpha}, GPTQ group_size=64 W4A4',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_gptq=True, gptq_group_size=64,
        )
    for alpha in [0.5, 0.7, 0.85]:
        tag = f'a{int(alpha*100)}'
        pool[f'svdquant_{tag}_w4a4'] = _s21_algo(
            f'SVDQuant a={alpha} W4A4',
            f'Smooth(a={alpha})+SVD(rank=32)+GPTQ W4A4',
            use_smoothquant=True, smoothquant_alpha=alpha,
            use_svd=True, svd_rank=32,
            use_gptq=True, gptq_group_size=64,
        )
    pool['svd_gptq_w4a4'] = _s21_algo(
        'SVD+GPTQ W4A4', 'SVD rank=32, GPTQ on residual W4A4',
        use_svd=True, svd_rank=32,
        use_gptq=True, gptq_group_size=64,
    )
    pool['awq_svd_rtn_w4a4'] = _s21_algo(
        'AWQ+SVD+RTN W4A4', 'AWQ n_grid=20, SVD rank=32, RTN W4A4',
        use_awq=True, awq_n_grid=20,
        use_svd=True, svd_rank=32,
    )
    return pool


class Stage21Searcher:
    """Stage 21: Janus-Pro-7B functional-group CKA search (W4A4)."""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantization_outputs/stage21_funcgroup",
        calibration_dataset: Optional[str] = None,
        algorithm_pool: Optional[Dict] = None,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        activation_data_file: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        max_mem_per_gpu: str = "40GiB",
        target_decoder_layers: Optional[List[int]] = None,
        seed: int = 42,
        subsample_step: int = 5,
        max_calib_samples: Optional[int] = None,
        cka_num_samples: int = 200,
        run_date: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_dataset = calibration_dataset
        self.algorithm_pool = algorithm_pool or build_stage21_pool()
        self.gpu_ids = gpu_ids
        self.max_mem_per_gpu = max_mem_per_gpu
        self.target_decoder_layers = target_decoder_layers
        self.seed = seed
        self.subsample_step = subsample_step
        self.max_calib_samples = max_calib_samples
        self.cka_num_samples = cka_num_samples
        self.run_date = run_date or datetime.now().strftime("%Y%m%d")

        # CUDA_VISIBLE_DEVICES 必须在任何 torch.cuda 调用之前设置，否则 _set_seed() 会
        # 先初始化 CUDA，此后 JanusModelLoader 里再改环境变量无效。
        if self.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)
            print(f"Using GPUs: {self.gpu_ids}")

        self._set_seed()
        self.activation_provider = LazyActivationProvider(
            gptq_hessian_index=gptq_hessian_index,
            smoothquant_stats=smoothquant_stats,
            awq_stats=awq_stats,
            legacy_activation_file=activation_data_file,
        )
        self._load_model()
        self.original_weights = {}
        self._save_original_weights()

        self.num_decoder_layers = JanusModelLoader.get_num_decoder_layers(self.model)
        if self.target_decoder_layers is None:
            self.target_decoder_layers = list(range(self.num_decoder_layers))
        self._print_banner()

    def _set_seed(self):
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_model(self):
        print(f"\nLoading Janus-Pro-7B from {self.model_path} ...")
        loader = JanusModelLoader(
            model_path=self.model_path,
            gpu_ids=self.gpu_ids,
        )
        components = loader.load_all()
        self.model = components["model"]
        self.model.eval()
        self.processor = components["processor"]
        self.tokenizer = components["tokenizer"]
        print("  Model loaded successfully")

    def _save_original_weights(self):
        print("\nSaving original weights (language_model.model.layers) ...")
        count = 0
        for name, module in self.model.named_modules():
            if not name.startswith("language_model.model.layers."):
                continue
            if isinstance(module, nn.Linear) and not isinstance(module, HybridQuantLinear):
                if module.weight.device.type == "meta":
                    continue
                self.original_weights[name] = {
                    "weight": module.weight.data.clone().cpu(),
                    "bias": module.bias.data.clone().cpu() if module.bias is not None else None,
                }
                count += 1
        print(f"  Saved {count} layers")

    def _print_banner(self):
        print("\n" + "=" * 80)
        print("Stage 21 (Janus-Pro-7B): Functional-Group CKA Search")
        print("=" * 80)
        print(f"  Model: {self.model_path}")
        print(f"  Run date: {self.run_date}")
        print(f"  Decoder layers: {self.num_decoder_layers}")
        print(f"  Functional groups: {[g['display'] for g in FUNCTIONAL_GROUPS]}")
        print(f"  Algorithm pool: {list(self.algorithm_pool.keys())}")
        print(f"  Output: {self.output_dir}")
        print("=" * 80 + "\n")

    # ---- module helpers ----

    def _get_group_sublayer_names(self, layer_idx: int, group: Dict) -> List[str]:
        prefix = f"language_model.model.layers.{layer_idx}"
        return [f"{prefix}.{s}" for s in group['suffixes']]

    def _get_module(self, name: str) -> Optional[nn.Module]:
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _replace_module(self, name: str, new_module: nn.Module):
        parts = name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    # ---- quantization apply / restore (per sublayer) ----

    def _apply_algorithm_to_layer(self, layer_name: str, algo_config: Dict):
        module = self._get_module(layer_name)
        if module is None or not isinstance(module, (nn.Linear, HybridQuantLinear)):
            return
        if isinstance(module, HybridQuantLinear):
            self._restore_layer(layer_name)
            module = self._get_module(layer_name)

        device = module.weight.device
        dtype = module.weight.dtype
        config = algo_config.copy()
        if config.get("use_gptq", False) and not self.activation_provider.available:
            config["use_gptq"] = False
        if config.get("use_awq", False) and not self.activation_provider.available:
            config["use_awq"] = False
        if config.get("use_gptq", False) and layer_name not in self.activation_provider:
            print(f"  [Warning] GPTQ: no activation data for {layer_name}, falling back to RTN")
            config["use_gptq"] = False
        if config.get("use_awq", False) and layer_name not in self.activation_provider:
            print(f"  [Warning] AWQ: no activation data for {layer_name}, falling back to RTN")
            config["use_awq"] = False

        quant_layer = HybridQuantLinear(
            in_features=module.in_features, out_features=module.out_features,
            bias=module.bias is not None,
            weight_bit=config.get("weight_bit", 4), act_bit=config.get("act_bit", 4),
            quant_percentile=config.get("quant_percentile", 0.999999),
            act_unsigned=config.get("act_unsigned", True),
            use_sparse=config.get("use_sparse", False),
            sparse_ratio=config.get("sparse_ratio", 0.0),
            sparse_threshold=config.get("sparse_threshold", None),
            use_smoothquant=config.get("use_smoothquant", False),
            smoothquant_alpha=config.get("smoothquant_alpha", 0.5),
            use_svd=config.get("use_svd", False), svd_rank=config.get("svd_rank", 0),
            use_block_quant=config.get("use_block_quant", False),
            use_block_quant_act=config.get("use_block_quant_act", False),
            block_size_weight=config.get("block_size_weight", 256),
            block_size_act=config.get("block_size_act", 256),
            use_gptq=config.get("use_gptq", False),
            gptq_group_size=config.get("gptq_group_size", 64),
            gptq_damp_percentage=config.get("gptq_damp_percentage", 0.01),
            gptq_block_size=config.get("gptq_block_size", 128),
            use_awq=config.get("use_awq", False),
            awq_alpha=config.get("awq_alpha", 0.5),
            awq_n_grid=config.get("awq_n_grid", 20),
            device=device, dtype=dtype,
        )
        if layer_name in self.original_weights:
            orig_w = self.original_weights[layer_name]["weight"].clone()
            orig_b = self.original_weights[layer_name]["bias"]
            if orig_b is not None:
                orig_b = orig_b.clone()
        else:
            orig_w = module.weight.data.clone()
            orig_b = module.bias.data.clone() if module.bias is not None else None
        quant_layer.weight.data = orig_w.to(device)
        if orig_b is not None:
            quant_layer.bias.data = orig_b.to(device)
        quant_layer = quant_layer.to(device)

        act_data = None
        if self.activation_provider.available and layer_name in self.activation_provider:
            act_data = self.activation_provider.get_activation(layer_name)
        quant_layer.prepare_weight(
            activation_data=act_data, layer_name=layer_name, verbose=False
        )
        self._replace_module(layer_name, quant_layer)
        del act_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_layer(self, layer_name: str):
        module = self._get_module(layer_name)
        if module is None:
            return
        if isinstance(module, HybridQuantLinear):
            device = module.weight.device if module.weight is not None else "cuda:0"
            dtype = module.weight.dtype if module.weight is not None else torch.bfloat16
            linear = nn.Linear(
                module.in_features, module.out_features,
                bias=module.bias is not None, device=device, dtype=dtype,
            )
            if layer_name in self.original_weights:
                linear.weight.data.copy_(
                    self.original_weights[layer_name]["weight"].to(device)
                )
                if (linear.bias is not None
                        and self.original_weights[layer_name]["bias"] is not None):
                    linear.bias.data.copy_(
                        self.original_weights[layer_name]["bias"].to(device)
                    )
            self._replace_module(layer_name, linear)
        elif isinstance(module, nn.Linear) and layer_name in self.original_weights:
            module.weight.data.copy_(
                self.original_weights[layer_name]["weight"].to(module.weight.device)
            )
            if (module.bias is not None
                    and self.original_weights[layer_name]["bias"] is not None):
                module.bias.data.copy_(
                    self.original_weights[layer_name]["bias"].to(module.bias.device)
                )

    def _apply_algorithm_to_group(self, layer_idx: int, group: Dict, algo_config: Dict):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._apply_algorithm_to_layer(name, algo_config)

    def _restore_group(self, layer_idx: int, group: Dict):
        for name in self._get_group_sublayer_names(layer_idx, group):
            self._restore_layer(name)

    def _redispatch_model(self):
        try:
            from accelerate import dispatch_model
            if hasattr(self.model, "hf_device_map"):
                self.model = dispatch_model(
                    self.model, device_map=self.model.hf_device_map, offload_dir=None,
                )
        except Exception as e:
            print(f"  [Warning] Re-dispatch failed: {e}")

    # ---- forward & CKA ----

    def _forward_calibration_sample(self, sample: Dict):
        JanusModelLoader.forward_sample(self.model, self.processor, sample)

    def _collect_group_hidden_states(
        self, layer_idx: int, group: Dict, calibration_samples: List[Dict],
    ) -> List[torch.Tensor]:
        """Hook on self_attn or mlp submodule to collect group-level hidden states."""
        captured_list = []
        captured = {}

        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h.detach().cpu()

        decoder_layer = JanusModelLoader.get_decoder_layers(self.model)[layer_idx]
        hook_target = getattr(decoder_layer, group['hook_target'])
        handle = hook_target.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                for sample in calibration_samples:
                    try:
                        self._forward_calibration_sample(sample)
                    except Exception:
                        captured.clear()
                        continue
                    if "output" in captured:
                        captured_list.append(captured["output"])
                    captured.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            handle.remove()
        return captured_list

    # ---- config helpers ----

    _FULL_CONFIG_TEMPLATE = {
        "weight_bit": 4, "act_bit": 4, "quant_percentile": 0.999999,
        "act_unsigned": True, "use_sparse": False, "sparse_ratio": 0.0,
        "sparse_threshold": None, "use_smoothquant": False,
        "smoothquant_alpha": 0.5, "use_svd": False, "svd_rank": 0,
        "use_gptq": False, "gptq_group_size": 64,
        "gptq_damp_percentage": 0.01, "gptq_block_size": 128,
        "use_block_quant": False, "use_block_quant_act": False,
        "block_size_weight": 256, "block_size_act": 256,
        "use_awq": False, "awq_alpha": 0.5, "awq_n_grid": 20,
    }

    def _build_full_layer_config(self, algo_key: str) -> Dict:
        cfg = dict(self._FULL_CONFIG_TEMPLATE)
        cfg.update(self.algorithm_pool[algo_key]["config"])
        if not cfg.get("use_svd", False):
            cfg["svd_rank"] = 0
        if not cfg.get("use_sparse", False):
            cfg["sparse_threshold"] = None
            cfg["sparse_ratio"] = 0.0
        return cfg

    def _subsample_for_cka(self, full_samples: List[Dict]) -> List[Dict]:
        import random
        n = self.cka_num_samples
        if len(full_samples) <= n:
            return full_samples
        by_type = {}
        for s in full_samples:
            qt = s.get("question_type", "unknown")
            by_type.setdefault(qt, []).append(s)
        rng = random.Random(self.seed)
        for v in by_type.values():
            rng.shuffle(v)
        selected = []
        types = sorted(by_type.keys())
        per_type = max(1, n // len(types))
        for t in types:
            selected.extend(by_type[t][:per_type])
        remaining_budget = n - len(selected)
        if remaining_budget > 0:
            pool = [s for t in types for s in by_type[t][per_type:]]
            rng.shuffle(pool)
            selected.extend(pool[:remaining_budget])
        selected = selected[:n]
        rng.shuffle(selected)
        return selected

    # ---- search ----

    def search(self) -> Dict:
        print("\n" + "=" * 80)
        print("Phase 1: Loading large calibration dataset")
        print("=" * 80)

        loader = LargeCalibrationLoader(
            dataset_path=self.calibration_dataset,
            max_samples=self.max_calib_samples,
        )
        all_calibration_samples = loader.load()
        cka_samples = self._subsample_for_cka(all_calibration_samples)

        available_algos = {}
        for key, algo in self.algorithm_pool.items():
            if algo["config"].get("use_gptq", False) and not self.activation_provider.available:
                print(f"    [SKIP] {key}: requires GPTQ Hessian data")
                continue
            if algo["config"].get("use_awq", False) and not self.activation_provider.available:
                print(f"    [SKIP] {key}: requires AWQ stats data")
                continue
            available_algos[key] = algo

        w4_algos = {
            k: v for k, v in available_algos.items()
            if v["config"].get("weight_bit", 4) == 4
        }
        print(f"\n  Available W4A4 algorithms: {list(w4_algos.keys())}")
        print(f"  CKA search uses: {len(cka_samples)} samples")

        if not w4_algos:
            raise RuntimeError("No W4A4 algorithms available.")

        print("\n" + "=" * 80)
        print("Phase 2: Functional-Group Greedy CKA Search (W4A4)")
        print("=" * 80)

        fallback = list(w4_algos.keys())[0]
        layer_group_assignments = {}
        search_log = []
        progress_file = self.output_dir / f"search_progress_funcgroup_w4a4_{self.run_date}.json"
        config_export_dir = Path(self.output_dir).parent / "configs"
        config_export_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        completed_layers = set()
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    prev = json.load(f)
                prev_log = prev.get('search_log', [])
                for entry in prev_log:
                    lidx = entry['layer_idx']
                    completed_layers.add(lidx)
                    layer_group_assignments[lidx] = entry['group_assignments']
                search_log = list(prev_log)
                if completed_layers:
                    print(f"\n  [Resume] Found progress for {len(completed_layers)} layers, restoring...")
                    for lidx in sorted(completed_layers):
                        for gname, algo_key in layer_group_assignments[lidx].items():
                            grp = next(g for g in FUNCTIONAL_GROUPS if g['name'] == gname)
                            self._apply_algorithm_to_group(
                                lidx, grp, w4_algos[algo_key]["config"]
                            )
                    self._redispatch_model()
                    remaining = set(self.target_decoder_layers) - completed_layers
                    next_layer = min(remaining) if remaining else 'DONE'
                    print(f"  [Resume] Continuing from layer {next_layer}.")
            except Exception as e:
                print(f"  [Resume] Failed ({e}), starting fresh.")
                completed_layers = set()
                layer_group_assignments = {}
                search_log = []

        for layer_idx in tqdm(self.target_decoder_layers, desc="  FuncGroup"):
            if layer_idx in completed_layers:
                continue
            print(f"\n  {'---' * 18}")
            print(f"  Decoder Layer {layer_idx}/{self.num_decoder_layers - 1}")

            group_assignments = {}

            for group in FUNCTIONAL_GROUPS:
                print(f"\n    Group: {group['display']}")

                ref_hs = self._collect_group_hidden_states(
                    layer_idx, group, cka_samples,
                )
                if not ref_hs:
                    group_assignments[group['name']] = fallback
                    print(f"      No ref hidden states, using fallback: {fallback}")
                    continue

                best_algo_key = None
                best_cka = -1.0
                algo_scores = {}

                for algo_key, algo_info in w4_algos.items():
                    print(f"      Trying: {algo_key} ...", end=" ")
                    try:
                        self._apply_algorithm_to_group(
                            layer_idx, group, algo_info["config"],
                        )
                        self._redispatch_model()
                        quant_hs = self._collect_group_hidden_states(
                            layer_idx, group, cka_samples,
                        )
                        cka = LinearCKA.compute_batched(
                            ref_hs, quant_hs, subsample_step=self.subsample_step,
                        ) if quant_hs else 0.0
                        algo_scores[algo_key] = cka
                        print(f"CKA = {cka:.6f}")
                        if cka > best_cka:
                            best_cka = cka
                            best_algo_key = algo_key
                    except Exception as e:
                        print(f"FAILED: {e}")
                        algo_scores[algo_key] = -1.0
                    finally:
                        self._restore_group(layer_idx, group)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if best_algo_key is None:
                    best_algo_key = fallback
                    best_cka = algo_scores.get(fallback, 0.0)

                group_assignments[group['name']] = best_algo_key
                print(f"      >>> Best for {group['name']}: "
                      f"{best_algo_key} (CKA={best_cka:.6f})")

                self._apply_algorithm_to_group(
                    layer_idx, group, w4_algos[best_algo_key]["config"],
                )
                self._redispatch_model()

                del ref_hs
                gc.collect()

            layer_group_assignments[layer_idx] = group_assignments

            search_log.append({
                "layer_idx": layer_idx,
                "group_assignments": group_assignments,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            with open(progress_file, "w") as f:
                json.dump({
                    "completed_layers": len(search_log),
                    "total_layers": len(self.target_decoder_layers),
                    "search_log": search_log,
                }, f, indent=2)
            self.activation_provider.clear_cache()
            gc.collect()

        # Phase 3: Export config
        print("\n" + "=" * 80)
        print("Phase 3: Exporting quantization config")
        print("=" * 80)

        export_cfg = {}
        for layer_idx, group_assignments in layer_group_assignments.items():
            for group in FUNCTIONAL_GROUPS:
                algo_key = group_assignments.get(group['name'], fallback)
                full_cfg = self._build_full_layer_config(algo_key)
                for sublayer_name in self._get_group_sublayer_names(layer_idx, group):
                    export_cfg[sublayer_name] = dict(full_cfg)

        config_path = config_export_dir / f"stage21_funcgroup_w4a4_{self.run_date}.json"
        with open(config_path, "w") as f:
            json.dump(dict(sorted(export_cfg.items())), f, indent=2)
        print(f"  Exported: {config_path}")

        results = {
            "bitwidth_results": {
                "4": {
                    "weight_bit": 4,
                    "layer_group_assignments": {
                        str(k): v for k, v in layer_group_assignments.items()
                    },
                    "search_log": search_log,
                    "exported_config_path": str(config_path),
                },
            },
            "metadata": {
                "stage": 21,
                "model_path": self.model_path,
                "model_type": "janus-pro-7b",
                "run_date": self.run_date,
                "num_decoder_layers": self.num_decoder_layers,
                "functional_groups": [g['name'] for g in FUNCTIONAL_GROUPS],
                "algorithm_pool": list(self.algorithm_pool.keys()),
                "calibration_dataset": str(self.calibration_dataset),
                "cka_search_samples": len(cka_samples),
                "seed": self.seed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        results_path = self.output_dir / f"stage21_search_results_{self.run_date}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"Stage 21 Summary (run_date={self.run_date}):")
        for gname in ['attn', 'mlp']:
            algo_counts = {}
            for ga in layer_group_assignments.values():
                a = ga.get(gname, fallback)
                algo_counts[a] = algo_counts.get(a, 0) + 1
            print(f"  {gname}: {dict(sorted(algo_counts.items(), key=lambda x: -x[1]))}")
        print(f"\n  Config: {config_path}")
        print(f"  Results: {results_path}")
        print(f"{'=' * 80}\n")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 21 (Janus-Pro-7B): Functional-Group CKA Search (W4A4)"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./quantization_outputs/stage21_funcgroup")
    parser.add_argument("--calibration_dataset", type=str, required=True)
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--activation_data", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB")
    parser.add_argument("--target_layers", type=str, default=None)
    parser.add_argument("--max_calib_samples", type=int, default=None)
    parser.add_argument("--cka_num_samples", type=int, default=200)
    parser.add_argument("--subsample_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_date", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        for key, val in yaml_config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, val)

    target_layers = None
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    searcher = Stage21Searcher(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_dataset=args.calibration_dataset,
        gptq_hessian_index=args.gptq_hessian_index,
        smoothquant_stats=args.smoothquant_stats,
        awq_stats=args.awq_stats,
        activation_data_file=args.activation_data,
        gpu_ids=args.gpu_ids,
        max_mem_per_gpu=args.max_mem_per_gpu,
        target_decoder_layers=target_layers,
        seed=args.seed,
        subsample_step=args.subsample_step,
        max_calib_samples=args.max_calib_samples,
        cka_num_samples=args.cka_num_samples,
        run_date=args.run_date,
    )
    results = searcher.search()
    print("\nStage 21 search completed!")
    print(f"Results: {searcher.output_dir / f'stage21_search_results_{searcher.run_date}.json'}")


if __name__ == "__main__":
    main()
