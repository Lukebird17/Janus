"""
Stage 0 (Janus-Pro-7B): 收集完整激活值用于 GPTQ / AWQ / SmoothQuant

Janus-Pro-7B LLM backbone 为 LlamaForCausalLM (30层)。
理解任务: SigLIP ViT + MLP connector → language_model
生成任务: VQ tokenizer → language_model

在线积累 Hessian + channel stats，不存原始 activation，内存恒定。

从 Bagel 版本迁移，适配 Janus VLChatProcessor 接口。
"""

import os
import sys
import gc
import json
import functools
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from tqdm import tqdm
from PIL import Image

_current_dir = Path(__file__).resolve().parent
_quant_dir = _current_dir.parent

if str(_quant_dir) not in sys.path:
    sys.path.insert(0, str(_quant_dir))

from utils.model_loader import JanusModelLoader


DEFAULT_UND_PROMPTS = [
    "What is the capital of France? Answer the question using a single word or phrase.",
    "Describe the main objects and their colors in this image.",
    "Is the sky blue? Answer the question using a single word or phrase.",
    "What color is grass? Answer the question using a single word or phrase.",
    "How many legs does a cat have? Answer the question using a single word or phrase.",
    "What is 2 + 2? Answer the question using a single word or phrase.",
    "Is water wet? Answer the question using a single word or phrase.",
    "What is the largest planet in our solar system? Answer the question using a single word or phrase.",
    "Is the Earth flat? Answer the question using a single word or phrase.",
    "What language is spoken in Japan? Answer the question using a single word or phrase.",
    "How many days are in a week? Answer the question using a single word or phrase.",
    "What is the boiling point of water in Celsius? Answer the question using a single word or phrase.",
    "Is the sun a star? Answer the question using a single word or phrase.",
    "What is the chemical symbol for gold? Answer the question using a single word or phrase.",
    "How many continents are there? Answer the question using a single word or phrase.",
    "What is the speed of light approximately? Answer the question using a single word or phrase.",
]


def _symlink(target: Path, link: Path):
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.name)


class FullActivationCollector:
    """收集 Janus language_model 中所有 Linear 层的激活统计量。

    在线积累 Hessian (X^T @ X) + channel_max + channel_mean。
    """

    def __init__(self, model: nn.Module, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hessian_sum: Dict[str, torch.Tensor] = {}
        self.hessian_nsamples: Dict[str, int] = {}
        self.act_channel_max: Dict[str, torch.Tensor] = {}
        self.act_channel_mean_sum: Dict[str, torch.Tensor] = {}
        self.act_layer_max: Dict[str, float] = {}
        self.weight_stats: Dict[str, Dict] = {}
        self.hooks: List = []
        self.token_count: Dict[str, int] = {}

    def _accumulate_hessian(self, name: str, tensor: torch.Tensor):
        with torch.no_grad():
            hidden_dim = tensor.shape[-1]
            x = tensor.view(-1, hidden_dim).float()
            n_tokens = x.shape[0]
            hessian_bytes = hidden_dim * hidden_dim * 8  # float64
            if hessian_bytes > 500 * 1024 * 1024:
                x_cpu = x.cpu()
                xtx = (x_cpu.t() @ x_cpu).double()
                del x_cpu
            else:
                xtx = (x.t() @ x).double().cpu()
            if name not in self.hessian_sum:
                self.hessian_sum[name] = xtx
                self.hessian_nsamples[name] = n_tokens
            else:
                self.hessian_sum[name].add_(xtx)
                self.hessian_nsamples[name] += n_tokens
            del xtx

    def _accumulate_channel_stats(self, name: str, tensor: torch.Tensor):
        with torch.no_grad():
            hidden_dim = tensor.shape[-1]
            t_flat = tensor.view(-1, hidden_dim).abs()
            n_tokens = t_flat.shape[0]

            layer_max = t_flat.max().float().cpu().item()
            if name in self.act_layer_max:
                self.act_layer_max[name] = max(self.act_layer_max[name], layer_max)
            else:
                self.act_layer_max[name] = layer_max

            ch_max = t_flat.max(dim=0)[0].float().cpu()
            if name in self.act_channel_max:
                self.act_channel_max[name] = torch.maximum(
                    self.act_channel_max[name], ch_max
                )
            else:
                self.act_channel_max[name] = ch_max

            ch_mean_sum = t_flat.sum(dim=0).float().cpu()
            if name in self.act_channel_mean_sum:
                self.act_channel_mean_sum[name].add_(ch_mean_sum)
            else:
                self.act_channel_mean_sum[name] = ch_mean_sum

            self.token_count[name] = self.token_count.get(name, 0) + n_tokens

    def _input_hook(self, m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_det = x.detach()
        self._accumulate_hessian(name, x_det)
        self._accumulate_channel_stats(name, x_det)

        if name not in self.weight_stats and hasattr(m, "weight") and m.weight is not None:
            with torch.no_grad():
                if m.weight.device.type != "meta":
                    w = m.weight.detach()
                    self.weight_stats[name] = {
                        "channel_max_input": w.abs().max(dim=0)[0].float().cpu(),
                        "channel_max_output": w.abs().max(dim=1)[0].float().cpu(),
                        "channel_max": w.abs().max(dim=0)[0].float().cpu(),
                        "layer_max": w.abs().max().float().cpu().item(),
                        "layer_mean": w.abs().mean().float().cpu().item(),
                        "num_channels_in": w.shape[1],
                        "num_channels_out": w.shape[0],
                        "num_channels": w.shape[1],
                    }

    def register_hooks(self, module_patterns: Optional[List[str]] = None):
        if module_patterns is None:
            module_patterns = ["language_model"]
        print(f"\nRegistering hooks (patterns={module_patterns}) ...")
        for name, m in self.model.named_modules():
            if not any(p in name for p in module_patterns):
                continue
            if isinstance(m, nn.Linear):
                self.hooks.append(
                    m.register_forward_hook(
                        functools.partial(self._input_hook, name=name)
                    )
                )
        print(f"  Registered {len(self.hooks)} hooks")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def collect_activations(
        self, model, processor, samples: List[Dict],
    ):
        """理解任务前向传播：所有样本 prefill，Hessian 和 channel stats 在线积累。"""
        print(f"\n  Running {len(samples)} samples (online Hessian + channel stats) ...")
        for idx, sample in enumerate(tqdm(samples, desc="Collect")):
            with torch.no_grad():
                try:
                    JanusModelLoader.forward_sample(model, processor, sample)
                except Exception as e:
                    print(f"\n  Error on sample {idx}: {e}")
                    continue
            if idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    def collect_weight_stats(self, module_patterns: Optional[List[str]] = None):
        if module_patterns is None:
            module_patterns = ["language_model"]
        for name, m in self.model.named_modules():
            if not any(p in name for p in module_patterns):
                continue
            if name in self.weight_stats:
                continue
            if isinstance(m, nn.Linear) and hasattr(m, "weight") and m.weight is not None:
                with torch.no_grad():
                    if m.weight.device.type == "meta":
                        continue
                    w = m.weight.detach()
                    self.weight_stats[name] = {
                        "channel_max_input": w.abs().max(dim=0)[0].float().cpu(),
                        "channel_max_output": w.abs().max(dim=1)[0].float().cpu(),
                        "channel_max": w.abs().max(dim=0)[0].float().cpu(),
                        "layer_max": w.abs().max().float().cpu().item(),
                        "layer_mean": w.abs().mean().float().cpu().item(),
                        "num_channels_in": w.shape[1],
                        "num_channels_out": w.shape[0],
                        "num_channels": w.shape[1],
                    }

    @staticmethod
    def _safe_name(layer_name: str) -> str:
        return layer_name.replace(".", "__")

    def save(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_names = sorted(self.hessian_sum.keys())

        # 1. GPTQ: per-layer Hessian
        gptq_dir = self.output_dir / "gptq_hessian"
        gptq_dir.mkdir(parents=True, exist_ok=True)
        gptq_index = {}
        gptq_disk_mb = 0.0

        print(f"\n[GPTQ] Saving per-layer Hessian ({len(layer_names)} layers) ...")
        for li, name in enumerate(layer_names):
            H = self.hessian_sum[name]
            n = self.hessian_nsamples[name]
            D = H.shape[0]
            safe = self._safe_name(name)
            out_path = gptq_dir / f"{safe}.pt"
            torch.save({"hessian_sum": H, "nsamples": n}, out_path)
            file_mb = out_path.stat().st_size / (1024 * 1024)
            gptq_disk_mb += file_mb
            gptq_index[name] = {
                "path": str(out_path), "hidden_dim": D,
                "nsamples": n, "file_mb": round(file_mb, 1),
            }
            if (li + 1) % 40 == 0 or (li + 1) == len(layer_names):
                print(f"  [{li+1}/{len(layer_names)}] cumulative: {gptq_disk_mb:.0f} MB")

        gptq_index_data = {
            "format": "gptq_hessian_v1",
            "num_layers": len(gptq_index),
            "total_disk_mb": round(gptq_disk_mb, 1),
            "timestamp": timestamp,
            "layers": gptq_index,
        }
        gptq_index_path = self.output_dir / f"gptq_hessian_index_{timestamp}.json"
        with open(gptq_index_path, "w") as f:
            json.dump(gptq_index_data, f, indent=2)
        _symlink(gptq_index_path, self.output_dir / "gptq_hessian_index_latest.json")
        print(f"  GPTQ Hessian: {gptq_dir} ({gptq_disk_mb:.0f} MB)")

        # 2. SmoothQuant
        smooth_data = {}
        for name in layer_names:
            n = self.token_count.get(name, 0)
            D = self.hessian_sum[name].shape[0]
            smooth_data[name] = {
                "act_channel_max": self.act_channel_max.get(name, torch.zeros(D)),
                "weight_channel_max": (
                    self.weight_stats[name]["channel_max"]
                    if name in self.weight_stats else torch.zeros(D)
                ),
                "nsamples": n,
            }
        smooth_path = self.output_dir / f"smoothquant_stats_{timestamp}.pt"
        torch.save(smooth_data, smooth_path)
        _symlink(smooth_path, self.output_dir / "smoothquant_stats_latest.pt")
        smooth_mb = smooth_path.stat().st_size / (1024 * 1024)
        print(f"  SmoothQuant: {smooth_path} ({smooth_mb:.1f} MB)")

        # 3. AWQ
        awq_data = {}
        for name in layer_names:
            n = self.token_count.get(name, 0)
            D = self.hessian_sum[name].shape[0]
            ch_mean = (
                self.act_channel_mean_sum[name] / n
                if name in self.act_channel_mean_sum and n > 0
                else torch.zeros(D)
            )
            awq_data[name] = {
                "channel_mean": ch_mean,
                "channel_max": self.act_channel_max.get(name, torch.zeros(D)),
                "nsamples": n,
            }
        awq_path = self.output_dir / f"awq_stats_{timestamp}.pt"
        torch.save(awq_data, awq_path)
        _symlink(awq_path, self.output_dir / "awq_stats_latest.pt")
        awq_mb = awq_path.stat().st_size / (1024 * 1024)
        print(f"  AWQ: {awq_path} ({awq_mb:.1f} MB)")

        total_mb = gptq_disk_mb + smooth_mb + awq_mb
        print(f"\n  Total disk: {total_mb:.0f} MB")
        print(f"    GPTQ Hessian : {gptq_disk_mb:.0f} MB ({len(gptq_index)} layers)")
        print(f"    SmoothQuant  : {smooth_mb:.1f} MB")
        print(f"    AWQ          : {awq_mb:.1f} MB")

        return gptq_index_path, smooth_path, awq_path


def load_external_calibration_dataset(dataset_path: str) -> List[Dict]:
    """从 JSON 文件加载校准样本。"""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("samples", data)
    if isinstance(raw, dict):
        raw = list(raw.values())

    samples = []
    for s in raw:
        img_path = s.get("image_path")
        img = None
        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                pass
        samples.append({
            "prompt": s.get("question", s.get("prompt", "")),
            "image": img,
            "task_type": "und",
        })

    img_count = sum(1 for s in samples if s["image"] is not None)
    print(f"  Loaded external calibration dataset: {len(samples)} samples "
          f"({img_count} with images)")
    return samples


def build_calibration_data(
    num_und: int = 16,
    mme_data_root: Optional[str] = None,
) -> List[Dict]:
    """构建校准数据集（仅理解任务，与 Bagel 不同无 gen 路径）。"""
    und_samples = []

    if mme_data_root and Path(mme_data_root).exists():
        data_root = Path(mme_data_root)
        categories = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
        if categories:
            samples_per_cat = max(1, num_und // len(categories))
            for cat in categories:
                cat_path = data_root / cat
                for txt_file in sorted(cat_path.glob("*.txt"))[:samples_per_cat]:
                    if len(und_samples) >= num_und:
                        break
                    with open(txt_file, "r") as f:
                        lines = f.readlines()
                    img_name = txt_file.stem
                    img_path = None
                    for ext in [".png", ".jpg", ".jpeg"]:
                        p = cat_path / f"{img_name}{ext}"
                        if p.exists():
                            img_path = p
                            break
                    if img_path is None:
                        img_dir = cat_path / "images"
                        if img_dir.exists():
                            for ext in [".png", ".jpg", ".jpeg"]:
                                p = img_dir / f"{img_name}{ext}"
                                if p.exists():
                                    img_path = p
                                    break
                    if img_path is None:
                        continue
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            try:
                                img = Image.open(str(img_path)).convert("RGB")
                                und_samples.append({
                                    "prompt": parts[0] + " Answer the question using a single word or phrase.",
                                    "image": img,
                                    "task_type": "und",
                                })
                            except Exception:
                                pass
                            if len(und_samples) >= num_und:
                                break

    if len(und_samples) < num_und:
        for p in DEFAULT_UND_PROMPTS[: num_und - len(und_samples)]:
            und_samples.append({"prompt": p, "image": None, "task_type": "und"})

    return und_samples


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0 (Janus-Pro-7B): Collect activation statistics"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of calibration samples (if no external dataset)")
    parser.add_argument("--mme_data_root", type=str, default=None,
                        help="MME Benchmark directory for real VQA images")
    parser.add_argument("--calibration_dataset", type=str, default=None,
                        help="External calibration dataset JSON")
    parser.add_argument("--module_patterns", type=str, default="language_model",
                        help="Comma-separated module name patterns to hook")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            _quant_dir / "quantization_outputs" / "stage0_full_activation"
        )

    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("Stage 0 (Janus-Pro-7B): Online Activation Statistics Collection")
    print("=" * 70)
    print(f"  Model path    : {args.model_path}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  GPU IDs       : {args.gpu_ids or 'all'}")
    print(f"  Mode          : Online Hessian (GPTQ) + Channel Stats (AWQ/SmoothQuant)")
    if args.calibration_dataset:
        print(f"  Calib source  : EXTERNAL ({args.calibration_dataset})")
    else:
        print(f"  Samples       : {args.num_samples}")
    print(f"  Module patterns: {args.module_patterns}")

    print("\n[1/4] Loading Janus-Pro-7B model ...")
    loader = JanusModelLoader(
        model_path=args.model_path,
        gpu_ids=args.gpu_ids,
    )
    components = loader.load_all()
    model = components["model"]
    processor = components["processor"]
    model.eval()

    print(f"  Model device: {next(model.parameters()).device}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total params: {total_params:.1f}M")

    print("\n[2/4] Building calibration data ...")
    if args.calibration_dataset:
        samples = load_external_calibration_dataset(args.calibration_dataset)
    else:
        samples = build_calibration_data(
            num_und=args.num_samples,
            mme_data_root=args.mme_data_root,
        )
    img_count = sum(1 for s in samples if s.get("image") is not None)
    print(f"  Total: {len(samples)} samples ({img_count} with images)")

    print("\n[3/4] Collecting activations ...")
    module_patterns = [p.strip() for p in args.module_patterns.split(",")]

    collector = FullActivationCollector(model=model, output_dir=args.output_dir)
    collector.register_hooks(module_patterns)
    collector.collect_activations(model, processor, samples)
    collector.remove_hooks()
    collector.collect_weight_stats(module_patterns)

    print("\n[4/4] Saving results ...")
    gptq_path, smooth_path, awq_path = collector.save()

    print("\n" + "=" * 70)
    print("Stage 0 Completed!")
    print(f"  GPTQ Hessian  : {gptq_path}")
    print(f"  SmoothQuant   : {smooth_path}")
    print(f"  AWQ           : {awq_path}")
    print(f"  Layers        : {len(collector.hessian_sum)}")
    print(f"  Input samples : {len(samples)}")
    token_counts = sorted(collector.token_count.items())
    if token_counts:
        min_n = min(c for _, c in token_counts)
        max_n = max(c for _, c in token_counts)
        print(f"  Tokens/layer  : min={min_n}, max={max_n}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
