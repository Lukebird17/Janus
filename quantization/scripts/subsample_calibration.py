"""
从完整校准数据集 JSON 中随机抽取 N 条样本，生成子集 JSON。

用于 CSC (Consistent Subset Calibration) 实验：
  - 同一子集同时用于 stage0（激活统计收集）和 stage21（CKA 搜索）
  - 保证 "搜索-量化" 的校准数据一致性

用法:
    python subsample_calibration.py \
        --input /path/to/calibration_dataset.json \
        --output /path/to/subset_N100_seed42.json \
        --num_samples 100 \
        --seed 42
"""

import json
import random
import argparse
from pathlib import Path


def subsample(input_path: str, output_path: str, num_samples: int, seed: int):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("samples", data)
    if isinstance(raw, dict):
        raw = list(raw.values())

    if num_samples >= len(raw):
        print(f"  Requested {num_samples} >= total {len(raw)}, using all samples")
        subset = raw
    else:
        rng = random.Random(seed)
        subset = rng.sample(raw, num_samples)

    out_data = {
        "num_samples": len(subset),
        "source": data.get("source", "unknown"),
        "description": (
            f"Subset of {len(subset)} from {len(raw)} samples "
            f"(seed={seed}) for CSC experiment"
        ),
        "parent_dataset": str(input_path),
        "parent_total": len(raw),
        "seed": seed,
        "question_types": list(set(s.get("question_type", "unknown") for s in subset)),
        "samples": subset,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(subset)} samples -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Subsample calibration dataset for CSC")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subsample(args.input, args.output, args.num_samples, args.seed)


if __name__ == "__main__":
    main()
