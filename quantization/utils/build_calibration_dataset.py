"""
Stage 20 校准数据集构建器

从 Flickr8K（或其他本地图文数据集）构建 ~1000 条多类型 VQA 校准样本。
通过将 caption 自动转化为多种 VQA 问题类型（existence, attribute, counting,
spatial, description），覆盖 MME 评测的主要认知维度，同时保证与 MME 测试集
完全无重叠。

生成的数据集同时用于：
  1. Stage 0 GPTQ/AWQ 激活收集
  2. Stage 20 混合精度搜索的 CKA 校准

用法:
    python build_calibration_dataset.py \
        --flickr8k_root /path/to/flickr8k \
        --output_dir /path/to/output \
        --num_samples 1000 \
        --seed 42
"""

import os
import re
import csv
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


# ================================================================
# VQA 问题模板 — 覆盖 MME 的认知维度
# ================================================================

# 通用描述类 (对应 MME scene/artwork 维度)
DESCRIBE_TEMPLATES = [
    "Describe the main content of this image in one sentence.",
    "What is happening in this image?",
    "Briefly describe this scene.",
    "What is the main subject of this image?",
    "Summarize what you see in this image.",
]

# 存在性判断 (对应 MME existence 维度)
EXISTENCE_TEMPLATES_POS = [
    "Is there {noun} in this image? Answer yes or no.",
    "Can you see {noun} in this image? Answer yes or no.",
    "Does this image contain {noun}? Answer yes or no.",
]
EXISTENCE_TEMPLATES_NEG = [
    "Is there {noun} in this image? Answer yes or no.",
    "Can you see {noun} in this image? Answer yes or no.",
]

# 属性/颜色 (对应 MME color 维度)
ATTRIBUTE_TEMPLATES = [
    "What color is the {noun} in this image? Answer briefly.",
    "Describe the appearance of the {noun} in this image.",
    "What does the {noun} look like in this image? Answer briefly.",
]

# 计数 (对应 MME count 维度)
COUNT_TEMPLATES = [
    "How many {noun} are there in this image? Answer with a number.",
    "Count the {noun} in this image. Answer with a number.",
]

# 位置/空间 (对应 MME position 维度)
POSITION_TEMPLATES = [
    "Where is the {noun} located in this image? Answer briefly.",
    "Describe the position of the {noun} in this image.",
    "Is the {noun} on the left or right side of the image? Answer briefly.",
]

# OCR / 文本相关 (对应 MME OCR 维度)
OCR_TEMPLATES = [
    "Is there any text visible in this image? If so, what does it say?",
    "Can you read any text or signs in this image? Answer briefly.",
]

# 常识推理 (对应 MME commonsense_reasoning 维度)
REASONING_TEMPLATES = [
    "What activity is likely happening in this image?",
    "What time of day does this image appear to be taken? Answer briefly.",
    "What is the weather like in this image? Answer briefly.",
    "What emotion or mood does this image convey? Answer briefly.",
]

# 用于负样本的不相关名词
DISTRACTOR_NOUNS = [
    "elephant", "airplane", "bicycle", "laptop", "guitar", "pizza",
    "train", "umbrella", "snowboard", "telescope", "helicopter",
    "violin", "microwave", "penguin", "cactus", "lighthouse",
    "submarine", "volcano", "dinosaur", "spaceship",
]


# ================================================================
# Caption → 名词提取 (简易规则)
# ================================================================

def extract_nouns_from_caption(caption: str) -> List[str]:
    """从 caption 中用简单规则提取名词短语。"""
    caption_lower = caption.lower().strip().rstrip(".")

    # "a/an/the + (adj)* + noun" 模式
    pattern = r'\b(?:a|an|the|some|two|three|four|five|several|many)\s+(?:\w+\s+){0,2}(\w+)'
    matches = re.findall(pattern, caption_lower)

    stopwords = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "shall", "the", "a", "an", "and", "or",
        "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "it", "its", "this", "that", "these", "those", "there", "here",
        "very", "more", "most", "also", "just", "only", "not", "no",
        "way", "set", "lot", "top", "side", "front", "back", "other",
    }

    nouns = []
    for w in matches:
        w = w.strip()
        if len(w) > 2 and w not in stopwords and w.isalpha():
            nouns.append(w)
    return list(dict.fromkeys(nouns))


def generate_questions_for_caption(
    caption: str,
    rng: random.Random,
) -> List[Dict]:
    """基于一条 caption 生成多种类型的 VQA 问题。"""
    questions = []
    nouns = extract_nouns_from_caption(caption)

    # 1. 描述类 (高权重, 稳定)
    questions.append({
        "question": rng.choice(DESCRIBE_TEMPLATES),
        "question_type": "description",
        "answer_hint": caption,
    })

    if nouns:
        noun = rng.choice(nouns)
        # 2. 存在性 — 正样本
        questions.append({
            "question": rng.choice(EXISTENCE_TEMPLATES_POS).format(noun=noun),
            "question_type": "existence",
            "answer_hint": "yes",
        })
        # 3. 属性
        questions.append({
            "question": rng.choice(ATTRIBUTE_TEMPLATES).format(noun=noun),
            "question_type": "attribute",
            "answer_hint": None,
        })
        # 4. 位置
        questions.append({
            "question": rng.choice(POSITION_TEMPLATES).format(noun=noun),
            "question_type": "position",
            "answer_hint": None,
        })

    if len(nouns) >= 2:
        # 5. 计数 (有多个名词时)
        noun = rng.choice(nouns)
        questions.append({
            "question": rng.choice(COUNT_TEMPLATES).format(noun=noun),
            "question_type": "count",
            "answer_hint": None,
        })

    # 6. 存在性 — 负样本
    available_distractors = [n for n in DISTRACTOR_NOUNS if n not in caption.lower()]
    if available_distractors:
        neg_noun = rng.choice(available_distractors)
        questions.append({
            "question": rng.choice(EXISTENCE_TEMPLATES_NEG).format(noun=neg_noun),
            "question_type": "existence_neg",
            "answer_hint": "no",
        })

    # 7. OCR (低概率)
    if rng.random() < 0.15:
        questions.append({
            "question": rng.choice(OCR_TEMPLATES),
            "question_type": "ocr",
            "answer_hint": None,
        })

    # 8. 常识推理 (低概率)
    if rng.random() < 0.2:
        questions.append({
            "question": rng.choice(REASONING_TEMPLATES),
            "question_type": "reasoning",
            "answer_hint": None,
        })

    return questions


# ================================================================
# Flickr8K 加载
# ================================================================

def load_flickr8k(flickr8k_root: str) -> List[Dict]:
    """加载 Flickr8K 数据，返回 [{image_path, captions: [str]}]。"""
    root = Path(flickr8k_root)
    captions_file = root / "captions.txt"
    images_dir = root / "Images"

    if not captions_file.exists():
        raise FileNotFoundError(f"captions.txt not found in {root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images/ directory not found in {root}")

    img_captions = defaultdict(list)
    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                img_name, caption = row[0], row[1]
                img_path = images_dir / img_name
                if img_path.exists():
                    img_captions[str(img_path)].append(caption.strip())

    result = []
    for img_path, captions in img_captions.items():
        result.append({"image_path": img_path, "captions": captions})
    return result


# ================================================================
# 主构建流程
# ================================================================

def build_calibration_dataset(
    flickr8k_root: str,
    num_samples: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """
    构建校准数据集。

    策略：
    1. 从 Flickr8K 加载所有图文对
    2. 随机采样图片子集
    3. 对每张图片基于 caption 生成多类型 VQA 问题
    4. 均衡采样各问题类型，凑够 num_samples 条
    """
    rng = random.Random(seed)

    print(f"Loading Flickr8K from {flickr8k_root} ...")
    all_items = load_flickr8k(flickr8k_root)
    print(f"  Loaded {len(all_items)} images with captions")

    rng.shuffle(all_items)

    # 为每张图生成候选问题
    all_candidates = []
    for item in all_items:
        caption = rng.choice(item["captions"])
        questions = generate_questions_for_caption(caption, rng)
        for q in questions:
            all_candidates.append({
                "image_path": item["image_path"],
                "caption": caption,
                "question": q["question"],
                "question_type": q["question_type"],
                "answer_hint": q.get("answer_hint"),
            })

    print(f"  Generated {len(all_candidates)} candidate QA pairs")

    # 按问题类型统计
    type_counts = defaultdict(list)
    for c in all_candidates:
        type_counts[c["question_type"]].append(c)

    print("  Question type distribution (before balancing):")
    for qt, items in sorted(type_counts.items()):
        print(f"    {qt}: {len(items)}")

    # 均衡采样：按类型轮流抽取
    types = sorted(type_counts.keys())
    for qt in types:
        rng.shuffle(type_counts[qt])

    selected = []
    type_indices = {qt: 0 for qt in types}
    seen_pairs = set()

    while len(selected) < num_samples:
        added_this_round = False
        for qt in types:
            if len(selected) >= num_samples:
                break
            pool = type_counts[qt]
            idx = type_indices[qt]
            while idx < len(pool):
                candidate = pool[idx]
                idx += 1
                pair_key = (candidate["image_path"], candidate["question"])
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    selected.append(candidate)
                    type_indices[qt] = idx
                    added_this_round = True
                    break
            type_indices[qt] = idx
        if not added_this_round:
            break

    # 如果均衡采样不够，补充剩余
    if len(selected) < num_samples:
        remaining = [c for c in all_candidates
                     if (c["image_path"], c["question"]) not in seen_pairs]
        rng.shuffle(remaining)
        for c in remaining[:num_samples - len(selected)]:
            selected.append(c)

    rng.shuffle(selected)

    print(f"\n  Final calibration dataset: {len(selected)} samples")
    final_type_counts = defaultdict(int)
    for s in selected:
        final_type_counts[s["question_type"]] += 1
    print("  Question type distribution (after balancing):")
    for qt, cnt in sorted(final_type_counts.items()):
        print(f"    {qt}: {cnt}")

    unique_images = len(set(s["image_path"] for s in selected))
    print(f"  Unique images used: {unique_images}")

    return selected


def save_calibration_dataset(
    samples: List[Dict],
    output_dir: str,
    run_date: str = None,
):
    """保存校准数据集为 JSON。"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_{run_date}" if run_date else ""
    filename = f"calibration_dataset_{len(samples)}samples{suffix}.json"
    filepath = out_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": len(samples),
            "source": "flickr8k",
            "description": "Multi-type VQA calibration dataset for quantization",
            "question_types": list(set(s["question_type"] for s in samples)),
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {filepath}")
    print(f"  Size: {filepath.stat().st_size / 1024:.1f} KB")

    # 同时保存一个 latest 软链接
    latest = out_path / f"calibration_dataset_latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(filepath.name)
    print(f"  Symlink: {latest} -> {filepath.name}")

    return str(filepath)


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build multi-type VQA calibration dataset from Flickr8K"
    )
    parser.add_argument("--flickr8k_root", type=str,
                        default="/data/14thdd/users/yongsencheng/Bagel/data/flickr8k",
                        help="Path to Flickr8K dataset root (with Images/ and captions.txt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: quantization_outputs/calibration_data)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of calibration samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_date", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).resolve().parent.parent
            / "quantization_outputs" / "calibration_data"
        )

    from datetime import datetime
    if args.run_date is None:
        args.run_date = datetime.now().strftime("%Y%m%d")

    print("=" * 60)
    print("Build Calibration Dataset (Flickr8K → Multi-Type VQA)")
    print("=" * 60)
    print(f"  Source:      {args.flickr8k_root}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Seed:        {args.seed}")
    print(f"  Run date:    {args.run_date}")

    samples = build_calibration_dataset(
        flickr8k_root=args.flickr8k_root,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    save_calibration_dataset(
        samples=samples,
        output_dir=args.output_dir,
        run_date=args.run_date,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
