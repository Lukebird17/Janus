"""
Stage 3 (Janus-Pro-7B): Quantized Model Benchmark Evaluation

支持 MME / MMVP / MMMU 理解类基准，以及基于官方 Janus T2I 流程的 GenEval 生成评测
（复用 Bagel 仓库中的 GenEval 数据与打分脚本）。

Usage:
    python stage3_test.py \
        --model_path /data/user/honglianglu/Bagel/models/Janus-Pro-7B \
        --stage2_config /path/to/quant_config.json \
        --gptq_hessian_index /path/to/gptq_hessian_index_latest.json \
        --smoothquant_stats /path/to/smoothquant_stats_latest.pt \
        --awq_stats /path/to/awq_stats_latest.pt \
        --benchmarks mme mmvp mmmu geneval
"""

import os
import sys
import json
import csv
import gc
import re
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
import argparse

_current_dir = Path(__file__).resolve().parent
_quant_dir = _current_dir.parent
_bagel_dir = Path("/data/user/honglianglu/Bagel")

if str(_quant_dir) not in sys.path:
    sys.path.insert(0, str(_quant_dir))

from layers.hybrid_quant_linear import HybridQuantLinear
from utils.model_loader import JanusModelLoader


def post_processing(response: str) -> str:
    response = response.replace('\n', '')
    response = response.replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    response = re.sub(r'[\u4e00-\u9fa5]', '', response)
    return response


class _LazyHessianProvider:
    """Lazy-load GPTQ Hessian data for quantization application."""

    def __init__(
        self,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        damp_ratio: float = 0.01,
    ):
        self._cache = OrderedDict()
        self.damp_ratio = damp_ratio
        self.hessian_index = None
        self.smooth_data = None
        self.awq_data = None

        if gptq_hessian_index and Path(gptq_hessian_index).exists():
            with open(gptq_hessian_index, "r") as f:
                idx = json.load(f)
            self.hessian_index = idx.get("layers", {})

        if smoothquant_stats and Path(smoothquant_stats).exists():
            self.smooth_data = torch.load(smoothquant_stats, map_location="cpu")

        if awq_stats and Path(awq_stats).exists():
            self.awq_data = torch.load(awq_stats, map_location="cpu")

    def __contains__(self, layer_name: str) -> bool:
        if self.hessian_index and layer_name in self.hessian_index:
            return True
        return False

    def get_activation(self, layer_name: str) -> Optional[torch.Tensor]:
        if layer_name in self._cache:
            self._cache.move_to_end(layer_name)
            return self._cache[layer_name]
        if self.hessian_index and layer_name in self.hessian_index:
            info = self.hessian_index[layer_name]
            data = torch.load(info["path"], map_location="cpu", weights_only=True)
            H_sum = data["hessian_sum"].double()
            n = data["nsamples"]
            act = self._reconstruct(H_sum, n)
            self._cache[layer_name] = act
            if len(self._cache) > 20:
                self._cache.popitem(last=False)
            return act
        return None

    def _reconstruct(self, H_sum, n):
        D = H_sum.shape[0]
        damp = self.damp_ratio * H_sum.diagonal().mean()
        H_reg = H_sum + damp * torch.eye(D, dtype=H_sum.dtype)
        try:
            L = torch.linalg.cholesky(H_reg)
            return L.t().float()
        except torch.linalg.LinAlgError:
            eigvals, eigvecs = torch.linalg.eigh(H_reg)
            eigvals = eigvals.clamp(min=0)
            return (eigvecs * eigvals.sqrt().unsqueeze(0)).t().float()

    def get_channel_max(self, layer_name: str) -> Optional[torch.Tensor]:
        if self.smooth_data and layer_name in self.smooth_data:
            return self.smooth_data[layer_name]["act_channel_max"]
        if self.awq_data and layer_name in self.awq_data:
            return self.awq_data[layer_name]["channel_max"]
        return None

    def get_channel_mean(self, layer_name: str) -> Optional[torch.Tensor]:
        if self.awq_data and layer_name in self.awq_data:
            return self.awq_data[layer_name]["channel_mean"]
        return None


class QuantizedModelEvaluator:
    """Janus-Pro-7B 量化模型评估器"""

    def __init__(
        self,
        model_path: str,
        stage2_config: str,
        gptq_hessian_index: Optional[str] = None,
        smoothquant_stats: Optional[str] = None,
        awq_stats: Optional[str] = None,
        output_dir: Optional[str] = None,
        gpu_ids: Optional[str] = None,
        data_root: Optional[str] = None,
    ):
        self.model_path = model_path
        self.stage2_config_file = Path(stage2_config)
        self.gpu_ids = gpu_ids
        self.data_root = Path(data_root) if data_root else _bagel_dir / "data"

        self.config_name = self.stage2_config_file.stem
        if output_dir:
            self.config_output_dir = Path(output_dir)
        else:
            self.config_output_dir = (
                _quant_dir / "quantization_outputs" / "eval" / self.config_name
            )
        self.config_output_dir.mkdir(parents=True, exist_ok=True)

        self.activation_provider = _LazyHessianProvider(
            gptq_hessian_index=gptq_hessian_index,
            smoothquant_stats=smoothquant_stats,
            awq_stats=awq_stats,
        )

        self.model = None
        self.processor = None
        self.tokenizer = None

        # GenEval（数据与评估脚本位于 Bagel 仓库）
        self.geneval_batch_size = 1
        self.geneval_num_images_per_prompt = 1
        self.geneval_native_size = 384
        self.geneval_resolution = 384
        self.geneval_cfg_scale = 5.0
        self.geneval_temperature = 1.0
        self.geneval_prompts_file = _bagel_dir / "eval" / "gen" / "geneval" / "prompts" / "evaluation_metadata.jsonl"
        self.geneval_model_path = _bagel_dir / "eval" / "gen" / "geneval" / "model"

    def _load_quant_config(self) -> Dict:
        print(f"\n  Loading quantization config: {self.stage2_config_file}")
        with open(self.stage2_config_file, 'r') as f:
            raw = json.load(f)
        if "layers" in raw:
            return raw["layers"]
        if "layer_configs" in raw:
            return raw["layer_configs"]
        return raw

    @staticmethod
    def _get_module(model, name: str):
        parts = name.split(".")
        m = model
        for part in parts:
            if hasattr(m, part):
                m = getattr(m, part)
            else:
                return None
        return m

    @staticmethod
    def _replace_module(model, name: str, new_module):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _apply_quantization(self, model, layer_configs: Dict):
        total = len(layer_configs)
        print(f"\n  Applying quantization to {total} layers...")
        if total == 0:
            print("  Empty config - evaluating original FP16 model")
            return

        applied = 0
        for layer_name, config in tqdm(layer_configs.items(), desc="  Quantize"):
            module = self._get_module(model, layer_name)
            if module is None or not isinstance(module, nn.Linear):
                continue

            device = module.weight.device
            dtype = module.weight.dtype
            has_bias = module.bias is not None

            block_size_weight = config.get('block_size_weight', config.get('block_size', 256))
            block_size_act = config.get('block_size_act', config.get('block_size', 256))

            quant_layer = HybridQuantLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=has_bias,
                weight_bit=config.get('weight_bit', 4),
                act_bit=config.get('act_bit', 4),
                quant_percentile=config.get('quant_percentile', 0.999999),
                act_unsigned=config.get('act_unsigned', True),
                use_sparse=config.get('use_sparse', False),
                sparse_ratio=config.get('sparse_ratio', 0.0),
                sparse_threshold=config.get('sparse_threshold', None),
                use_smoothquant=config.get('use_smoothquant', False),
                smoothquant_alpha=config.get('smoothquant_alpha', 0.5),
                use_svd=config.get('use_svd', False),
                svd_rank=config.get('svd_rank', 0),
                use_block_quant=config.get('use_block_quant', False),
                use_block_quant_act=config.get('use_block_quant_act', False),
                block_size_weight=block_size_weight,
                block_size_act=block_size_act,
                use_gptq=config.get('use_gptq', False),
                gptq_group_size=config.get('gptq_group_size', 64),
                gptq_damp_percentage=config.get('gptq_damp_percentage', 0.01),
                gptq_block_size=config.get('gptq_block_size', 128),
                use_awq=config.get('use_awq', False),
                awq_alpha=config.get('awq_alpha', 0.5),
                awq_n_grid=config.get('awq_n_grid', 20),
                device=device,
                dtype=dtype,
            )
            quant_layer.weight.data = module.weight.data.clone()
            if has_bias:
                quant_layer.bias.data = module.bias.data.clone()

            act_data = None
            if layer_name in self.activation_provider:
                act_data = self.activation_provider.get_activation(layer_name)
            quant_layer.prepare_weight(
                activation_data=act_data, layer_name=layer_name, verbose=False
            )
            self._replace_module(model, layer_name, quant_layer)
            applied += 1
            del act_data
            if applied % 30 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        print(f"  Applied quantization to {applied}/{total} layers")

    def load_and_quantize_model(self):
        print(f"\n  Loading Janus-Pro-7B from {self.model_path} ...")
        loader = JanusModelLoader(
            model_path=self.model_path,
            gpu_ids=self.gpu_ids,
        )
        components = loader.load_all()
        self.model = components["model"]
        self.model.eval()
        self.processor = components["processor"]
        self.tokenizer = components["tokenizer"]

        layer_configs = self._load_quant_config()
        self._apply_quantization(self.model, layer_configs)
        return self.model

    def _chat(self, image, prompt, max_new_tokens=512):
        return JanusModelLoader.chat(
            self.model, self.processor, image, prompt,
            max_new_tokens=max_new_tokens,
        )

    # ================================================================
    # MME Evaluation
    # ================================================================
    def run_mme_evaluation(self):
        print(f"\n[MME] Running MME Evaluation...")
        mme_data_root = self.data_root / "mme" / "MME_Benchmark_release_version"
        mme_questions_root = _bagel_dir / "eval" / "vlm" / "eval" / "mme" / "Your_Results"

        if not mme_data_root.exists():
            print(f"  MME data not found at {mme_data_root}")
            return None
        if not mme_questions_root.exists():
            print(f"  MME questions not found at {mme_questions_root}")
            return None

        mme_output_dir = self.config_output_dir / "mme_results"
        mme_output_dir.mkdir(parents=True, exist_ok=True)

        prompt = 'Answer the question using a single word or phrase.'
        categories_to_process = []

        for question_file in mme_questions_root.iterdir():
            if not question_file.name.endswith('.txt'):
                continue
            category = question_file.stem
            output_file = mme_output_dir / question_file.name
            if output_file.exists():
                with open(question_file, 'r') as f:
                    nq = len(f.readlines())
                with open(output_file, 'r') as f:
                    na = len(f.readlines())
                if nq == na:
                    print(f"  {category} complete ({na}/{nq}), skipping")
                    continue
                else:
                    output_file.unlink()
            categories_to_process.append(question_file)

        if not categories_to_process:
            print(f"  All MME categories complete, skipping generation")
        else:
            print(f"  Processing {len(categories_to_process)} categories...")
            for question_file in categories_to_process:
                category = question_file.stem
                output_file = mme_output_dir / question_file.name
                with open(question_file, 'r') as fin:
                    lines = fin.readlines()
                with open(output_file, 'w') as fout:
                    for line in tqdm(lines, desc=f"  {category}"):
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            continue
                        img, question, gt = parts[0], parts[1], parts[2]
                        question = question + ' ' + prompt

                        img_path = mme_data_root / category / img
                        if not img_path.exists():
                            img_path = mme_data_root / category / "images" / img
                        image = Image.open(img_path).convert('RGB')

                        response = self._chat(image, question, max_new_tokens=20)
                        response = post_processing(response)
                        print(img, question, gt, response, sep='\t', file=fout)

        # Calculate scores using Bagel's MME calculator
        results_file = mme_output_dir / "results.txt"
        if not results_file.exists():
            print(f"\n  Calculating MME scores...")
            import subprocess
            subprocess.run([
                sys.executable, "-m", "eval.vlm.eval.mme.calculation",
                "--out-dir", str(mme_output_dir.resolve())
            ], cwd=str(_bagel_dir), check=True)
        else:
            print(f"  MME scores already calculated")

        if results_file.exists():
            with open(results_file, 'r') as f:
                print(f"  {f.read().strip()[:500]}")
        return {"output_dir": str(mme_output_dir)}

    # ================================================================
    # MMVP Evaluation
    # ================================================================
    def run_mmvp_evaluation(self):
        print(f"\n[MMVP] Running MMVP Evaluation...")
        mmvp_root = self.data_root / "MMVP"
        csv_path = mmvp_root / "Questions.csv"
        if not csv_path.exists():
            print(f"  MMVP data not found at {csv_path}")
            return None

        mmvp_output_dir = self.config_output_dir / "mmvp_results"
        mmvp_output_dir.mkdir(parents=True, exist_ok=True)

        results_file = mmvp_output_dir / "results.txt"
        if results_file.exists():
            print(f"  MMVP results already exist, skipping.")
            with open(results_file, 'r') as f:
                print(f"  {f.read().strip()}")
            return {"output_dir": str(mmvp_output_dir)}

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]

        prompt = "Answer with the option's letter from the given choices directly."
        outputs = []

        for row in tqdm(data, desc="  MMVP"):
            data_id = row.get('lndex') or row.get('Index')
            question = row['Question']

            options_raw = row['Options'].split('(b)')
            options_raw[0] = options_raw[0].replace('(a)', '').strip()
            if len(options_raw) > 1:
                options_raw[1] = options_raw[1].strip()

            gt = row.get('Correct Answer', '')
            gt = gt.replace('(a)', 'A').replace('(b)', 'B')

            choice_list = []
            for i, c in enumerate(options_raw):
                letter = chr(ord('A') + i)
                choice_list.append(f'{letter}. {c.strip()}')
            choice_txt = '\n'.join(choice_list)

            full_question = question
            if choice_txt:
                full_question += '\n' + choice_txt
            full_question += '\n' + prompt

            img_path = mmvp_root / 'MMVP Images' / f'{data_id}.jpg'
            image = Image.open(img_path).convert('RGB')

            response = self._chat(image, full_question, max_new_tokens=100)
            pred = response.strip()
            option_keys = [chr(ord('A') + i) for i in range(len(options_raw))]
            if len(pred) >= 1 and pred[0] in option_keys:
                pred = pred[0]
            else:
                for k, c in zip(option_keys, options_raw):
                    if c.strip().lower() in pred.lower():
                        pred = k
                        break

            outputs.append({
                'data_id': data_id, 'question': question,
                'answer': pred, 'gt_answer': gt,
            })

        jsonl_path = mmvp_output_dir / "results.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in outputs:
                f.write(json.dumps(item) + '\n')

        num_correct, num_total = 0, 0
        idx, round_correct = 0, 0
        for item in outputs:
            idx += 1
            if item['answer'] == item['gt_answer']:
                round_correct += 1
            if idx == 2:
                idx = 0
                if round_correct == 2:
                    num_correct += 1
                round_correct = 0
                num_total += 1

        accuracy = num_correct / num_total if num_total > 0 else 0.0
        result_text = f"MMVP pair-wise accuracy: {accuracy:.4f} ({num_correct}/{num_total})"
        print(f"  {result_text}")
        with open(results_file, 'w') as f:
            f.write(result_text + '\n')

        return {"output_dir": str(mmvp_output_dir), "accuracy": accuracy}

    # ================================================================
    # MMMU Evaluation
    # ================================================================
    def run_mmmu_evaluation(self):
        print(f"\n[MMMU] Running MMMU Evaluation...")
        mmmu_output_dir = self.config_output_dir / "mmmu_results"
        mmmu_output_dir.mkdir(parents=True, exist_ok=True)

        results_file = mmmu_output_dir / "results.txt"
        if results_file.exists():
            print(f"  MMMU results already exist, skipping.")
            with open(results_file, 'r') as f:
                for line in f:
                    print(f"  {line.strip()}")
            return {"output_dir": str(mmmu_output_dir)}

        sys.path.insert(0, str(_bagel_dir))
        from eval.vlm.eval.mmmu.data_utils import (
            CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT, process_single_sample
        )
        from eval.vlm.eval.mmmu.eval_utils import (
            evaluate as mmmu_evaluate,
            calculate_ins_level_acc,
            parse_open_response,
        )
        from datasets import concatenate_datasets, load_dataset

        mmmu_root = 'MMMU/MMMU'
        cache_dir = str(_bagel_dir / "eval" / "vlm" / "data" / "MMMU")
        split = 'validation'

        print(f"  Loading MMMU {split} dataset...")
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values(), desc="  Loading subjects"):
            sub_dataset = load_dataset(mmmu_root, subject, split=split, cache_dir=cache_dir)
            sub_dataset_list.append(sub_dataset)
        dataset = concatenate_datasets(sub_dataset_list)
        print(f"  Loaded {len(dataset)} samples")

        prompts = {
            'multiple-choice': "Answer with the option's letter from the given choices directly.",
            'open': 'Answer the question using a single word or phrase.'
        }

        prediction_dict = {}
        outputs_for_jsonl = []

        for i in tqdm(range(len(dataset)), desc="  MMMU inference"):
            raw_data = dataset[i]
            sample = process_single_sample(raw_data)
            data_id = sample['id']
            question = sample['question'].strip()
            pil_images = sample['image']
            question_type = sample['question_type']

            choices = eval(sample['options'])
            gt_answer = sample.get('answer', None)

            choice_list = []
            options = {}
            letters = 'ABCDEFGHIJKLM'
            for ci, c in enumerate(choices):
                choice_list.append(f'{letters[ci]}. {c.strip()}')
                options[letters[ci]] = c.strip()
            choice_txt = '\n'.join(choice_list)

            image = None
            for pi, pil_image in enumerate(pil_images):
                if pil_image is not None:
                    if pi == 0:
                        image = pil_image.convert('RGB').resize(
                            (pil_image.width * 2, pil_image.height * 2), Image.BILINEAR
                        )
                    break

            full_question = question
            if choice_txt:
                full_question += '\n' + choice_txt
            full_question += '\n' + prompts[question_type]
            full_question = full_question.strip()

            response = self._chat(image, full_question, max_new_tokens=10)
            pred = response.strip()
            option_keys = list(options.keys())
            if len(pred) >= 1 and pred[0] in option_keys:
                pred = pred[0]
            elif len(pred) == 0:
                pred = "C"
            else:
                for k, v in options.items():
                    if v.lower() in pred.lower():
                        pred = k
                        break

            prediction_dict[data_id] = pred
            outputs_for_jsonl.append({
                'data_id': data_id, 'question': question,
                'answer': pred, 'gt_answer': gt_answer,
            })

        pred_json_path = mmmu_output_dir / "prediction.json"
        with open(pred_json_path, 'w') as f:
            json.dump(prediction_dict, f, indent=2)

        jsonl_path = mmmu_output_dir / "results.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in outputs_for_jsonl:
                f.write(json.dumps(item) + '\n')

        print(f"  Evaluating MMMU predictions...")
        answer_path = _bagel_dir / "eval" / "vlm" / "eval" / "mmmu" / "answer_dict_val.json"
        with open(answer_path, 'r') as f:
            answer_dict = json.load(f)

        output_dict_w_cat = {}
        for did, parsed_pred in prediction_dict.items():
            category = '_'.join(did.split('_')[1:-1])
            if category not in output_dict_w_cat:
                output_dict_w_cat[category] = {}
            output_dict_w_cat[category][did] = parsed_pred

        answer_dict_w_cat = {}
        for did, entry in answer_dict.items():
            category = '_'.join(did.split('_')[1:-1])
            if category not in answer_dict_w_cat:
                answer_dict_w_cat[category] = {}
            answer_dict_w_cat[category][did] = entry

        evaluation_result = {}
        for category in CAT_SHORT2LONG.values():
            try:
                cat_outputs = output_dict_w_cat[category]
                cat_answers = answer_dict_w_cat[category]
            except KeyError:
                continue
            examples_to_eval = []
            for did, parsed_pred in cat_outputs.items():
                q_type = cat_answers[did]['question_type']
                if q_type != 'multiple-choice':
                    parsed_pred = parse_open_response(parsed_pred)
                examples_to_eval.append({
                    'id': did,
                    'question_type': q_type,
                    'answer': cat_answers[did]['ground_truth'],
                    'parsed_pred': parsed_pred,
                })
            judge_dict, metric_dict = mmmu_evaluate(examples_to_eval)
            metric_dict['num_example'] = len(examples_to_eval)
            evaluation_result[category] = metric_dict

        printable_results = {}
        for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
            in_domain_cat_results = {}
            for cat_name in in_domain_cats:
                if cat_name in evaluation_result:
                    in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
            in_domain_data_num = sum(r['num_example'] for r in in_domain_cat_results.values())
            printable_results['Overall-' + domain] = {
                'num': int(in_domain_data_num), 'acc': round(in_domain_ins_acc, 3)
            }
            for cat_name, cat_results in in_domain_cat_results.items():
                printable_results[cat_name] = {
                    'num': int(cat_results['num_example']), 'acc': round(cat_results['acc'], 3)
                }

        all_ins_acc = calculate_ins_level_acc(evaluation_result)
        total_num = sum(r['num_example'] for r in evaluation_result.values())
        printable_results['Overall'] = {'num': total_num, 'acc': round(all_ins_acc, 3)}

        with open(results_file, 'w') as f:
            for key, value in printable_results.items():
                line = f"{key}: num={value['num']}, acc={value['acc']}"
                f.write(line + '\n')
                print(f"  {line}")

        overall_acc = printable_results['Overall']['acc']
        print(f"\n  MMMU Overall Accuracy: {overall_acc}")
        return {"output_dir": str(mmmu_output_dir), "accuracy": overall_acc}

    # ================================================================
    # GenEval (Janus T2I + Bagel 官方打分脚本)
    # ================================================================
    @torch.inference_mode()
    def _janus_generate_parallel(
        self,
        prompt: str,
        parallel_size: int,
        cfg_weight: float,
        temperature: float,
        width: int,
        height: int,
        image_token_num_per_image: int = 576,
        patch_size: int = 16,
    ) -> List[Image.Image]:
        """Janus-Pro 并行文生图（与 demo/app_januspro.py 一致）。"""
        device = next(self.model.parameters()).device
        vl_chat_processor = self.processor
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt = self.model

        w_align = width // 16 * 16
        h_align = height // 16 * 16

        messages = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        text = text + vl_chat_processor.image_start_tag
        input_ids = torch.as_tensor(
            tokenizer.encode(text), dtype=torch.long, device=device
        )

        tokens = torch.zeros(
            (parallel_size * 2, len(input_ids)), dtype=torch.long, device=device
        )
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros(
            (parallel_size, image_token_num_per_image),
            dtype=torch.long,
            device=device,
        )

        pkv = None
        for ti in range(image_token_num_per_image):
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                outputs = vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=pkv,
                )
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, ti] = next_token.squeeze(dim=-1)
            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        patches = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[
                parallel_size,
                8,
                w_align // patch_size,
                h_align // patch_size,
            ],
        )

        dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        visual_img = np.zeros((parallel_size, w_align, h_align, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        out_res = self.geneval_resolution
        images: List[Image.Image] = []
        for i in range(parallel_size):
            pil = Image.fromarray(visual_img[i])
            if out_res and (pil.size[0] != out_res or pil.size[1] != out_res):
                pil = pil.resize((out_res, out_res), Image.LANCZOS)
            images.append(pil)
        return images

    def run_geneval_evaluation(self):
        print(f"\n[GenEval] Running GenEval Evaluation...")
        if not self.geneval_prompts_file.exists():
            print(f"  GenEval prompts file not found: {self.geneval_prompts_file}")
            return None

        geneval_output_dir = self.config_output_dir / "geneval_results"
        geneval_images_dir = geneval_output_dir
        geneval_images_dir.mkdir(parents=True, exist_ok=True)

        with open(self.geneval_prompts_file, "r") as f:
            metadatas = [json.loads(line) for line in f]

        total_prompts = len(metadatas)
        batch_size = max(1, self.geneval_batch_size)
        num_images_per_prompt = max(1, self.geneval_num_images_per_prompt)
        w = h = self.geneval_native_size
        cfg_w = self.geneval_cfg_scale
        temp = self.geneval_temperature

        print(f"  Output directory: {geneval_images_dir}")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Native size: {w}x{h}, output resize: {self.geneval_resolution}")

        prompts_to_process = []
        prompts_completed = []
        prompts_incomplete = []

        for idx in range(total_prompts):
            outpath = geneval_images_dir / f"{idx:05d}"
            sample_path = outpath / "samples"
            all_images_exist = True
            for img_idx in range(num_images_per_prompt):
                if not (sample_path / f"{img_idx:05d}.png").exists():
                    all_images_exist = False
                    break
            if all_images_exist:
                prompts_completed.append(idx)
            else:
                if sample_path.exists():
                    shutil.rmtree(outpath)
                    print(f"  Prompt {idx} incomplete, removed and will regenerate")
                    prompts_incomplete.append(idx)
                prompts_to_process.append(idx)

        print(
            f"  {len(prompts_completed)} complete, "
            f"{len(prompts_incomplete)} incomplete, {len(prompts_to_process)} to generate"
        )

        for idx in prompts_to_process:
            metadata = metadatas[idx]
            prompt = metadata["prompt"]
            outpath = geneval_images_dir / f"{idx:05d}"
            outpath.mkdir(parents=True, exist_ok=True)
            sample_path = outpath / "samples"
            sample_path.mkdir(parents=True, exist_ok=True)
            with open(outpath / "metadata.jsonl", "w") as f:
                json.dump(metadata, f)

            image_list = []
            num_batches = (num_images_per_prompt + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                current_bs = min(
                    batch_size, num_images_per_prompt - len(image_list)
                )
                batch_images = self._janus_generate_parallel(
                    prompt,
                    parallel_size=current_bs,
                    cfg_weight=cfg_w,
                    temperature=temp,
                    width=w,
                    height=h,
                )
                image_list.extend(batch_images)

            for img_idx, image in enumerate(image_list):
                bbox = image.getbbox()
                if bbox:
                    image = image.crop(bbox)
                image.save(sample_path / f"{img_idx:05d}.png")

            print(
                f"  Generated {len(image_list)} images for prompt "
                f"{prompts_to_process.index(idx)+1}/{len(prompts_to_process)}: "
                f"'{prompt[:50]}...'"
            )

        results_file = geneval_output_dir / "results.jsonl"
        summary_file = geneval_output_dir / "geneval_results.txt"

        if results_file.exists():
            print(f"\n  GenEval evaluation results already exist: {results_file}")
        else:
            print(f"\n  Running GenEval image scoring (torchrun)...")
            geneval_images_dir_abs = geneval_images_dir.resolve()
            results_file_abs = results_file.resolve()
            geneval_model_path_abs = Path(self.geneval_model_path).resolve()

            if self.gpu_ids and isinstance(self.gpu_ids, str):
                num_gpus = len(self.gpu_ids.split(","))
            else:
                num_gpus = torch.cuda.device_count()
            nproc = min(2, max(1, num_gpus))

            subprocess.run(
                [
                    "torchrun",
                    "--nnodes=1",
                    "--node_rank=0",
                    f"--nproc_per_node={nproc}",
                    "--master_addr=127.0.0.1",
                    "--master_port=29511",
                    str(
                        _bagel_dir
                        / "eval"
                        / "gen"
                        / "geneval"
                        / "evaluation"
                        / "evaluate_images_mp.py"
                    ),
                    str(geneval_images_dir_abs),
                    "--outfile",
                    str(results_file_abs),
                    "--model-path",
                    str(geneval_model_path_abs),
                ],
                check=True,
            )

        if summary_file.exists():
            print(f"\n  GenEval summary already exists: {summary_file}")
        else:
            print(f"\n  Generating GenEval summary...")
            subprocess.run(
                [
                    sys.executable,
                    str(
                        _bagel_dir
                        / "eval"
                        / "gen"
                        / "geneval"
                        / "evaluation"
                        / "summary_scores.py"
                    ),
                    str(results_file.resolve()),
                ],
                check=True,
            )

        if summary_file.exists():
            with open(summary_file, "r") as sf:
                print(sf.read())

        return {"output_dir": str(geneval_output_dir), "total_prompts": total_prompts}

    # ================================================================
    # Main run
    # ================================================================
    def run(self, benchmarks: List[str] = None):
        if benchmarks is None:
            benchmarks = ['mme']

        print(f"\n{'=' * 80}")
        print(f"Janus-Pro-7B Quantized Model Evaluation")
        print(f"{'=' * 80}")
        print(f"  Config: {self.config_name}")
        print(f"  Benchmarks: {', '.join(benchmarks)}")
        print(f"{'=' * 80}\n")

        model = self.load_and_quantize_model()
        results = {}

        if 'mme' in benchmarks:
            results['mme'] = self.run_mme_evaluation()
        if 'mmvp' in benchmarks:
            results['mmvp'] = self.run_mmvp_evaluation()
        if 'mmmu' in benchmarks:
            results['mmmu'] = self.run_mmmu_evaluation()
        if 'geneval' in benchmarks:
            results['geneval'] = self.run_geneval_evaluation()

        report = {
            'config_name': self.config_name,
            'model_path': self.model_path,
            'model_type': 'janus-pro-7b',
            'benchmarks': benchmarks,
            'results': results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        report_path = self.config_output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n{'=' * 80}")
        print(f"Evaluation Complete!")
        print(f"  Output: {self.config_output_dir}")
        print(f"  Report: {report_path}")
        print(f"{'=' * 80}\n")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 (Janus-Pro-7B): Quantized Model Evaluation"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--stage2_config", type=str, required=True,
                        help="Quantization config JSON from stage20/21")
    parser.add_argument("--gptq_hessian_index", type=str, default=None)
    parser.add_argument("--smoothquant_stats", type=str, default=None)
    parser.add_argument("--awq_stats", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for benchmark data (default: Bagel/data)")
    parser.add_argument("--benchmarks", nargs='+',
                        choices=['mme', 'mmvp', 'mmmu', 'geneval'],
                        default=['mme'],
                        help="Benchmarks to run")
    parser.add_argument("--geneval_resolution", type=int, default=384,
                        help="GenEval: resize generated images to this size (native Janus T2I is 384)")
    parser.add_argument("--geneval_native_size", type=int, default=384,
                        help="GenEval: Janus internal generation resolution (multiple of 16)")
    parser.add_argument("--geneval_cfg_scale", type=float, default=5.0,
                        help="GenEval: classifier-free guidance weight")
    parser.add_argument("--geneval_temperature", type=float, default=1.0,
                        help="GenEval: sampling temperature")
    args = parser.parse_args()

    evaluator = QuantizedModelEvaluator(
        model_path=args.model_path,
        stage2_config=args.stage2_config,
        gptq_hessian_index=args.gptq_hessian_index,
        smoothquant_stats=args.smoothquant_stats,
        awq_stats=args.awq_stats,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        data_root=args.data_root,
    )
    evaluator.geneval_resolution = args.geneval_resolution
    evaluator.geneval_native_size = args.geneval_native_size
    evaluator.geneval_cfg_scale = args.geneval_cfg_scale
    evaluator.geneval_temperature = args.geneval_temperature
    evaluator.run(benchmarks=args.benchmarks)


if __name__ == "__main__":
    main()
