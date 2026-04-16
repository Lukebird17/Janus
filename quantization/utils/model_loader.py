"""
Janus-Pro-7B Model Loader for Quantization

Loads the Janus multi-modality model using the official Janus package interface.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Optional

JANUS_REPO_DIR = Path(__file__).resolve().parent.parent.parent


class JanusModelLoader:
    """Load Janus-Pro-7B model for quantization experiments."""

    def __init__(
        self,
        model_path: str,
        gpu_ids: Optional[str] = None,
        max_mem_per_gpu: str = "40GiB",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.max_mem_per_gpu = max_mem_per_gpu

        # 注意：若进程已执行过 torch.cuda 相关 API，此处再设置无效；调用方应在 __main__
        # 或类 __init__ 最早阶段设置 CUDA_VISIBLE_DEVICES（见 stage20/stage21）。
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
            print(f"Using GPUs: {gpu_ids}")

        if str(JANUS_REPO_DIR) not in sys.path:
            sys.path.insert(0, str(JANUS_REPO_DIR))

    def load_all(self):
        """Load model, processor, tokenizer.

        Returns:
            dict with keys: model, processor, tokenizer
        """
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        from transformers import AutoModelForCausalLM

        print(f"Loading VLChatProcessor from {self.model_path} ...")
        processor = VLChatProcessor.from_pretrained(self.model_path)
        tokenizer = processor.tokenizer

        print(f"Loading MultiModalityCausalLM from {self.model_path} ...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        model = model.to(self.dtype).cuda().eval()

        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Model loaded: {total_params:.2f}B params on {next(model.parameters()).device}")

        return {
            'model': model,
            'processor': processor,
            'tokenizer': tokenizer,
        }

    @staticmethod
    def get_language_model(model):
        """Get the LLM submodule from the full model."""
        return model.language_model

    @staticmethod
    def get_decoder_layers(model):
        """Get decoder layer modules."""
        return model.language_model.model.layers

    @staticmethod
    def get_num_decoder_layers(model):
        """Get number of decoder layers."""
        return len(model.language_model.model.layers)

    @staticmethod
    def _prepare_inputs(model, processor, image, prompt):
        """Prepare inputs_embeds and attention_mask for the language model.

        Returns:
            (inputs_embeds, attention_mask) on model device.
        """
        device = next(model.parameters()).device
        tokenizer = processor.tokenizer

        if image is not None:
            conversation = [
                {"role": "User", "content": f"<image_placeholder>\n{prompt}",
                 "images": ["__placeholder__"]},
                {"role": "Assistant", "content": ""},
            ]
            prepare_inputs = processor(
                conversations=conversation,
                images=[image],
                force_batchify=True,
            ).to(device)
        else:
            conversation = [
                {"role": "User", "content": prompt},
                {"role": "Assistant", "content": ""},
            ]
            prepare_inputs = processor(
                conversations=conversation,
                images=None,
                force_batchify=True,
            ).to(device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        return inputs_embeds, prepare_inputs.attention_mask

    @staticmethod
    def chat(model, processor, image, prompt, max_new_tokens=512):
        """Run multimodal understanding inference (with generation).

        Returns:
            str: model response
        """
        tokenizer = processor.tokenizer
        inputs_embeds, attention_mask = JanusModelLoader._prepare_inputs(
            model, processor, image, prompt,
        )
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    @staticmethod
    def forward_sample(model, processor, sample):
        """Run a single calibration sample through the model (prefill only).

        Used for activation collection and CKA computation.
        No autoregressive generation -- just one forward pass through all layers.
        """
        image = sample.get("image")
        prompt = sample.get("prompt", "")
        inputs_embeds, attention_mask = JanusModelLoader._prepare_inputs(
            model, processor, image, prompt,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
