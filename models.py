"""Lazy local model loading, with one model resident at a time."""

from __future__ import annotations

import gc
import os
import threading


IMAGE_MODEL_ID = "stabilityai/sd-turbo"
CHAT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class ModelHub:
    def __init__(self):
        self.image_model_id = IMAGE_MODEL_ID
        self.chat_model_id = os.environ.get("CHAT_MODEL_ID", CHAT_MODEL_ID)
        self._image_pipeline = None
        self._chat_tokenizer = None
        self._chat_model = None
        self._lock = threading.RLock()

    def _unload(self):
        self._image_pipeline = None
        self._chat_tokenizer = None
        self._chat_model = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _image(self):
        if self._image_pipeline is None:
            self._unload()
            import torch
            from diffusers import AutoPipelineForText2Image

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            options = {
                "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
                "use_safetensors": True,
            }
            if device != "cpu":
                options["variant"] = "fp16"
            self._image_pipeline = AutoPipelineForText2Image.from_pretrained(
                self.image_model_id, **options
            ).to(device)
        return self._image_pipeline

    def generate_image(self, prompt: str, *, steps: int, seed: int):
        with self._lock:
            import torch

            pipeline = self._image()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.inference_mode():
                return pipeline(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    width=512,
                    height=512,
                    generator=generator,
                ).images[0]

    def _chat(self):
        if self._chat_model is None:
            self._unload()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._chat_tokenizer = AutoTokenizer.from_pretrained(self.chat_model_id)
            self._chat_model = AutoModelForCausalLM.from_pretrained(
                self.chat_model_id, torch_dtype=torch.float32, use_safetensors=True
            ).to("cpu")
            self._chat_model.eval()
        return self._chat_tokenizer, self._chat_model

    def reply(self, messages: list[dict]) -> str:
        with self._lock:
            import torch

            tokenizer, model = self._chat()
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()
            return answer or "I couldn't form a reply. Please try again."
