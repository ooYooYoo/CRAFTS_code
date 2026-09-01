import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QwenLocal:
    """Local Qwen inference wrapper with 4-bit quantization."""

    def __init__(self, model_path: str, tokenizer_path: str = None, adapter_path: str = None):
        self.model_path = model_path
        tokenizer_path = tokenizer_path or model_path

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        device_map = os.environ.get("CRAFTS_DEVICE_MAP", "auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.requires_grad_(False)
        self.adapter_loaded = False
        if adapter_path is not None:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.adapter_loaded = True
            except Exception as exc:
                raise RuntimeError(
                    f"Required LoRA adapter failed to load from {adapter_path}: {exc}"
                ) from exc

        self.model.eval()
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if trainable:
            raise RuntimeError(
                f"Inference model unexpectedly has {trainable} trainable parameters"
            )

        self.gen_cfg = {
            "max_new_tokens": 2048,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
        }

    def set_generation_config(
        self,
        max_new_tokens: int = None,
        do_sample: bool = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
    ):
        if max_new_tokens is not None:
            self.gen_cfg["max_new_tokens"] = int(max_new_tokens)
        if do_sample is not None:
            self.gen_cfg["do_sample"] = bool(do_sample)
        if temperature is not None:
            self.gen_cfg["temperature"] = float(temperature)
        if top_p is not None:
            self.gen_cfg["top_p"] = float(top_p)
        if top_k is not None:
            self.gen_cfg["top_k"] = int(top_k)

    @torch.inference_mode()
    def chat(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg["max_new_tokens"],
                do_sample=self.gen_cfg["do_sample"],
                temperature=self.gen_cfg["temperature"],
                top_p=self.gen_cfg["top_p"],
                top_k=self.gen_cfg["top_k"],
            )
            generated_ids = output[0][inputs["input_ids"].shape[-1]:]
            return self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
        except Exception as exc:
            print("[warning] QwenLocal.chat exception:", repr(exc))
            return None
