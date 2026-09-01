import argparse
import inspect
import json
import os


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

            if "messages" not in row or not isinstance(row["messages"], list):
                raise ValueError(
                    f"Missing or invalid 'messages' field at line "
                    f"{line_number} in {path}"
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No SFT samples found in {path}")

    return rows


def build_text(example, tokenizer):
    messages = example["messages"]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        # Compatibility with tokenizer versions that do not expose
        # the Qwen3 enable_thinking argument.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return {"text": text}


def build_sft_config(SFTConfig, args):
    """Build an SFTConfig compatible with multiple TRL versions."""
    config_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "bf16": True,
        "fp16": False,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "remove_unused_columns": False,
        "dataset_text_field": "text",
        "packing": False,
        "seed": args.seed,
    }

    # TRL has used both names across versions.
    parameters = inspect.signature(SFTConfig).parameters
    if "max_length" in parameters:
        config_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in parameters:
        config_kwargs["max_seq_length"] = args.max_seq_length

    return SFTConfig(**config_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory-supervised LoRA SFT for the CRAFTS student model."
    )

    # Student backbone only. Teacher trajectories are generated during
    # the build stage and supplied through --data_path.
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2018)

    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_jsonl(args.data_path)
    print(f"[Distill] Loaded {len(rows)} samples from {args.data_path}")

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        lambda example: build_text(example, tokenizer),
        desc="Serialize teacher trajectories",
    )

    # QLoRA-style loading keeps the frozen student backbone memory-efficient;
    # only the LoRA adapters are optimized below.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = build_sft_config(
        SFTConfig=SFTConfig,
        args=args,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save only the trained PEFT adapter and tokenizer configuration.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[Distill] Saved student LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
