"""
SupportIQ - LoRA Fine-Tuning Module
Fine-tunes Qwen/Qwen2.5-0.5B-Instruct using Hugging Face PEFT LoRA
for structured customer support message classification.

Author: SupportIQ ML Engineering Team
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import psutil
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)


SYSTEM_PROMPT = (
    "You are a customer support triage AI. Classify the customer support message into JSON "
    "with category, priority, and sentiment."
)


class SupportClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset that prepares inputs using the model's chat template and
    masks prompt tokens with -100 so loss is computed solely on the assistant response.
    """

    def __init__(self, data_path: str, tokenizer: Any, max_seq_length: int = 384, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.features = []

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if max_samples and max_samples < len(records):
            # Stratified subsampling across composite class
            from sklearn.model_selection import train_test_split
            labels = [f"{r.get('category')}_{r.get('priority')}_{r.get('sentiment')}" for r in records]
            records, _ = train_test_split(
                records,
                train_size=max_samples,
                random_state=42,
                stratify=labels
            )

        for rec in records:
            feature = self._process_record(rec)
            if feature is not None:
                self.features.append(feature)

    def _process_record(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["input"]},
            {"role": "assistant", "content": record["output"]}
        ]
        prompt_only_messages = messages[:2]

        # Render chat templates
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_seq_length,
            add_special_tokens=False
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

        # Mask prompt tokens with -100 (cross-entropy ignored index)
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


@dataclass
class DataCollatorForCompletionOnly:
    """
    Pads input_ids, attention_mask, and labels dynamically to the batch's max sequence length.
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        batch_labels = [f["labels"] for f in features]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded
        }


def resolve_model_path(model_path: str) -> str:
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct"))
    if model_path == "Qwen/Qwen2.5-0.5B-Instruct" and os.path.exists(local_dir):
        return local_dir
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="SupportIQ LoRA Fine-Tuning")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_dir", type=str, default="SupportIQ/data")
    parser.add_argument("--output_dir", type=str, default="SupportIQ/outputs")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional subset for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    args.model_name_or_path = resolve_model_path(args.model_name_or_path)
    set_seed(args.seed)

    print("=" * 70)
    print("SupportIQ – LoRA Fine-Tuning Pipeline")
    print("=" * 70)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Execution Device: {device.upper()}")
    if device == "cpu":
        num_threads = 8
        torch.set_num_threads(num_threads)
        print(f"Configured PyTorch CPU Intra-op Threads: {num_threads}")

    # Output paths
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    # 1. Load Tokenizer
    print(f"\n1. Loading Tokenizer: {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Datasets
    print("2. Preparing Datasets with Chat Template and Prompt Masking...")
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "validation.jsonl")

    train_dataset = SupportClassificationDataset(train_path, tokenizer, args.max_seq_length, max_samples=args.max_train_samples)
    val_dataset = SupportClassificationDataset(val_path, tokenizer, args.max_seq_length, max_samples=50 if device == "cpu" else None)

    print(f"   Train samples loaded: {len(train_dataset)}")
    print(f"   Validation samples loaded: {len(val_dataset)}")

    # 3. Load Base Model
    print(f"\n3. Loading Base Model: {args.model_name_or_path}...")
    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False  # Cache disabled for training backward pass

    # 4. LoRA Adapter Configuration
    print("\n4. Configuring LoRA Adapters...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / all_params
    print(f"   Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_pct:.3f}%)")
    print(f"   LoRA Hyperparameters: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"   Target Modules: {target_modules}")

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0
    )

    collator = DataCollatorForCompletionOnly(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )

    # 6. Execute Training
    print("\n5. Starting Training Execution...")
    start_time = time.time()
    train_result = trainer.train()
    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes).")

    # 7. Evaluate on Validation Set
    print("\n6. Running Final Validation Set Evaluation...")
    eval_metrics = trainer.evaluate()
    print(f"   Validation Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

    # 8. Save LoRA Adapter & Tokenizer
    print(f"\n7. Saving Final LoRA Adapter to: {adapter_dir}...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("   LoRA adapter and tokenizer successfully saved!")

    # 9. Record Training Summary Log
    peak_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "device": device,
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules
        },
        "training_config": {
            "num_epochs": args.num_epochs,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed
        },
        "parameters": {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": round(trainable_pct, 4)
        },
        "metrics": {
            "training_duration_seconds": round(training_duration, 2),
            "final_train_loss": round(train_result.training_loss, 4),
            "final_eval_loss": round(eval_metrics.get("eval_loss", 0.0), 4),
            "peak_process_ram_mb": round(peak_ram_mb, 2),
            "peak_gpu_vram_mb": round(gpu_mem, 2) if gpu_mem else None
        }
    }

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary recorded to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

