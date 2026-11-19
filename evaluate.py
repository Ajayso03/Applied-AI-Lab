"""
SupportIQ - Evaluation Pipeline
Rigorously evaluates Base Model vs Fine-Tuned LoRA Model on the exact same
held-out test set (>= 50 examples).

Computes:
1. Exact JSON Match Accuracy
2. JSON Parsing Success Rate
3. Category Accuracy
4. Priority Accuracy
5. Sentiment Accuracy
6. Mean Field-Level Accuracy

Produces:
- results/evaluation.json
- results/predictions.csv
- results/comparison.png

Author: SupportIQ ML Engineering Team
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a customer support triage AI. Classify the customer support message into JSON "
    "with category, priority, and sentiment."
)


def extract_json_payload(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempts to parse text directly as JSON, or extracts first valid JSON object block.
    Returns (parsed_dict, is_valid_json).
    """
    text_clean = text.strip()
    
    # Direct attempt
    try:
        data = json.loads(text_clean)
        if isinstance(data, dict):
            return data, True
    except Exception:
        pass

    # Regex extraction of outer {...}
    match = re.search(r"\{[^{}]*\}", text_clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data, True
        except Exception:
            pass

    return None, False


def run_model_inference(
    model: Any,
    tokenizer: Any,
    test_records: List[Dict[str, Any]],
    device: str,
    desc: str = "Evaluating"
) -> List[Dict[str, Any]]:
    """
    Runs greedy generation on all test records and returns predictions and latency.
    """
    results = []
    print(f"\n--- {desc} ({len(test_records)} samples) ---")
    
    model.eval()
    for idx, record in enumerate(test_records, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["input"]}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        t0 = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        
        # Decode only the generated response
        generated_ids = output_tokens[0][input_ids.shape[1]:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        parsed_dict, is_valid = extract_json_payload(raw_output)
        
        results.append({
            "raw_output": raw_output,
            "parsed": parsed_dict,
            "is_valid_json": is_valid,
            "latency_sec": elapsed
        })
        
        if idx % 25 == 0 or idx == len(test_records):
            print(f"Processed {idx:3d}/{len(test_records):3d} records...")
            
    return results


def compute_metrics(
    test_records: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes exact match, field accuracies, and parsing rates.
    """
    total = len(test_records)
    valid_json_count = 0
    exact_match_count = 0
    cat_match_count = 0
    prio_match_count = 0
    sent_match_count = 0
    field_accuracies = []

    for rec, pred in zip(test_records, predictions):
        target = json.loads(rec["output"])
        is_valid = pred["is_valid_json"]
        parsed = pred["parsed"]

        if is_valid:
            valid_json_count += 1

        cat_match = (parsed.get("category") == target.get("category")) if parsed else False
        prio_match = (parsed.get("priority") == target.get("priority")) if parsed else False
        sent_match = (parsed.get("sentiment") == target.get("sentiment")) if parsed else False

        if cat_match:
            cat_match_count += 1
        if prio_match:
            prio_match_count += 1
        if sent_match:
            sent_match_count += 1

        exact_match = cat_match and prio_match and sent_match
        if exact_match:
            exact_match_count += 1

        field_acc = (int(cat_match) + int(prio_match) + int(sent_match)) / 3.0
        field_accuracies.append(field_acc)

    metrics = {
        "total_test_samples": total,
        "json_parsing_success_rate": round(100.0 * valid_json_count / total, 2),
        "exact_json_match_accuracy": round(100.0 * exact_match_count / total, 2),
        "category_accuracy": round(100.0 * cat_match_count / total, 2),
        "priority_accuracy": round(100.0 * prio_match_count / total, 2),
        "sentiment_accuracy": round(100.0 * sent_match_count / total, 2),
        "mean_field_accuracy": round(100.0 * sum(field_accuracies) / total, 2)
    }
    return metrics


def generate_comparison_plot(
    base_metrics: Dict[str, float],
    ft_metrics: Dict[str, float],
    output_png: str
):
    """
    Generates a publication-grade bar chart comparing Base vs Fine-Tuned models.
    """
    labels = [
        "JSON Parsing\nRate",
        "Exact JSON\nMatch",
        "Category\nAccuracy",
        "Priority\nAccuracy",
        "Sentiment\nAccuracy",
        "Mean Field\nAccuracy"
    ]
    keys = [
        "json_parsing_success_rate",
        "exact_json_match_accuracy",
        "category_accuracy",
        "priority_accuracy",
        "sentiment_accuracy",
        "mean_field_accuracy"
    ]

    base_vals = [base_metrics[k] for k in keys]
    ft_vals = [ft_metrics[k] for k in keys]

    x = range(len(labels))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 6))

    rects1 = ax.bar([i - width/2 for i in x], base_vals, width, label="Base Model (Qwen2.5-0.5B-Instruct)", color="#7293CB", edgecolor="#4A6A99")
    rects2 = ax.bar([i + width/2 for i in x], ft_vals, width, label="Fine-Tuned (SupportIQ LoRA)", color="#E17C05", edgecolor="#A85700")

    ax.set_ylabel("Accuracy / Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("SupportIQ: Base Model vs. LoRA Fine-Tuned Model Performance", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10, fontweight="semibold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc="upper left")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Comparison plot successfully saved to: {output_png}")


def resolve_model_path(model_path: str) -> str:
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct"))
    if model_path == "Qwen/Qwen2.5-0.5B-Instruct" and os.path.exists(local_dir):
        return local_dir
    return model_path


def resolve_path(path: str, fallback_relative: str) -> str:
    if os.path.exists(path):
        return path
    if path.startswith("SupportIQ/") or path.startswith("SupportIQ\\"):
        trimmed = path[len("SupportIQ/"):].lstrip("/\\")
        if os.path.exists(trimmed):
            return trimmed
    local_target = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", fallback_relative))
    if os.path.exists(local_target):
        return local_target
    return path


def main():
    parser = argparse.ArgumentParser(description="SupportIQ Model Evaluation")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="outputs/lora_adapter")
    parser.add_argument("--test_data_path", type=str, default="data/test.jsonl")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--data_path", type=str, default=None, help="Alias for --test_data_path")
    parser.add_argument("--output_dir", type=str, default=None, help="Alias for --results_dir")
    args = parser.parse_args()

    if args.data_path:
        args.test_data_path = args.data_path
    if args.output_dir:
        args.results_dir = args.output_dir

    args.base_model_path = resolve_model_path(args.base_model_path)
    args.adapter_dir = resolve_path(args.adapter_dir, os.path.join("outputs", "lora_adapter"))
    args.test_data_path = resolve_path(args.test_data_path, os.path.join("data", "test.jsonl"))
    args.results_dir = resolve_path(args.results_dir, "results")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluation running on device: {device.upper()}")

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Load Test Dataset
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(test_records)} held-out test samples from {args.test_data_path}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Evaluate Base Model
    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"\nLoading Base Model ({args.base_model_path})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)

    base_preds = run_model_inference(base_model, tokenizer, test_records, device, desc="Evaluating Base Model")
    base_metrics = compute_metrics(test_records, base_preds)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Evaluate Fine-Tuned LoRA Model
    print(f"\nLoading Fine-Tuned Model (Base + LoRA Adapter: {args.adapter_dir})...")
    ft_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    ft_model = PeftModel.from_pretrained(ft_base, args.adapter_dir).to(device)

    ft_preds = run_model_inference(ft_model, tokenizer, test_records, device, desc="Evaluating LoRA Fine-Tuned Model")
    ft_metrics = compute_metrics(test_records, ft_preds)
    del ft_model
    del ft_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Print Comparison Table
    print("\n" + "=" * 70)
    print("SUPPORTIQ EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<32} | {'Base Model':<15} | {'Fine-Tuned (LoRA)':<15}")
    print("-" * 70)
    for k in ["json_parsing_success_rate", "exact_json_match_accuracy", "category_accuracy", "priority_accuracy", "sentiment_accuracy", "mean_field_accuracy"]:
        name = k.replace("_", " ").title()
        print(f"{name:<32} | {base_metrics[k]:>13.2f}% | {ft_metrics[k]:>15.2f}%")
    print("=" * 70)

    # 6. Save results to JSON
    eval_json_path = os.path.join(args.results_dir, "evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": base_metrics,
            "finetuned_model": ft_metrics
        }, f, indent=2)
    print(f"Evaluation metrics saved to: {eval_json_path}")

    # 7. Save predictions to CSV
    csv_path = os.path.join(args.results_dir, "predictions.csv")
    csv_rows = []
    for idx, (rec, b_p, ft_p) in enumerate(zip(test_records, base_preds, ft_preds)):
        target = json.loads(rec["output"])
        b_dict = b_p["parsed"] or {}
        ft_dict = ft_p["parsed"] or {}

        csv_rows.append({
            "id": rec["id"],
            "input": rec["input"],
            "expected_category": target.get("category"),
            "expected_priority": target.get("priority"),
            "expected_sentiment": target.get("sentiment"),
            "base_raw_output": b_p["raw_output"],
            "base_category": b_dict.get("category"),
            "base_priority": b_dict.get("priority"),
            "base_sentiment": b_dict.get("sentiment"),
            "base_exact_match": (b_dict == target),
            "finetuned_raw_output": ft_p["raw_output"],
            "finetuned_category": ft_dict.get("category"),
            "finetuned_priority": ft_dict.get("priority"),
            "finetuned_sentiment": ft_dict.get("sentiment"),
            "finetuned_exact_match": (ft_dict == target)
        })

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Predictions CSV saved to: {csv_path}")

    # 8. Generate Visual Comparison Chart
    png_path = os.path.join(args.results_dir, "comparison.png")
    generate_comparison_plot(base_metrics, ft_metrics, png_path)


if __name__ == "__main__":
    main()

