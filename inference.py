"""
SupportIQ - Standalone Inference Pipeline
Runs inference with the fine-tuned SupportIQ LoRA model to classify
customer support messages into structured JSON.

Author: SupportIQ ML Engineering Team
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a customer support triage AI. Classify the customer support message into JSON "
    "with category, priority, and sentiment."
)


def resolve_model_path(model_path: str) -> str:
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-0.5B-Instruct"))
    if model_path == "Qwen/Qwen2.5-0.5B-Instruct" and os.path.exists(local_dir):
        return local_dir
    return model_path


def resolve_adapter_path(adapter_path: str) -> str:
    if os.path.exists(adapter_path):
        return adapter_path
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "lora_adapter"))
    if os.path.exists(local_dir):
        return local_dir
    alt_local = os.path.join("outputs", "lora_adapter")
    if os.path.exists(alt_local):
        return alt_local
    return adapter_path


def load_model_and_tokenizer(
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str = "outputs/lora_adapter"
):
    base_model_id = resolve_model_path(base_model_id)
    adapter_path = resolve_adapter_path(adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SupportIQ model on {device.upper()}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)

    if os.path.exists(adapter_path):
        print(f"Applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    else:
        print(f"Warning: Adapter path '{adapter_path}' not found. Using raw base model.")
        model = base_model

    model.eval()
    return model, tokenizer, device


def classify_message(model: Any, tokenizer: Any, device: str, message: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    latency = time.time() - start_time

    generated_ids = output_tokens[0][input_ids.shape[1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    try:
        parsed_json = json.loads(raw_output)
    except json.JSONDecodeError:
        # Fallback regex extraction if needed
        import re
        match = re.search(r"\{[^{}]*\}", raw_output)
        if match:
            try:
                parsed_json = json.loads(match.group(0))
            except Exception:
                parsed_json = {"error": "JSON parse error", "raw": raw_output}
        else:
            parsed_json = {"error": "JSON parse error", "raw": raw_output}

    return {
        "classification": parsed_json,
        "raw_output": raw_output,
        "latency_ms": round(latency * 1000, 2),
        "tokens_generated": len(generated_ids)
    }


def main():
    parser = argparse.ArgumentParser(description="SupportIQ Customer Support Message Classifier")
    parser.add_argument("--message", type=str, default=None, help="Input customer support message")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="SupportIQ/outputs/lora_adapter")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI session")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.adapter_dir)

    if args.interactive:
        print("\n=== SupportIQ Interactive Classification CLI ===")
        print("Type your customer message below (or 'exit' to quit):\n")
        while True:
            try:
                msg = input("Customer Message > ").strip()
                if not msg:
                    continue
                if msg.lower() in ["exit", "quit", "q"]:
                    break
                res = classify_message(model, tokenizer, device, msg)
                print("\nClassification Result:")
                print(json.dumps(res["classification"], indent=2))
                print(f"(Latency: {res['latency_ms']} ms | Tokens: {res['tokens_generated']})\n")
            except (KeyboardInterrupt, EOFError):
                break
    elif args.message:
        res = classify_message(model, tokenizer, device, args.message)
        print(json.dumps(res, indent=2))
    else:
        sample_msg = "I was double charged $149 for invoice #83920. Reverse this immediately!"
        print(f"\nNo message provided. Testing with sample message:\n\"{sample_msg}\"\n")
        res = classify_message(model, tokenizer, device, sample_msg)
        print("Result:")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

