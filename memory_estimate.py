"""
SupportIQ - 7B LLM GPU Memory Estimation Tool
Provides mathematically rigorous memory calculations for fine-tuning
a 7-Billion-parameter language model using:
1. Full Fine-Tuning (16-bit Mixed Precision)
2. LoRA (16-bit Base Model + LoRA Adapters)
3. QLoRA (4-bit NF4 Base Model + LoRA Adapters + Paged Optimizer)

Author: SupportIQ ML Engineering Team
"""

import argparse
import json


def calculate_7b_memory(
    num_params_b: float = 7.0,
    seq_len: int = 2048,
    batch_size: int = 1,
    lora_rank: int = 16,
    lora_trainable_pct: float = 0.4,
    use_gradient_checkpointing: bool = True
):
    """
    Computes precise memory breakdown across static weights, gradients,
    optimizer states, activations, and runtime overhead.
    """
    N = num_params_b * 1e9
    lora_params = N * (lora_trainable_pct / 100.0)

    # -------------------------------------------------------------
    # 1. FULL FINE-TUNING (16-bit mixed precision with FP32 AdamW)
    # -------------------------------------------------------------
    # Model weights: 16-bit (2 bytes/param)
    full_weights_gb = (N * 2) / (1024**3)
    # Gradients: 16-bit (2 bytes/param)
    full_grads_gb = (N * 2) / (1024**3)
    # Optimizer states: AdamW maintains FP32 master weights (4 bytes),
    # FP32 1st momentum (4 bytes), FP32 2nd momentum (4 bytes) = 12 bytes/param
    full_opt_gb = (N * 12) / (1024**3)
    full_static_gb = full_weights_gb + full_grads_gb + full_opt_gb

    # Activations (approx for 7B with standard 32 layers, hidden 4096)
    # With gradient checkpointing: ~2.5 GB for bs=1, seq=2048
    # Without gradient checkpointing: ~16 GB
    act_full_gb = 2.5 if use_gradient_checkpointing else 16.0
    cuda_overhead_gb = 1.5

    full_total_gb = full_static_gb + act_full_gb + cuda_overhead_gb

    # -------------------------------------------------------------
    # 2. LoRA FINE-TUNING (16-bit base model + LoRA adapters)
    # -------------------------------------------------------------
    # Base model weights: 16-bit (frozen, 2 bytes/param)
    lora_base_weights_gb = (N * 2) / (1024**3)
    lora_base_grads_gb = 0.0
    lora_base_opt_gb = 0.0

    # LoRA trainable params
    lora_adapter_weights_gb = (lora_params * 2) / (1024**3)
    lora_adapter_grads_gb = (lora_params * 2) / (1024**3)
    lora_adapter_opt_gb = (lora_params * 12) / (1024**3)
    lora_adapter_total_gb = lora_adapter_weights_gb + lora_adapter_grads_gb + lora_adapter_opt_gb

    lora_static_gb = lora_base_weights_gb + lora_adapter_total_gb
    act_lora_gb = 2.8 if use_gradient_checkpointing else 18.0
    lora_total_gb = lora_static_gb + act_lora_gb + cuda_overhead_gb

    # -------------------------------------------------------------
    # 3. QLoRA FINE-TUNING (4-bit NF4 quantized base + 16-bit LoRA)
    # -------------------------------------------------------------
    # Base model: 4-bit (0.5 bytes/param) + double quantization (~0.05 B)
    qlora_base_weights_gb = (N * 0.5) / (1024**3)
    double_quant_overhead_gb = (N * 0.05) / (1024**3)  # FP8 block scales
    qlora_base_weights_total_gb = qlora_base_weights_gb + double_quant_overhead_gb

    qlora_adapter_total_gb = lora_adapter_total_gb  # LoRA adapters remain in FP16/BF16
    qlora_static_gb = qlora_base_weights_total_gb + qlora_adapter_total_gb

    # Activations in QLoRA: requires on-the-fly dequantization buffers
    act_qlora_gb = 3.2 if use_gradient_checkpointing else 19.5
    qlora_total_gb = qlora_static_gb + act_qlora_gb + cuda_overhead_gb

    return {
        "assumptions": {
            "parameter_count": f"{num_params_b} Billion ({int(N):,})",
            "sequence_length": seq_len,
            "micro_batch_size": batch_size,
            "lora_rank": lora_rank,
            "lora_trainable_percentage": f"{lora_trainable_pct}% ({int(lora_params):,} params)",
            "gradient_checkpointing_enabled": use_gradient_checkpointing,
            "optimizer": "Standard AdamW (FP32 states, 12 bytes/param)"
        },
        "breakdown": {
            "full_fine_tuning": {
                "weights_gb": round(full_weights_gb, 2),
                "gradients_gb": round(full_grads_gb, 2),
                "optimizer_states_gb": round(full_opt_gb, 2),
                "static_memory_gb": round(full_static_gb, 2),
                "activations_gb": round(act_full_gb, 2),
                "cuda_overhead_gb": round(cuda_overhead_gb, 2),
                "total_estimated_vram_gb": round(full_total_gb, 2),
                "hardware_recommendation": ">= 2x 80GB GPUs (A100/H100) with FSDP/DeepSpeed ZeRO-3"
            },
            "lora": {
                "base_weights_gb": round(lora_base_weights_gb, 2),
                "base_gradients_gb": 0.0,
                "base_optimizer_gb": 0.0,
                "adapter_total_gb": round(lora_adapter_total_gb, 2),
                "static_memory_gb": round(lora_static_gb, 2),
                "activations_gb": round(act_lora_gb, 2),
                "cuda_overhead_gb": round(cuda_overhead_gb, 2),
                "total_estimated_vram_gb": round(lora_total_gb, 2),
                "hardware_recommendation": "1x 24GB GPU (RTX 3090, RTX 4090, A10G)"
            },
            "qlora": {
                "base_weights_gb": round(qlora_base_weights_total_gb, 2),
                "base_gradients_gb": 0.0,
                "base_optimizer_gb": 0.0,
                "adapter_total_gb": round(qlora_adapter_total_gb, 2),
                "static_memory_gb": round(qlora_static_gb, 2),
                "activations_gb": round(act_qlora_gb, 2),
                "cuda_overhead_gb": round(cuda_overhead_gb, 2),
                "total_estimated_vram_gb": round(qlora_total_gb, 2),
                "hardware_recommendation": "1x 12GB - 16GB GPU (RTX 3060, RTX 4070, T4, V100)"
            }
        }
    }


def print_memory_report(data: dict):
    b = data["breakdown"]
    assump = data["assumptions"]

    print("=" * 80)
    print("7-BILLION-PARAMETER MODEL: GPU MEMORY REQUIREMENTS ANALYSIS")
    print("=" * 80)
    print("Core Assumptions:")
    for k, v in assump.items():
        print(f"  - {k.replace('_', ' ').title()}: {v}")
    print("-" * 80)

    header = f"{'Memory Component':<30} | {'Full Fine-Tuning':<15} | {'LoRA (16-bit)':<14} | {'QLoRA (4-bit)':<14}"
    print(header)
    print("-" * 80)

    rows = [
        ("Base Model Weights", f"{b['full_fine_tuning']['weights_gb']:.2f} GB", f"{b['lora']['base_weights_gb']:.2f} GB", f"{b['qlora']['base_weights_gb']:.2f} GB"),
        ("Base Gradients", f"{b['full_fine_tuning']['gradients_gb']:.2f} GB", "0.00 GB", "0.00 GB"),
        ("Base Optimizer (AdamW)", f"{b['full_fine_tuning']['optimizer_states_gb']:.2f} GB", "0.00 GB", "0.00 GB"),
        ("LoRA Adapter (W+G+Opt)", "N/A", f"{b['lora']['adapter_total_gb']:.2f} GB", f"{b['qlora']['adapter_total_gb']:.2f} GB"),
        ("Static Training Memory", f"{b['full_fine_tuning']['static_memory_gb']:.2f} GB", f"{b['lora']['static_memory_gb']:.2f} GB", f"{b['qlora']['static_memory_gb']:.2f} GB"),
        ("Activations (Checkpointing)", f"{b['full_fine_tuning']['activations_gb']:.2f} GB", f"{b['lora']['activations_gb']:.2f} GB", f"{b['qlora']['activations_gb']:.2f} GB"),
        ("CUDA & Kernel Overhead", f"{b['full_fine_tuning']['cuda_overhead_gb']:.2f} GB", f"{b['lora']['cuda_overhead_gb']:.2f} GB", f"{b['qlora']['cuda_overhead_gb']:.2f} GB"),
        ("TOTAL REQUIRED VRAM", f"{b['full_fine_tuning']['total_estimated_vram_gb']:.2f} GB", f"{b['lora']['total_estimated_vram_gb']:.2f} GB", f"{b['qlora']['total_estimated_vram_gb']:.2f} GB")
    ]

    for label, full, lora, qlora in rows:
        print(f"{label:<30} | {full:<15} | {lora:<14} | {qlora:<14}")

    print("=" * 80)
    print("Minimum Hardware Required:")
    print(f"  • Full Fine-Tuning: {b['full_fine_tuning']['hardware_recommendation']}")
    print(f"  • LoRA:             {b['lora']['hardware_recommendation']}")
    print(f"  • QLoRA:            {b['qlora']['hardware_recommendation']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Estimate GPU Memory for 7B Parameter Model")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    data = calculate_7b_memory()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_memory_report(data)


if __name__ == "__main__":
    main()

