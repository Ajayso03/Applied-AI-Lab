"""
End-to-End Benchmarking Orchestrator.
Executes systematic, reproducible evaluations of Q4_K_M vs Q8_0 GGUF configurations
across prompt processing speed, generation throughput, peak memory usage, and output quality.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from inference import LlamaInferenceEngine
from model_manager import ModelManager, MODEL_CONFIGS
from quality_evaluation import QualityEvaluator


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def generate_benchmark_charts(
    summary_stats: Dict[str, Any],
    comparisons: List[Dict[str, Any]],
    output_dir: str = RESULTS_DIR,
):
    """Generates publication-quality charts for TPS, memory, and output quality."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    quants = list(summary_stats.keys())
    q4_key = "Q4_K_M" if "Q4_K_M" in summary_stats else quants[0]
    q8_key = "Q8_0" if "Q8_0" in summary_stats else (quants[1] if len(quants) > 1 else quants[0])

    # 1. Performance (Tokens per Second) Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(quants))
    width = 0.35

    gen_speeds = [summary_stats[q]["generation_tps_mean"] for q in quants]
    gen_stds = [summary_stats[q]["generation_tps_std"] for q in quants]
    prompt_speeds = [summary_stats[q]["prompt_eval_tps_mean"] for q in quants]

    rects1 = ax.bar(x - width/2, gen_speeds, width, yerr=gen_stds, label='Generation Speed (tokens/s)', color='#2b5c8f', capsize=5)
    rects2 = ax.bar(x + width/2, prompt_speeds, width, label='Prompt Processing (tokens/s)', color='#46a575', alpha=0.85)

    ax.set_ylabel('Throughput (Tokens / Second)', fontsize=11, fontweight='bold')
    ax.set_title('Inference Throughput: Q4_K_M vs Q8_0 (AMD Ryzen 5 5600H CPU)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quants, fontsize=11, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data values on top
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f} t/s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f} t/s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[Benchmark] Saved performance chart: {chart1_path}")

    # 2. Memory Consumption Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(quants))
    width = 0.35

    disk_sizes = [summary_stats[q]["model_file_size_gb"] for q in quants]
    peak_rams = [summary_stats[q]["peak_ram_gb_max"] for q in quants]

    r1 = ax.bar(x - width/2, disk_sizes, width, label='Model Weight Storage (GB)', color='#d97724')
    r2 = ax.bar(x + width/2, peak_rams, width, label='Peak Process RAM (GB)', color='#803b82')

    ax.set_ylabel('Memory Footprint (GB)', fontsize=11, fontweight='bold')
    ax.set_title('Memory Consumption: Disk Storage vs. Active Peak RAM', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(quants, fontsize=11, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f} GB', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
    for rect in r2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f} GB', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "memory_comparison.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[Benchmark] Saved memory chart: {chart2_path}")

    # 3. Quality Comparison Chart across 10 Prompts
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = [c["category"].replace(" ", "\n") for c in comparisons]
    x = np.arange(len(categories))
    width = 0.35

    q4_scores = [c["q4_score"] for c in comparisons]
    q8_scores = [c["q8_score"] for c in comparisons]

    b1 = ax.bar(x - width/2, q4_scores, width, label='Q4_K_M Score (max 10)', color='#337ab7')
    b2 = ax.bar(x + width/2, q8_scores, width, label='Q8_0 Score (max 10)', color='#e67e22')

    ax.set_ylabel('Evaluation Rubric Score (0-10)', fontsize=11, fontweight='bold')
    ax.set_title('Evaluation Quality Across 10 Capabilities: Q4_K_M vs Q8_0', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 11)
    ax.axhline(10, color='gray', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "quality_comparison.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[Benchmark] Saved quality chart: {chart3_path}")


class BenchmarkOrchestrator:
    """Orchestrates comprehensive multi-prompt benchmarks across quantisation tiers."""

    def __init__(
        self,
        backend: str = "cpu",
        prompts_file: str = os.path.join(DATA_DIR, "benchmark_prompts.json"),
        results_dir: str = RESULTS_DIR,
        n_ctx: int = 2048,
        n_threads: int = 6,
    ):
        self.backend = backend
        self.prompts_file = prompts_file
        self.results_dir = results_dir
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model_manager = ModelManager()

        with open(prompts_file, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

    def run_benchmark(self, quants: List[str] = ["Q4_K_M", "Q8_0"]) -> Dict[str, Any]:
        """Runs benchmarks on each quantisation configuration."""
        raw_results = {}
        summary_stats = {}
        collected_outputs = {q: {} for q in quants}

        for q_type in quants:
            print("\n" + "=" * 65)
            print(f"STARTING BENCHMARK: {q_type} on {self.backend.upper()}")
            print("=" * 65)

            engine = LlamaInferenceEngine(
                quant_type=q_type,
                backend=self.backend,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                model_manager=self.model_manager,
            )

            # Warmup run to prime OS page caches and model memory mappings
            print(f"[{q_type}] Executing warm-up run...")
            warmup_res = engine.generate("Warm-up: State Newton's second law briefly.", max_tokens=32)
            print(f"[{q_type}] Warmup completed: success={warmup_res['success']}, load_time={warmup_res['timings'].get('load_time_ms')} ms")

            prompt_results = []
            gen_tps_list = []
            prompt_tps_list = []
            latency_list = []
            peak_ram_list = []

            for idx, p in enumerate(self.prompts, 1):
                p_id = p["id"]
                p_cat = p["category"]
                p_text = p["prompt"]

                print(f"[{q_type}] Running [{idx}/10] {p_id} ({p_cat})...")
                res = engine.generate(p_text, max_tokens=256, temperature=0.0, seed=42)

                if res["success"]:
                    collected_outputs[q_type][p_id] = res["output_text"]
                    timings = res["timings"]
                    gen_tps = timings.get("generation_tps") or 0.0
                    prompt_tps = timings.get("prompt_eval_tps") or 0.0
                    total_time = res["total_elapsed_sec"]
                    peak_ram = res.get("peak_ram_gb") or 0.0

                    if gen_tps > 0: gen_tps_list.append(gen_tps)
                    if prompt_tps > 0: prompt_tps_list.append(prompt_tps)
                    latency_list.append(total_time)
                    if peak_ram > 0: peak_ram_list.append(peak_ram)

                    prompt_results.append({
                        "prompt_id": p_id,
                        "category": p_cat,
                        "success": True,
                        "prompt_tokens": timings.get("prompt_tokens"),
                        "generated_tokens": timings.get("eval_tokens"),
                        "generation_tps": gen_tps,
                        "prompt_eval_tps": prompt_tps,
                        "total_elapsed_sec": total_time,
                        "load_time_ms": timings.get("load_time_ms"),
                        "eval_time_ms": timings.get("eval_time_ms"),
                        "peak_ram_mb": res.get("peak_ram_mb"),
                        "peak_ram_gb": peak_ram,
                        "output_text": res["output_text"],
                    })
                    print(f"       -> Generated {timings.get('eval_tokens')} tokens @ {gen_tps:.2f} t/s | Latency: {total_time:.2f}s | Peak RAM: {peak_ram:.2f} GB")
                else:
                    print(f"       -> FAILED: {res.get('error')}")
                    prompt_results.append({
                        "prompt_id": p_id,
                        "category": p_cat,
                        "success": False,
                        "error": res.get("error"),
                    })

            raw_results[q_type] = prompt_results

            # Compute summary stats
            file_size_gb = round(os.path.getsize(engine.model_path) / (1024**3), 3)
            summary_stats[q_type] = {
                "quant_type": q_type,
                "backend": engine.backend_description,
                "model_file_size_gb": file_size_gb,
                "total_prompts": len(self.prompts),
                "successful_prompts": len([r for r in prompt_results if r["success"]]),
                "failed_prompts": len([r for r in prompt_results if not r["success"]]),
                "generation_tps_mean": round(statistics.mean(gen_tps_list), 2) if gen_tps_list else 0.0,
                "generation_tps_std": round(statistics.stdev(gen_tps_list), 2) if len(gen_tps_list) > 1 else 0.0,
                "prompt_eval_tps_mean": round(statistics.mean(prompt_tps_list), 2) if prompt_tps_list else 0.0,
                "prompt_eval_tps_std": round(statistics.stdev(prompt_tps_list), 2) if len(prompt_tps_list) > 1 else 0.0,
                "latency_sec_mean": round(statistics.mean(latency_list), 2) if latency_list else 0.0,
                "latency_sec_std": round(statistics.stdev(latency_list), 2) if len(latency_list) > 1 else 0.0,
                "peak_ram_gb_max": round(max(peak_ram_list), 3) if peak_ram_list else 0.0,
                "peak_ram_gb_mean": round(statistics.mean(peak_ram_list), 3) if peak_ram_list else 0.0,
            }

        # Quality evaluation
        print("\n" + "=" * 65)
        print("RUNNING QUALITY EVALUATION & SCORING")
        print("=" * 65)
        evaluator = QualityEvaluator(self.prompts_file)
        csv_quality = os.path.join(self.results_dir, "quality_comparison.csv")
        json_quality = os.path.join(self.results_dir, "quality_comparison.json")
        comparisons = evaluator.compare_prompts(
            q4_outputs=collected_outputs.get("Q4_K_M", {}),
            q8_outputs=collected_outputs.get("Q8_0", {}),
            output_csv_path=csv_quality,
            output_json_path=json_quality,
        )

        q4_mean_score = round(statistics.mean([c["q4_score"] for c in comparisons]), 2)
        q8_mean_score = round(statistics.mean([c["q8_score"] for c in comparisons]), 2)

        if "Q4_K_M" in summary_stats: summary_stats["Q4_K_M"]["average_quality_score"] = q4_mean_score
        if "Q8_0" in summary_stats: summary_stats["Q8_0"]["average_quality_score"] = q8_mean_score

        # Save machine-readable results
        final_benchmark_data = {
            "summary": summary_stats,
            "raw_prompt_runs": raw_results,
            "quality_evaluation": comparisons,
        }

        json_out = os.path.join(self.results_dir, "benchmark_results.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(final_benchmark_data, f, indent=2)
        print(f"[Benchmark] Saved JSON results: {json_out}")

        # Save summary CSV
        csv_out = os.path.join(self.results_dir, "benchmark_results.csv")
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "quant_type", "backend", "model_file_size_gb", "total_prompts",
                "successful_prompts", "failed_prompts", "generation_tps_mean", "generation_tps_std",
                "prompt_eval_tps_mean", "prompt_eval_tps_std", "latency_sec_mean", "latency_sec_std",
                "peak_ram_gb_max", "peak_ram_gb_mean", "average_quality_score"
            ])
            writer.writeheader()
            for q_stat in summary_stats.values():
                writer.writerow(q_stat)
        print(f"[Benchmark] Saved CSV results: {csv_out}")

        # Generate Visual Charts
        print("\n[Benchmark] Generating visual comparison charts...")
        generate_benchmark_charts(summary_stats, comparisons, self.results_dir)

        return final_benchmark_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute full LocalLLM benchmark suite.")
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q8_0"])
    args = parser.parse_args()

    orchestrator = BenchmarkOrchestrator(backend=args.backend, n_threads=args.threads)
    results = orchestrator.run_benchmark(quants=args.quants)
    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 65)
    for q, s in results["summary"].items():
        print(f"{q}: {s['generation_tps_mean']} t/s | File: {s['model_file_size_gb']} GB | Peak RAM: {s['peak_ram_gb_max']} GB | Quality: {s.get('average_quality_score')}/10")
