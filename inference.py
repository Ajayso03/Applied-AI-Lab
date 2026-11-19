"""
Inference Interface for llama.cpp.
Provides a unified, robust interface supporting CPU and Vulkan acceleration,
deterministic generation, configurable context/generation bounds,
timing/telemetry extraction, and real-time process memory tracking.
"""

import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from model_manager import ModelManager, MODEL_CONFIGS
from memory_monitor import monitor_subprocess_memory


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")

DEFAULT_CPU_CLI = os.path.join(TOOLS_DIR, "llama-cpu", "llama-cli.exe")
DEFAULT_VULKAN_CLI = os.path.join(TOOLS_DIR, "llama-vulkan", "llama-cli.exe")


def format_qwen_chat_prompt(user_prompt: str, system_prompt: str = "You are a helpful, precise, and concise AI assistant.") -> str:
    """Formats prompt according to Qwen2.5 ChatML template."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_llama_timings(output_text: str) -> Dict[str, Any]:
    """
    Parses GGML/llama.cpp performance metrics from console output or JSON logs.
    """
    timings: Dict[str, Any] = {
        "load_time_ms": None,
        "prompt_eval_time_ms": None,
        "prompt_tokens": None,
        "prompt_eval_tps": None,
        "eval_time_ms": None,
        "eval_tokens": None,
        "generation_tps": None,
        "total_time_ms": None,
    }

    # Bracketed summary: [ Prompt: 74.4 t/s | Generation: 26.6 t/s ]
    m_bracket = re.search(r"\[\s*Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s\s*\]", output_text)
    if m_bracket:
        timings["prompt_eval_tps"] = float(m_bracket.group(1))
        timings["generation_tps"] = float(m_bracket.group(2))

    # JSON streaming timings
    m_prompt_json = re.search(r'"prompt_n":(\d+),"prompt_ms":([\d\.]+).*?"prompt_per_second":([\d\.]+)', output_text)
    if m_prompt_json:
        timings["prompt_tokens"] = int(m_prompt_json.group(1))
        timings["prompt_eval_time_ms"] = float(m_prompt_json.group(2))
        if not timings["prompt_eval_tps"]:
            timings["prompt_eval_tps"] = round(float(m_prompt_json.group(3)), 2)

    m_eval_json = re.search(r'"predicted_n":(\d+),"predicted_ms":([\d\.]+).*?"predicted_per_second":([\d\.]+)', output_text)
    if m_eval_json:
        timings["eval_tokens"] = int(m_eval_json.group(1))
        timings["eval_time_ms"] = float(m_eval_json.group(2))
        if not timings["generation_tps"]:
            timings["generation_tps"] = round(float(m_eval_json.group(3)), 2)

    # Standard perf context prints
    m_load = re.search(r"load time\s*=\s*([\d\.]+)\s*ms", output_text)
    if m_load:
        timings["load_time_ms"] = float(m_load.group(1))

    m_peval = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d\.]+)\s*tokens per second\)", output_text)
    if m_peval:
        timings["prompt_eval_time_ms"] = float(m_peval.group(1))
        timings["prompt_tokens"] = int(m_peval.group(2))
        timings["prompt_eval_tps"] = float(m_peval.group(3))

    m_eval = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)\s*\(\s*([\d\.]+)\s*tokens per second\)", output_text)
    if m_eval:
        timings["eval_time_ms"] = float(m_eval.group(1))
        timings["eval_tokens"] = int(m_eval.group(2))
        timings["generation_tps"] = float(m_eval.group(3))

    m_tot = re.search(r"total time\s*=\s*([\d\.]+)\s*ms", output_text)
    if m_tot:
        timings["total_time_ms"] = float(m_tot.group(1))

    return timings


class LlamaInferenceEngine:
    """
    High-level abstraction for executing deterministic or stochastic inference
    against quantised GGUF models using official llama.cpp binaries.
    """

    def __init__(
        self,
        quant_type: str = "Q4_K_M",
        backend: str = "cpu",
        n_ctx: int = 2048,
        n_threads: int = 6,
        model_manager: Optional[ModelManager] = None,
    ):
        self.quant_type = quant_type
        self.backend = backend.lower()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model_manager = model_manager or ModelManager()

        self.model_path = self.model_manager.get_model_path(self.quant_type)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found for {self.quant_type} at: {self.model_path}")

        # Select binary
        if self.backend == "vulkan":
            self.binary_path = DEFAULT_VULKAN_CLI
            if not os.path.exists(self.binary_path):
                raise FileNotFoundError(f"Vulkan llama-cli binary missing: {self.binary_path}")
            self.backend_description = "Vulkan GPU Acceleration (AMD Radeon RX 6500M / Vulkan 1.3)"
            self.n_gpu_layers = 99
        else:
            self.binary_path = DEFAULT_CPU_CLI
            if not os.path.exists(self.binary_path):
                raise FileNotFoundError(f"CPU llama-cli binary missing: {self.binary_path}")
            self.backend_description = f"CPU Vectorized (AVX2 / FMA, {self.n_threads} OpenMP threads)"
            self.n_gpu_layers = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 42,
        raw_prompt: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes single-turn inference, capturing stdout, stderr, timings, and RAM.
        """
        formatted_prompt = prompt if raw_prompt else format_qwen_chat_prompt(prompt)

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", formatted_prompt,
            "-n", str(max_tokens),
            "-c", str(self.n_ctx),
            "--threads", str(self.n_threads),
            "--temp", str(temperature),
            "-s", str(seed),
            "--single-turn",
            "--perf",
            "--simple-io",
            "--no-warmup",
        ]

        if self.backend == "vulkan":
            cmd.extend(["-ngl", str(self.n_gpu_layers)])
        else:
            cmd.extend(["-ngl", "0"])

        start_time = time.perf_counter()

        try:
            # Launch process with pipes
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # Monitor memory while running
            mem_stats = monitor_subprocess_memory(proc)

            stdout, stderr = proc.communicate()
            total_elapsed_sec = time.perf_counter() - start_time

            if proc.returncode != 0:
                return {
                    "success": False,
                    "error": f"Process exited with code {proc.returncode}: {stderr[-1000:]}",
                    "quant_type": self.quant_type,
                    "backend": self.backend_description,
                }

            combined_logs = stdout + "\n" + stderr
            timings = parse_llama_timings(combined_logs)

            # Clean generated text: extract text after user prompt prompt marker and before [ Prompt:
            output_text = stdout
            if "\n> " in output_text:
                after_prompt = output_text.split("\n> ", 1)[-1]
                lines = after_prompt.splitlines()
                output_text = "\n".join(lines[1:]) if len(lines) > 1 else after_prompt
            elif "> " in output_text:
                after_prompt = output_text.split("> ", 1)[-1]
                lines = after_prompt.splitlines()
                output_text = "\n".join(lines[1:]) if len(lines) > 1 else after_prompt

            if "[ Prompt:" in output_text:
                output_text = output_text.split("[ Prompt:", 1)[0]
            if "Exiting..." in output_text:
                output_text = output_text.split("Exiting...", 1)[0]
            output_text = output_text.strip()

            # Calculate tokens if not explicitly in JSON
            if timings["generation_tps"] and not timings["eval_tokens"]:
                approx_tokens = max(1, int(len(output_text.split()) * 1.3))
                timings["eval_tokens"] = approx_tokens
                timings["eval_time_ms"] = round((approx_tokens / timings["generation_tps"]) * 1000, 2)

            return {
                "success": True,
                "quant_type": self.quant_type,
                "backend": self.backend_description,
                "prompt": prompt,
                "output_text": output_text,
                "model_path": self.model_path,
                "model_file_size_gb": round(os.path.getsize(self.model_path) / (1024**3), 3),
                "peak_ram_mb": mem_stats.get("peak_rss_mb"),
                "peak_ram_gb": mem_stats.get("peak_rss_gb"),
                "total_elapsed_sec": round(total_elapsed_sec, 3),
                "timings": timings,
                "raw_stderr": stderr[-2000:] if stderr else "",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "quant_type": self.quant_type,
                "backend": self.backend_description,
                "total_elapsed_sec": round(time.perf_counter() - start_time, 3),
            }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run llama.cpp inference.")
    parser.add_argument("--quant", choices=["Q4_K_M", "Q8_0"], default="Q4_K_M")
    parser.add_argument("--backend", choices=["cpu", "vulkan"], default="cpu")
    parser.add_argument("--prompt", default="Why does quantisation save memory in neural networks?")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    engine = LlamaInferenceEngine(quant_type=args.quant, backend=args.backend)
    res = engine.generate(args.prompt, max_tokens=args.max_tokens)
    print("Inference Success:", res["success"])
    if res["success"]:
        print("Backend:", res["backend"])
        print("Output:\n", res["output_text"])
        print("Timings:", res["timings"])
        print(f"Peak RAM: {res['peak_ram_mb']} MB")
    else:
        print("Error:", res.get("error"))
