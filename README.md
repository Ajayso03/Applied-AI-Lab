# LocalLLM Benchmark: Local Model Inference and Quantisation Optimisation
### Task 3 Technical Evaluation – tndrr AI Internship Assignment

[![Model: Qwen2.5-1.5B-Instruct](https://img.shields.io/badge/Model-Qwen2.5--1.5B--Instruct-blue.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
[![Runtime: llama.cpp](https://img.shields.io/badge/Runtime-llama.cpp%20b12532-orange.svg)](https://github.com/ggml-org/llama.cpp)
[![Format: GGUF v3](https://img.shields.io/badge/Format-GGUF%20v3-green.svg)](https://github.com/ggerganov/ggml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

---

## 1. Executive Summary

This project delivers a complete, reproducible technical implementation of **Task 3: Local Model Inference and Quantisation Optimisation**. It benchmarks and evaluates **`Qwen/Qwen2.5-1.5B-Instruct`** in two quantisation configurations:
- **Configuration A:** `Q4_K_M` (4-bit Medium K-quantisation, ~1.04 GB weights)
- **Configuration B:** `Q8_0` (8-bit standard integer quantisation, ~1.89 GB weights)

Both configurations are evaluated using **official `llama.cpp` Windows binaries (b12532)** with identical inference parameters across **10 original multi-disciplinary evaluation prompts**.

### Key Technical Findings:
1. **Inference Throughput:** `Q4_K_M` delivers **26.57 tokens/s**, representing a **+45.75% generation speedup** over `Q8_0` (18.23 tokens/s) on the AMD Ryzen 5 5600H CPU due to reduced memory-bus saturation in the bandwidth-bound decoding regime.
2. **Prompt Processing (Prefill):** `Q4_K_M` achieves **90.89 tokens/s** prompt prefill, operating **+98.41% faster (~2×)** than `Q8_0` (45.81 tokens/s).
3. **Storage & Memory Footprint:** `Q4_K_M` weighs **1.041 GB** vs **1.764 GB** on disk (**-40.99% storage savings**), with process peak RAM maintained at **1.65 GB**.
4. **Reasoning Quality:** On a 10-point multi-disciplinary rubric spanning logic, math, coding, JSON schema, and negative constraint following, `Q4_K_M` achieved **9.20 / 10** vs **8.95 / 10** for `Q8_0` (100% functional retention).
5. **Hardware Backend:** CPU vectorized inference (AVX2/FMA with 6 OpenMP threads) provides rock-solid deterministic performance. Vulkan acceleration is detected on the discrete AMD Radeon RX 6500M (4 GB VRAM) as an optional offload backend.

### Consolidated Performance Comparison Table

| Metric | Configuration A (`Q4_K_M`) | Configuration B (`Q8_0`) | Delta / Relative Change |
| :--- | :--- | :--- | :--- |
| **Model Weight File Size** | **1.041 GB** | **1.764 GB** | **-40.99% (-0.723 GB saved on disk)** |
| **Mean Generation Throughput** | **26.57 t/s** ($\sigma = 0.64$) | **18.23 t/s** ($\sigma = 0.61$) | **+45.75% higher generation speed (+8.34 t/s)** |
| **Mean Prompt Processing Speed** | **90.89 t/s** ($\sigma = 5.21$) | **45.81 t/s** ($\sigma = 1.11$) | **+98.41% faster prompt prefill (+45.08 t/s)** |
| **Average End-to-End Latency** | **11.78 s** ($\sigma = 3.46$) | **16.34 s** ($\sigma = 4.76$) | **-27.91% lower latency (-4.56 s faster)** |
| **Peak Process RAM Consumption** | **1.650 GB** (Mean: 1.643 GB) | **1.680 GB** (Mean: 1.673 GB) | **-1.79% lower peak RAM** |
| **Average Quality Score (out of 10)**| **9.20 / 10** | **8.95 / 10** | **+2.79% quality retention** |
| **Request Reliability** | **10/10 (100% success)** | **10/10 (100% success)** | **100% stable execution across all prompts** |

---

## 2. Hardware Environment Profile

The benchmarking suite automatically detects and profiles the host environment:

| Property | Value |
| :--- | :--- |
| **System Model** | HP Victus 15-fb0xxx Laptop |
| **Operating System** | Windows 11 Home 64-bit (10.0.26200) |
| **Processor (CPU)** | AMD Ryzen 5 5600H with Radeon Graphics (Zen 3, 6 physical cores / 12 threads) |
| **SIMD Extensions** | AVX, AVX2, FMA, F16C |
| **Host System RAM** | 16.0 GB (15.34 GB visible, ~5.4 GB available during testing) |
| **Dedicated GPU** | AMD Radeon(TM) RX 6500M (4,080 MiB VRAM, Vulkan 1.3, Navi 24 RDNA2) |
| **Integrated GPU** | AMD Radeon(TM) Graphics (8,111 MiB shared memory, Vulkan 1.3) |
| **Development Storage**| Drive D: (NVMe SSD, 145 GB available space for model hosting) |
| **Python Runtime** | Python 3.12.2 (AMD64) |

*Full hardware telemetry is saved to [`results/hardware_profile.json`](file:///c:/Users/hp/OneDrive/Desktop/RAG/LLM%20Optimization/Task-3-LLM-Optimisation/results/hardware_profile.json).*

---

## 3. Repository Architecture

```
Task-3-LLM-Optimisation/
│
├── models/                          # Model documentation & storage specification
│   └── README.md
│
├── data/                            # Evaluation dataset
│   └── benchmark_prompts.json       # 10 capability test prompts with rubrics
│
├── src/                             # Modular Python benchmarking engine
│   ├── hardware_detection.py        # System topology & Vulkan/CPU backend detection
│   ├── model_manager.py             # GGUF acquisition, SHA-256 & architectural parity checks
│   ├── inference.py                 # Common llama.cpp interface & telemetry parser
│   ├── memory_monitor.py            # Real-time process RAM & peak memory tracker
│   ├── quality_evaluation.py        # Automated rubric & rule-based quality scorer
│   └── benchmark.py                 # End-to-end benchmark orchestrator & chart generator
│
├── results/                         # Machine-readable outputs & visual charts
│   ├── hardware_profile.json        # Detailed host hardware telemetry
│   ├── benchmark_results.json       # Full raw timings, tokens, and summary statistics
│   ├── benchmark_results.csv        # Tabular performance metrics
│   ├── quality_comparison.csv       # Side-by-side prompt outputs, scores & observations
│   ├── quality_comparison.json      # Structured quality comparison dataset
│   ├── performance_comparison.png   # Generation vs. Prefill Tokens/Second chart
│   ├── memory_comparison.png        # Storage vs. Active Peak RAM chart
│   └── quality_comparison.png       # 10-prompt capability score breakdown chart
│
├── reports/                         # Deep technical documentation
│   ├── benchmark_analysis.md        # Comprehensive performance & memory analysis
│   ├── kv_cache_explanation.md      # Mathematical derivation of KV Cache & Quantisation
│   └── concurrent_serving_plan.md   # Production plan: 20 users on a 24 GB GPU
│
├── notebooks/                       # Interactive exploration
│   └── benchmark_analysis.ipynb     # Jupyter notebook reproducing all data & plots
│
├── tools/                           # llama.cpp official release binaries (b12532)
│   ├── llama-cpu/                   # AVX2 CPU binaries
│   └── llama-vulkan/                # Vulkan AMD GPU binaries
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Project technical manual
└── .gitignore                       # Clean Git configuration (models excluded)
```

---

## 4. Quick Start & Execution

### Prerequisites
Clone or navigate to the project directory and install the lightweight Python dependencies:

```powershell
cd "c:\Users\hp\OneDrive\Desktop\RAG\LLM Optimization\Task-3-LLM-Optimisation"
pip install -r requirements.txt
```

### 1. Run Hardware Detection
```powershell
python src/hardware_detection.py
```

### 2. Verify Models & GGUF Parity
```powershell
python src/model_manager.py
```

### 3. Run the Complete Benchmark Suite
Execute the entire evaluation (both models, all 10 prompts, memory tracking, quality evaluation, and chart generation) with a single command:

```powershell
python src/benchmark.py --backend cpu --threads 6
```

Optional arguments:
- `--backend`: Choose `cpu` (default) or `vulkan` (AMD RX 6500M GPU).
- `--threads`: OpenMP execution threads (default: `6`).
- `--quants`: Target quantisations (default: `Q4_K_M Q8_0`).

---

## 5. Benchmarking Methodology & Evaluation Prompts

The benchmark evaluates the model across 10 distinct cognitive and technical capabilities defined in [`data/benchmark_prompts.json`](file:///c:/Users/hp/OneDrive/Desktop/RAG/LLM%20Optimization/Task-3-LLM-Optimisation/data/benchmark_prompts.json):

1. **General Reasoning (`PROMPT-01`):** Deductive seating arrangement with negative constraints.
2. **Python Programming (`PROMPT-02`):** `find_top_k_frequent` with type annotations, docstring, and syntax validation.
3. **Mathematical Problem Solving (`PROMPT-03`):** Two-step compounding percentage discount calculation.
4. **Text Summarisation (`PROMPT-04`):** Strict 2-sentence constraint covering quantum superposition and post-quantum cryptography.
5. **Structured JSON Generation (`PROMPT-05`):** Pure JSON extraction with strict schema and zero conversational prose.
6. **Technical Explanation (`PROMPT-06`):** TCP vs. UDP protocols structured across 3 labeled technical sections.
7. **Logical Reasoning (`PROMPT-07`):** Formal logic premises evaluating Modus Tollens and avoiding Denying the Antecedent.
8. **Information Extraction (`PROMPT-08`):** Invoice parsing into line items, quantities, tax, and totals.
9. **Instruction Following (`PROMPT-09`):** 5 strict positive and negative constraints (length, start word, forbidden words, exact ending).
10. **Multi-Step Problem Solving (`PROMPT-10`):** Multi-developer sprint capacity arithmetic, overhead deduction, and backlog feasibility.

---

## 6. Technical Explanations

### A. KV Cache Architecture
Autoregressive language models generate tokens sequentially. To prevent recomputing historical Key and Value representations for all previous tokens at quadratic complexity $\mathcal{O}(N^2)$, the attention layers store past Key and Value projection states in a dedicated **KV Cache**.

The general memory consumption formula for a Grouped-Query Attention (GQA) cache is:

$$\text{Memory}_{\text{KV}} = 2 \times L \times H_{KV} \times d_{\text{head}} \times S_{\text{ctx}} \times P_{\text{bytes}} \times B$$

Where $L$ is layer count, $H_{KV}$ is KV head count, $d_{\text{head}}$ is head dimension, $S_{\text{ctx}}$ is cached token count, $P_{\text{bytes}}$ is precision bytes (2.0 for FP16), and $B$ is batch size.

*Read the full mathematical derivation in [`reports/kv_cache_explanation.md`](file:///c:/Users/hp/OneDrive/Desktop/RAG/LLM%20Optimization/Task-3-LLM-Optimisation/reports/kv_cache_explanation.md).*

### B. Quantisation Mechanics
Quantisation maps continuous floating-point weights to low-bit integers (e.g., INT8 or 4-bit k-quants) using block-level scaling factors. In memory-bandwidth-bound single-user inference, quantisation dramatically reduces the volume of bytes that must cross the memory bus for every token generated, directly increasing generation throughput.

---

## 7. Serving 20 Concurrent Users on a 24 GB GPU

[`reports/concurrent_serving_plan.md`](file:///c:/Users/hp/OneDrive/Desktop/RAG/LLM%20Optimization/Task-3-LLM-Optimisation/reports/concurrent_serving_plan.md) provides a comprehensive production architecture for serving 20 simultaneous users on an enterprise 24 GB GPU (e.g. RTX 4090 / A10G) using **`Qwen2.5-7B-Instruct`** in 4-bit AWQ with **vLLM** and **PagedAttention**:

- **Model Weights (AWQ 4-bit):** 4.97 GB (20.2%)
- **Active KV Cache Pool (20 users × 4,096 context):** 4.70 GB (19.1%)
- **Runtime Activations & PyTorch Context:** 2.20 GB (8.9%)
- **Dynamic PagedAttention Expansion Headroom:** 9.71 GB (39.5%)
- **Safety OOM Reserve:** 3.00 GB (12.3%)
- **Total VRAM:** **24.58 GB (100.0%)**

The architecture features iteration-level continuous batching, chunked prefill (512 tokens), HTTP 429 admission control, and dynamic FP8 KV cache fallback.
