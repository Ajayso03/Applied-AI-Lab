# Technical Deep-Dive: Key-Value (KV) Cache and Quantisation Mechanics in LLM Inference

**Author:** Senior Machine Learning Engineer & Performance Benchmarking Researcher  
**Target Architecture:** Qwen2.5 / Llama Family (GQA Transformer Decoders)  
**Runtime:** llama.cpp (GGML Engine)  

---

## Part 1: The Key-Value (KV) Cache Architecture

### 1. Mathematical Foundation: Self-Attention in Autoregressive Generation

In a decoder-only transformer architecture, self-attention maps an input sequence representation into Queries ($Q$), Keys ($K$), and Values ($V$) via linear projection matrices:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

Where:
- $X \in \mathbb{R}^{S \times d_{\text{model}}}$ is the sequence representation at sequence length $S$.
- $W_Q \in \mathbb{R}^{d_{\text{model}} \times (H_Q \cdot d_{\text{head}})}$.
- $W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times (H_{KV} \cdot d_{\text{head}})}$.

The scaled dot-product attention for each head is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M\right) V$$

Where $M$ is the causal attention mask ($M_{i,j} = -\infty$ for $j > i$).

---

### 2. The Computational Inefficiency of Autoregressive Generation Without Caching

During inference, autoregressive text generation proceeds token-by-token:
1. At step $t=1$, the model processes $S$ prompt tokens.
2. At step $t=2$, the model generates token $y_1$ and appends it to form a sequence of length $S+1$.
3. At step $t$, the sequence length is $S + t - 1$.

If previous states are discarded:
- To generate token $y_t$, the model must re-embed all $S + t - 1$ tokens.
- Every transformer layer must recompute $W_Q, W_K, W_V$ projections and causal self-attention over the entire historical sequence.
- The computational complexity for generating $T$ tokens across context $S$ scales quadratically:

$$\mathcal{O}\left(\sum_{t=1}^T (S + t)^2 \cdot d_{\text{model}}\right) \approx \mathcal{O}\left(T \cdot S^2 \cdot d_{\text{model}} + T^2 \cdot S \cdot d_{\text{model}} + \frac{1}{3}T^3 \cdot d_{\text{model}}\right)$$

This represents immense wasted compute: **99%+ of the Floating Point Operations (FLOPs) would be redundant re-evaluations of static historical tokens**.

---

### 3. Why Caching Works: The Autoregressive Invariant

Due to the causal attention mask, token $i$ can only attend to tokens $j \le i$. Therefore:
- The Key and Value representations of token $j$ ($K_j = x_j W_K$ and $V_j = x_j W_V$) **do not change** as subsequent tokens $j+1, j+2, \dots$ are generated.
- Only the Query vector of the *current* token ($Q_{\text{current}} = x_{\text{current}} W_Q$) is needed to query all historical Key vectors ($K_{1:t}$) and aggregate historical Value vectors ($V_{1:t}$).

By caching the calculated Key and Value activation tensors for all previous tokens in memory:
- Step complexity drops from $\mathcal{O}((S+t)^2)$ to $\mathcal{O}(S+t)$.
- Time per generated token remains bounded and dominated by memory bandwidth rather than quadratic arithmetic compute.

```
Token t-1: [Prompt Tokens (1..S)] + [Generated Tokens (1..t-1)]
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Historical K         Historical V  ───► STORED IN KV CACHE
             │                   │
Token t:     │  New Token t      │
             │       │           │
             │   Query Q_t       │
             ▼       ▼           ▼
        Scaled Dot-Product Attention: Softmax(Q_t · K_cached^T / sqrt(d)) · V_cached
```

---

### 4. KV Cache Memory Scaling & Mathematical Derivation

For each cached token, the model must store its Key vector and Value vector across every layer of the transformer.

#### General Analytical Formula

$$\text{Memory}_{\text{KV}} = 2 \times L \times H_{KV} \times d_{\text{head}} \times S_{\text{ctx}} \times P_{\text{bytes}} \times B$$

Where:
- $2$: Accounts for the two cached tensors per layer (one Key tensor, one Value tensor).
- $L$: Total number of transformer layers (`block_count`).
- $H_{KV}$: Number of Key-Value attention heads per layer.
- $d_{\text{head}}$: Dimension of each attention head ($d_{\text{model}} / H_Q$).
- $S_{\text{ctx}}$: Total sequence length cached (prompt tokens + generated tokens).
- $P_{\text{bytes}}$: Storage precision in bytes per element (e.g., FP16 = 2.0 bytes, FP8 = 1.0 byte, Q4 = 0.5 bytes).
- $B$: Batch size (number of concurrent sequences).

---

### 5. Multi-Head Attention (MHA) vs. Grouped-Query Attention (GQA)

Traditional transformers (e.g., standard LLaMA-1, GPT-3) used **Multi-Head Attention (MHA)**, where $H_{KV} = H_Q$. Each Query head possessed an independent Key and Value head.

Modern architectures (including **Qwen2.5** and **Llama-3**) employ **Grouped-Query Attention (GQA)**:
- Multiple Query heads ($H_Q$) share a smaller number of Key-Value heads ($H_{KV}$).
- Grouping ratio: $G = H_Q / H_{KV}$.

#### Comparison on Qwen2.5-1.5B:
- $L = 28$ layers
- $H_Q = 12$ query heads
- $H_{KV} = 2$ key-value heads ($G = 6:1$ ratio)
- $d_{\text{head}} = 128$
- Precision: 16-bit float ($P_{\text{bytes}} = 2.0$)

**If Qwen2.5-1.5B used MHA ($H_{KV} = 12$):**
$$\text{Bytes per token} = 2 \times 28 \times 12 \times 128 \times 2 = 172,032 \text{ bytes} \approx \mathbf{168.0 \text{ KB/token}}$$
At 32,768 context length, 1 sequence = **5.50 GB**.

**With Grouped-Query Attention ($H_{KV} = 2$):**
$$\text{Bytes per token} = 2 \times 28 \times 2 \times 128 \times 2 = 28,672 \text{ bytes} \approx \mathbf{28.0 \text{ KB/token}}$$
At 32,768 context length, 1 sequence = **0.917 GB** (an **83.3% reduction** in KV cache memory!).

---

## Part 2: Quantisation Mechanics & Trade-offs

### 1. Fundamentals of Numerical Quantisation

Quantisation is the process of mapping high-precision, continuous floating-point weights (e.g., IEEE FP32 or FP16) to discrete, low-bitwidth integer representations (e.g., INT8 or 4-bit integers).

| Precision Format | Bits per Weight | Numerical Range | Dynamic Range / Precision |
| :--- | :--- | :--- | :--- |
| **FP32** (Single Precision) | 32 bits (4 bytes) | $\pm 3.4 \times 10^{38}$ | 23-bit mantissa, 8-bit exponent |
| **FP16 / BF16** (Half Precision) | 16 bits (2 bytes) | $\pm 65,504$ / $\pm 3.39 \times 10^{38}$ | Standard baseline for LLM training |
| **INT8 / Q8_0** | 8 bits (1 byte) | $[-128, 127]$ | Linear mapping via scaling factor $\Delta$ |
| **Q4_K_M** (4-bit Mixed) | ~4.5 bits effective | $[-8, 7]$ | Non-linear block scales with 6-bit super-blocks |

---

### 2. Block-Level Quantisation in llama.cpp / GGML

Uniform integer quantisation calculates a scale factor $\Delta$ across a block of $N$ weights:

$$W_{\text{quant}} = \text{clamp}\left(\text{round}\left(\frac{W}{\Delta}\right), -2^{b-1}, 2^{b-1}-1\right), \quad \Delta = \frac{\max(|W|)}{2^{b-1}-1}$$

During matrix-vector multiplication (GEMV), weights are dequantised on-the-fly:

$$\hat{W} = W_{\text{quant}} \times \Delta$$

#### Why Block Scales and Super-Blocks Matter:
- If a single scale $\Delta$ is used for an entire layer, a single outlier weight will compress all normal weights into 0 or 1, obliterating model reasoning.
- GGML uses **blocks** (e.g., $N=32$ in `Q8_0` or $N=256$ in `k-quants`).
- In **`Q4_K_M`**, a 256-element block is divided into 16 mini-blocks of 16 weights. Each mini-block has a 4-bit scale, which is itself scaled by a 6-bit super-block scale factor. This captures localized weight distributions while adding minimal metadata overhead (~0.5 bits per weight).

---

### 3. The Memory-Bandwidth Bottleneck: Why Quantisation Improves Speed

LLM autoregressive generation (batch size = 1) is strictly **memory bandwidth bound**, not compute bound.
- For each token generated, every single parameter of the model must be loaded from RAM into the CPU/GPU caches and ALUs to compute vector products with a single token vector ($1 \times d_{\text{model}}$).
- The arithmetic intensity is extremely low:

$$\text{Arithmetic Intensity} = \frac{2 \times N_{\text{parameters}} \text{ FLOPs}}{N_{\text{parameters}} \times \text{Bytes per parameter}} \approx \frac{2}{P_{\text{bytes}}} \approx 0.5 \text{ to } 1.0 \text{ FLOP/byte}$$

On a CPU with dual-channel DDR4-3200 RAM (~40 GB/s bandwidth):
- Loading an FP16 1.5B model (3.0 GB) per token:
  $$\text{Theoretical Max TPS} = \frac{40 \text{ GB/s}}{3.0 \text{ GB}} \approx 13.3 \text{ tokens/s}$$
- Loading a Q4_K_M 1.5B model (0.98 GB) per token:
  $$\text{Theoretical Max TPS} = \frac{40 \text{ GB/s}}{0.98 \text{ GB}} \approx \mathbf{40.8 \text{ tokens/s}}$$

**Quantisation accelerates generation because 3× fewer bytes must travel across the memory bus per generated token.**

---

### 4. When Quantisation Does NOT Improve Speed

Quantisation does not guarantee speedups in all environments:
1. **Compute-Bound Regimes (Prompt Processing / High Batch Sizes):**
   When processing long prompts ($S > 512$) or large batches, matrix-matrix multiplications (GEMM) are compute bound. Dequantizing low-bit integers back to floating point introduces extra ALU overhead.
2. **CPUs Lacking Hardware Vector Acceleration:**
   If the CPU lacks AVX2, AVX-512 VNNI, or ARM NEON/dotprod instructions, integer unpacking must be simulated via bit shifts and masks, which can bottleneck the CPU pipeline.
3. **Small Models with Overhead Saturation:**
   For ultra-small models (< 500M parameters), thread synchronization, OS scheduling, and memory mapping latency exceed memory bus transfer times.
