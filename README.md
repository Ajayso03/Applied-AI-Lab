# DisasterRAG – Custom Retrieval-Augmented Generation Engine
### Built from Scratch for tndrr AI Internship Assignment – Task 2

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.15.0-green.svg)](https://github.com/facebookresearch/faiss)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![NDMA Documents](https://img.shields.io/badge/NDMA%20Documents-488%20Pages-red.svg)](https://ndma.gov.in)
[![No Frameworks](https://img.shields.io/badge/Frameworks-100%25%20From%20Scratch-purple.svg)](#3-rag-architecture)

> **Core Mandate:** Complete, technically rigorous RAG implementation developed **entirely from first principles** without LangChain, LlamaIndex, Haystack, or any third-party RAG abstraction layers. Prioritizes mathematical transparency, reproducibility, genuine empirical retrieval benchmarks, and exact document provenance citations.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [RAG Architecture](#3-rag-architecture)
4. [Document Sources & Provenance](#4-document-sources--provenance)
5. [Document Ingestion Pipeline](#5-document-ingestion-pipeline)
6. [Chunking Methodology](#6-chunking-methodology)
7. [Embedding Model Selection](#7-embedding-model-selection)
8. [Vector Similarity Calculations & Mathematical Derivation](#8-vector-similarity-calculations--mathematical-derivation)
9. [Retrieval Implementation](#9-retrieval-implementation)
10. [Answer Generation & Guardrails](#10-answer-generation--guardrails)
11. [Evaluation Methodology](#11-evaluation-methodology)
12. [Recall@5 Comparison & Empirical Findings](#12-recall5-comparison--empirical-findings)
13. [Two Retrieval Failure Cases](#13-two-retrieval-failure-cases)
14. [BM25 In-Depth Explanation](#14-bm25-in-depth-explanation)
15. [Re-Ranker In-Depth Explanation](#15-re-ranker-in-depth-explanation)
16. [Installation Instructions](#16-installation-instructions)
17. [Execution Commands (Windows PowerShell)](#17-execution-commands-windows-powershell)
18. [Hardware Requirements & Execution Environment](#18-hardware-requirements--execution-environment)
19. [System Limitations](#19-system-limitations)
20. [Future Architectural Improvements](#20-future-architectural-improvements)

---

## 1. Project Overview

**DisasterRAG** is a domain-specific Question-Answering engine operating over the official disaster-management guidelines published by the **National Disaster Management Authority (NDMA)**, Government of India. The system provides verifiable, hallucination-resistant answers grounded strictly in official policy guidelines for flood mitigation, earthquake engineering, cyclone risk reduction, and national emergency response frameworks.

Every response produced by DisasterRAG attributes its claims to source documents and exact page numbers (e.g., `[Doc: NDMA_Floods_Guidelines.pdf, Page: 58]`).

---

## 2. Problem Statement

Standard Large Language Models (LLMs) suffer from two fundamental weaknesses when deployed in high-stakes public safety and regulatory environments:
1. **Hallucination & Factual Drift**: LLMs generate plausible-sounding but factually invalid disaster protocols, code numbers, and statutory authorities.
2. **Lack of Verifiable Provenance**: LLMs cannot reliably cite exact page numbers, statutory paragraphs, or official publication URLs from authoritative government sources.

DisasterRAG solves this by implementing an independent, transparent retrieval pipeline that decouples dense/sparse passage retrieval from language generation, enabling rigorous quantitative evaluation (Recall@5, MRR@5) before any text generation occurs.

---

## 3. RAG Architecture

The DisasterRAG pipeline is designed with clean modular separation:

```
[Official NDMA PDFs (488 Pages)]
             │
             ▼
    [Document Loader]  ◄─── Extracts text via PyMuPDF; preserves page metadata & URLs
             │
             ▼
     [Text Cleaner]    ◄─── Normalizes whitespace, fixes hyphenated linebreaks, removes artifacts
             │
             ▼
     [Chunking Engine] ◄─── Tokenizer-aware sliding window (Config A: ~300 / Config B: ~600)
             │
             ▼
   [Embedding Generator] ◄─── sentence-transformers/all-MiniLM-L6-v2 + L2 Normalization
             │
             ▼
    [Vector Store]     ◄─── Dual Engine: Pure NumPy Dot-Product & FAISS IndexFlatIP
             │
      ┌──────┴──────────────────────────┐
      ▼                                 ▼
[Dense Semantic Search]        [BM25 Keyword Search] (Optional Hybrid)
      │                                 │
      └──────► [Reciprocal Rank Fusion] ◄┘
                       │
                       ▼
            [Retrieved Top-5 Passages]
                       │
                       ▼
            [Answer Generator] ◄─── Strict contextual prompt; cites Doc, Page & URL
                       │
                       ▼
            [Grounded Answer Output]
```

### File Hierarchy
```
Custom-RAG/
├── documents/
│   ├── README.md                              # Document manifest and source URLs
│   ├── NDMA_Floods_Guidelines.pdf             # 168 pages
│   ├── NDMA_Earthquakes_Guidelines.pdf        # 74 pages
│   ├── NDMA_Cyclones_Guidelines.pdf           # 190 pages
│   └── NDMA_National_DM_Policy_2009.pdf       # 56 pages
├── data/
│   ├── extracted_pages.jsonl                  # 488 page records with metadata
│   ├── chunks_300.jsonl                       # 857 chunks (~300 tokens, 50 overlap)
│   ├── chunks_600.jsonl                       # 429 chunks (~600 tokens, 100 overlap)
│   └── evaluation_questions.json              # 15 ground-truth evaluation questions
├── embeddings/
│   ├── embeddings_300.npy                     # (857, 384) float32 NumPy array
│   └── embeddings_600.npy                     # (429, 384) float32 NumPy array
├── indexes/
│   ├── index_300/                             # FAISS IndexFlatIP & chunks metadata
│   └── index_600/                             # FAISS IndexFlatIP & chunks metadata
├── src/
│   ├── document_loader.py                     # PyMuPDF ingestion & page preservation
│   ├── text_cleaner.py                        # Deterministic text normalization
│   ├── chunking.py                            # Tokenizer-aware sliding chunker
│   ├── embeddings.py                          # Offline embedding inference & caching
│   ├── vector_store.py                        # Pure NumPy & FAISS vector search
│   ├── retrieval.py                           # Standalone semantic retrieval CLI
│   ├── answer_generator.py                    # Prompt-based grounded synthesis & fallback
│   ├── rag_pipeline.py                        # Unified end-to-end RAG pipeline
│   ├── evaluate.py                            # Recall@5 benchmark & chart generation
│   └── hybrid_retrieval.py                    # BM25 + Dense Reciprocal Rank Fusion
├── results/
│   ├── recall_comparison.json                 # Quantitative benchmark metrics
│   ├── question_results.csv                   # Per-question rank, score, and hit status
│   ├── failure_analysis.md                    # In-depth analysis of 2 genuine failure cases
│   ├── hybrid_comparison.json                 # Dense vs Hybrid benchmark metrics
│   └── recall_comparison.png                  # Matplotlib performance visualization
├── app/
│   └── streamlit_app.py                       # Interactive Streamlit Web UI
├── notebooks/
│   └── retrieval_analysis.ipynb               # Complete Jupyter evaluation walkthrough
├── requirements.txt                           # Version-pinned dependencies
├── .env.example                               # Environment configuration template
├── .gitignore                                 # Git ignore patterns
└── README.md                                  # Complete technical documentation
```

---

## 4. Document Sources & Provenance

All documents were obtained directly from official public portals of the National Disaster Management Authority (NDMA), Government of India:

| # | File Name | Official Document Title | Pages | Size | Canonical Source URL |
|---|---|---|:---:|:---:|---|
| 1 | `NDMA_Floods_Guidelines.pdf` | National Disaster Management Guidelines: Management of Floods | 168 | 5.84 MB | [ndma.gov.in Guidelines/flood.pdf](https://ndma.gov.in/sites/default/files/PDF/Guidelines/flood.pdf) |
| 2 | `NDMA_Earthquakes_Guidelines.pdf` | National Disaster Management Guidelines: Management of Earthquakes | 74 | 3.16 MB | [ndma.gov.in Guidelines/earthquakes.pdf](https://ndma.gov.in/sites/default/files/PDF/Guidelines/earthquakes.pdf) |
| 3 | `NDMA_Cyclones_Guidelines.pdf` | National Disaster Management Guidelines: Management of Cyclones | 190 | 4.52 MB | [ndma.gov.in Guidelines/cyclones.pdf](https://ndma.gov.in/sites/default/files/PDF/Guidelines/cyclones.pdf) |
| 4 | `NDMA_National_DM_Policy_2009.pdf` | National Policy on Disaster Management 2009 | 56 | 1.38 MB | [ndma.gov.in PDF/national-dm-policy2009.pdf](https://ndma.gov.in/sites/default/files/PDF/national-dm-policy2009.pdf) |

**Total Ingested Pages:** **488 pages** (requirement: $\ge 30$ pages).  
**Total Words Extracted:** **159,273 words** across **1,090,350 characters**.

---

## 5. Document Ingestion Pipeline

The document ingestion pipeline (`src/document_loader.py` and `src/text_cleaner.py`) implements:
1. **PyMuPDF (`fitz`) Text Extraction**: Iterates over PDF pages sequentially, extracting raw text streams without external CLI utilities.
2. **Metadata Preservation**: For every page, preserves:
   - `document_name`: Exact PDF filename.
   - `page_number`: 1-indexed human-readable page number matching the physical document.
   - `source_url`: Verifiable government source URL.
   - `raw_char_count` & `cleaned_char_count`.
3. **Empty Page & Scan Handling**: Detects cover illustrations, unpopulated divider pages, and empty scans by enforcing a minimum text threshold (`min_char_threshold = 25`). Empty pages are flagged (`"is_empty": true`) and excluded from downstream chunking.
4. **Deterministic Cleaning Pipeline**:
   - NFKC Unicode normalization (normalizing typographic ligatures e.g. `fi`, `fl`).
   - Linebreak de-hyphenation (e.g. `disas-\nter` $\rightarrow$ `disaster`).
   - Normalization of horizontal whitespace runs and consecutive newlines ($\ge 3$ newlines $\rightarrow$ `\n\n`).
   - Removal of non-printable control characters.
5. **Structured Output**: Extracted pages are serialized to `data/extracted_pages.jsonl`.

---

## 6. Chunking Methodology

Implemented in `src/chunking.py`. Two distinct configurations are evaluated:

| Parameter | Configuration A | Configuration B |
|---|:---:|:---:|
| **Target Chunk Size** | ~300 tokens | ~600 tokens |
| **Token Overlap** | 50 tokens | 100 tokens |
| **Effective Step Size** | 250 tokens | 500 tokens |
| **Total Chunks Generated** | **857 chunks** | **429 chunks** |
| **Empirical Mean Tokens** | **299.3 tokens** | **597.6 tokens** |
| **Boundary Discipline** | Document-contained | Document-contained |

### Tokenizer-Aware Strategy
Unlike naive character-splitting or regex word-splitting:
1. The chunker uses the exact HuggingFace `AutoTokenizer` of the embedding model (`all-MiniLM-L6-v2`) to encode token IDs directly.
2. Token windows slide with deterministic overlap.
3. Every token maintains a back-pointer to its source PDF page (`token_page_map`). The primary `page_number` for each chunk is assigned via mode aggregation (the page contributing the majority of tokens in that chunk), while a full list (`page_numbers`) records multi-page boundaries.
4. Chunks never cross document boundaries.

---

## 7. Embedding Model Selection

We selected **`sentence-transformers/all-MiniLM-L6-v2`**:
- **Architecture**: 6-layer MiniLM transformer with 384-dimensional output embeddings.
- **Parameters**: ~22.7 Million parameters (extremely lightweight).
- **Execution Mode**: CPU-only inference using PyTorch (`device="cpu"`), requiring zero NVIDIA CUDA dependencies.
- **Offline Caching**: Automatically uses the pre-cached weights stored in `D:\AI-Assignments\Models\HuggingFace\hub\models--sentence-transformers--all-MiniLM-L6-v2`.
- **Latency**: Processes ~64 chunks in ~2.8 seconds on AMD Ryzen 5 5600H CPU (~43ms per chunk batch).

---

## 8. Vector Similarity Calculations & Mathematical Derivation

Implemented in `src/vector_store.py`.

### Mathematical Derivation
Let $\mathbf{q} \in \mathbb{R}^D$ denote the query embedding vector, and let $\mathbf{x}_i \in \mathbb{R}^D$ denote the embedding of chunk $i$ ($D = 384$).

The Cosine Similarity between the query and chunk is defined as:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{x}_i) = \frac{\mathbf{q} \cdot \mathbf{x}_i}{\|\mathbf{q}\|_2 \|\mathbf{x}_i\|_2} = \frac{\sum_{j=1}^D q_j x_{ij}}{\sqrt{\sum_{j=1}^D q_j^2} \sqrt{\sum_{j=1}^D x_{ij}^2}}$$

During the offline embedding phase (`src/embeddings.py`), every chunk embedding vector is pre-normalized using the Euclidean $L_2$ norm:
$$\mathbf{x}_i \leftarrow \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2} \implies \|\mathbf{x}_i\|_2 = 1.0 \quad \forall i \in \{1, \dots, N\}$$

Similarly, upon query arrival, the query vector is normalized:
$$\mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|_2} \implies \|\mathbf{q}\|_2 = 1.0$$

Substituting $\|\mathbf{q}\|_2 = 1$ and $\|\mathbf{x}_i\|_2 = 1$ into the cosine similarity equation:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{x}_i) = \mathbf{q} \cdot \mathbf{x}_i$$

### Vectorized Matrix Formulation
Let $\mathbf{X} \in \mathbb{R}^{N \times D}$ denote the full normalized corpus embedding matrix. The similarity scores across all $N$ corpus chunks are computed simultaneously via single matrix-vector multiplication:
$$\mathbf{s} = \mathbf{X} \mathbf{q} \in \mathbb{R}^N$$

The top-$k$ retrieved indices are obtained by descending argsort:
$$\mathcal{I}_{\text{top-}k} = \text{argsort}(\mathbf{s})[::-1][:k]$$

### Dual-Engine Verification
DisasterRAG provides both:
1. **Pure NumPy Engine** (`search_numpy`): Uses `np.dot(self.embeddings, q)` and `np.argsort`.
2. **FAISS Engine** (`search_faiss`): Uses `faiss.IndexFlatIP` (Inner Product on unit vectors), leveraging AVX2 SIMD acceleration.

Both engines produce **identical similarity scores** to within machine epsilon ($\max |\text{score}_{\text{NumPy}} - \text{score}_{\text{FAISS}}| < 10^{-6}$).

---

## 9. Retrieval Implementation

Implemented in `src/retrieval.py`.
- Completely decoupled from LLM generation.
- Accepts natural language queries via Python API or command-line.
- Returns structured results containing rank, cosine score, chunk ID, document name, page number, canonical URL, and chunk text.
- Measures and logs query embedding latency and search latency in milliseconds.

CLI usage:
```powershell
python src/retrieval.py --query "What are the structural mitigation measures for flood management?" --config 300 --top-k 5
```

---

## 10. Answer Generation & Guardrails

Implemented in `src/answer_generator.py` and `src/rag_pipeline.py`.

### Guardrail System Prompt
```
You are DisasterRAG, an authoritative AI assistant specializing in public disaster-management guidelines.
Your goal is to provide accurate, grounded answers based EXCLUSIVELY on the provided official context passages.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the provided context passages. Do NOT invent, extrapolate, or introduce external knowledge.
2. If the context passages do not contain enough information to answer the question, state:
   "The provided documents do not contain sufficient information to answer this question."
3. Every factual statement must cite its source document and page number using: [Doc: <filename>, Page: <page_number>].
4. At the end of your answer, provide a dedicated "Citations" section listing the unique source documents, page numbers, and original URLs.
```

### Multi-Backend Architecture
1. **Offline Grounded Extractive Synthesis (Default)**: Runs 100% locally on CPU without API keys. Identifies semantically matching sentences from the retrieved passages, enforces factual fidelity, and formats citations.
2. **External API Support**: Supports OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), or Groq if environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`) are provided.

---

## 11. Evaluation Methodology

Evaluation is executed by `src/evaluate.py` using `data/evaluation_questions.json`.

### Evaluation Dataset (15 Questions)
15 original questions were created based on the 488 pages, covering all 4 documents:
- Questions 1–4: `NDMA_National_DM_Policy_2009.pdf` (NEC, NDRF, Capacity Development, Force Structure).
- Questions 5–8: `NDMA_Earthquakes_Guidelines.pdf` (RVS, Lifeline Retrofitting, IS 1893, Education).
- Questions 9–12: `NDMA_Floods_Guidelines.pdf` (CWC Forecasting, Structural Embankments, Floodplain Zoning, Drainage Congestion).
- Questions 13–15: `NDMA_Cyclones_Guidelines.pdf` (NCRMP Components, NDCI Computing, Bio-shields/Mangroves).

### Relevance Judgement Rules
A retrieved passage is counted as a relevant hit if:
1. **Rule 1 (Exact Chunk ID)**: Matches the ground-truth chunk ID; OR
2. **Rule 2 (Exact Page Match)**: Document matches and page equals the ground-truth page (or is included in the chunk's `page_numbers` span); OR
3. **Rule 3 (Adjacent Page with Evidence)**: Document matches, page is within $\pm 1$ of the ground-truth page, and key ground-truth textual evidence is present in the chunk.

### Metric Definitions
- **Recall@5**:
  $$\text{Recall@5} = \frac{\text{Questions with } \ge 1 \text{ relevant passage in top 5}}{Q}$$
- **Mean Reciprocal Rank (MRR@5)**:
  $$\text{MRR@5} = \frac{1}{Q} \sum_{i=1}^Q \frac{1}{\text{rank}_i} \quad (\text{rank}_i = \text{rank of first relevant chunk, or } \infty \text{ if missed})$$

---

## 12. Recall@5 Comparison & Empirical Findings

Below are the **actual, unadjusted empirical results** from running `python src/evaluate.py`:

| Metric | Configuration A (300 tokens, 50 overlap) | Configuration B (600 tokens, 100 overlap) | Delta (Config B vs A) |
|---|:---:|:---:|:---:|
| **Total Questions Evaluated** | 15 | 15 | 0 |
| **Total Chunks in Corpus** | 857 chunks | 429 chunks | -428 chunks |
| **Hits @ 1** | 4 / 15 | **8 / 15** | **+4 (+100%)** |
| **Recall @ 1** | **26.7%** | **53.3%** | **+26.6%** |
| **Hits @ 3** | 11 / 15 | 10 / 15 | -1 |
| **Recall @ 3** | **73.3%** | **66.7%** | -6.6% |
| **Hits @ 5** | 12 / 15 | 12 / 15 | 0 |
| **Recall @ 5** | **80.0% (12/15)** | **80.0% (12/15)** | **0.0%** |
| **MRR @ 5** | **0.5056** | **0.6333** | **+0.1277 (+25.3%)** |
| **Average Query Latency** | 26.10 ms | **21.21 ms** | **-4.89 ms (18.7% faster)** |

![Recall@5 Comparison Chart](results/recall_comparison.png)

### Why Configuration B (600 Tokens) Performed Better Overall:
1. **Substantially Higher MRR (0.6333 vs 0.5056)**: Relevant passages were positioned higher in the ranking list. Configuration B achieved an **8/15 (53.3%) Recall@1**, whereas Configuration A achieved only **4/15 (26.7%) Recall@1**.
2. **Context Completeness**: A 600-token chunk encapsulates complete institutional mandates (such as the four components of NCRMP or the complete list of NEC members) in a single coherent context block. In 300-token chunks, multi-item regulatory lists were frequently severed across chunk boundaries.
3. **Lower Index Search Latency**: With 429 vectors versus 857 vectors, the vector search operation had $50\%$ fewer vector inner products to compute, reducing average latency from 26.1ms to 21.2ms.

---

## 13. Two Retrieval Failure Cases

Forensic analysis documented in `results/failure_analysis.md`:

### Failure Case 1: Q11 (Cross-Chapter Semantic Competition)
- **Query:** *"What is Flood Plain Zoning and what is its objective in non-structural flood regulation?"*
- **Expected:** `NDMA_Floods_Guidelines.pdf`, Page 41 (Executive Summary Chapter 6).
- **Actual Retrieved Top-1:** `NDMA_Floods_Guidelines.pdf`, Page 78 (`score=0.7493`), followed by Page 110 (`score=0.7415`).
- **Cause:** Page 41 is a high-level summary, while Page 78 is the substantive full chapter on Flood Plain Zoning. The dense bi-encoder assigned a higher similarity score to the detailed operational text on Page 78 than the introductory summary on Page 41.
- **Fix:** Cross-encoder re-ranking or section-header hierarchical chunking.

### Failure Case 2: Q05 (Semantic Dilution in Large Chunks)
- **Query:** *"What is Rapid Visual Screening (RVS) and what is its role in seismic assessment?"*
- **Expected:** `NDMA_Earthquakes_Guidelines.pdf`, Page 21 (Glossary).
- **Outcome:** **Succeeded in Config 300** (Rank 2, Score: 0.5635), but **Failed in Config 600** (Scores $< 0.41$, fell out of Top-5).
- **Cause:** In 600-token chunks, the concise 40-word RVS definition was merged with multiple unrelated glossary entries (*Resilience*, *Risk*, *Risk Management*). The pooled embedding vector became generic, losing its specific "Rapid Visual Screening" signal.
- **Fix:** BM25 keyword search or hybrid retrieval.

---

## 14. BM25 In-Depth Explanation

### What is BM25?
BM25 (Best Matching 25) is a non-linear probabilistic term-matching ranking algorithm rooted in the Probabilistic Relevance Framework. Given a query $Q = \{q_1, \dots, q_n\}$ and document $D$:

$$\text{BM25}(D, Q) = \sum_{i=1}^n \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### Key Mathematical Components:
1. **Inverse Document Frequency (IDF)**:
   $$\text{IDF}(q_i) = \ln\left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$
   Measures keyword specificity. Common terms appearing across every page (e.g. *"disaster"*, *"management"*) receive near-zero weight. Rare, discriminative terms (e.g. *"RVS"*, *"IS 1893"*, *"Teraflops"*) receive high scores.
2. **Term Frequency Saturation ($k_1 \approx 1.5$)**:
   Unlike raw TF where score scales linearly with repetitions, BM25 uses an asymptotic curve: as term frequency $f(q_i, D)$ increases, marginal score gains diminish, preventing keyword-stuffed documents from dominating.
3. **Document Length Normalization ($b \approx 0.75$)**:
   Penalizes long documents that contain terms merely due to verbosity, normalizing by ratio $\frac{|D|}{\text{avgdl}}$.

### Why BM25 Complements Semantic Dense Search:
- **Bi-Encoder Blind Spots**: Dense embeddings often struggle with exact acronyms (e.g., *"RVS"*, *"NEC"*), alphanumeric codes (e.g., *"IS:1893"*), and rare proper nouns.
- **Complementarity**: Dense models excel at semantic paraphrase (understanding that *"deluges"* relates to *"floods"*), while BM25 excels at exact keyword precision.

---

## 15. Re-Ranker In-Depth Explanation

### What is a Re-Ranker?
A Re-Ranker is a **Cross-Encoder transformer** that takes the query and a candidate document together as a single input sequence:
$$\text{Input} = \text{[CLS]} \circ \text{Query} \circ \text{SEP} \circ \text{Passage} \circ \text{[SEP]}$$

```
Bi-Encoder (Retriever):
  Query ──► [Transformer A] ──► Vector q ──┐
                                           ├──► Dot Product ──► Score
  Doc   ──► [Transformer B] ──► Vector d ──┘
  (Fast O(1) Search via FAISS, but NO token-to-token cross-attention)

Cross-Encoder (Re-Ranker):
  [CLS] Query [SEP] Passage [SEP] ──► [Full Transformer] ──► Non-linear Relevance Score
  (Every query token attends to every passage token simultaneously via Self-Attention)
```

### Computational Trade-Offs
- **Vector Retriever**: Computes document vectors offline once. At query time, inner product takes $< 2$ milliseconds over thousands of chunks.
- **Cross-Encoder Re-Ranker**: Requires a full forward pass through all transformer layers for every $(Query, Passage)$ pair at query time. Evaluating 1,000 documents with a cross-encoder would take several seconds.
- **Two-Stage Architecture**: The optimal production architecture uses the Bi-Encoder / BM25 to retrieve the top 30 candidates ($< 25\text{ms}$), and the Cross-Encoder to re-rank only those 30 candidates ($< 80\text{ms}$).

---

## 16. Installation Instructions

### Prerequisites
- Windows 10/11 (PowerShell)
- Python 3.10 to 3.12
- At least 4 GB free disk space on `D:` or system drive

### Setup
```powershell
# Clone or navigate to repository
cd D:\AI-Assignments\Custom-RAG

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 17. Execution Commands (Windows PowerShell)

All commands are verified and work directly in Windows PowerShell:

```powershell
# 1. Verify and ingest PDF documents into JSONL
python src/document_loader.py --input-dir documents --output-file data/extracted_pages.jsonl

# 2. Generate both 300-token and 600-token chunk datasets
python src/chunking.py --input-file data/extracted_pages.jsonl --output-dir data

# 3. Generate and cache dense vector embeddings
python src/embeddings.py --data-dir data --output-dir embeddings --batch-size 64

# 4. Build and save FAISS and NumPy vector indexes
python src/vector_store.py --data-dir data --embeddings-dir embeddings --indexes-dir indexes

# 5. Run Semantic Vector Retrieval CLI (Config 300 or 600)
python src/retrieval.py --query "What is the 4-stage cyclone warning system?" --config 300 --top-k 5

# 6. Run End-to-End RAG Answer Generation Pipeline
python src/rag_pipeline.py --query "What is the role of the Central Water Commission in flood forecasting?" --config 300

# 7. Run Full Retrieval Evaluation Benchmark (Computes Recall@5 & generates chart)
python src/evaluate.py --questions data/evaluation_questions.json --results-dir results

# 8. Run Hybrid Retrieval Experiment (Dense + BM25 + Reciprocal Rank Fusion)
python src/hybrid_retrieval.py --benchmark --config 300

# 9. Launch Interactive Streamlit Web Interface
streamlit run app/streamlit_app.py
```

---

## 18. Hardware Requirements & Execution Environment

Developed and benchmarked on the target machine:
- **Processor**: AMD Ryzen 5 5600H (6 Cores, 12 Threads, up to 4.2 GHz)
- **Memory**: 16 GB DDR4 RAM (Peak RAM during embedding inference: ~1.4 GB)
- **GPU**: AMD Radeon RX 6500M (Not required; pipeline runs 100% CPU-compatible)
- **Operating System**: Microsoft Windows 11 Home (64-bit)
- **Primary Storage**: `D:\` drive (`D:\AI-Assignments\Custom-RAG`)

---

## 19. System Limitations

1. **Fixed Sliding Window Chunking**: While token-aware, fixed token windows (300/600) occasionally split cohesive tables or multi-part bullet lists across boundaries.
2. **Bi-Encoder Vocabulary Capacity**: MiniLM truncates inputs past 256 tokens; for 600-token chunks, text past token position 256 does not influence the dense embedding vector.
3. **Scanned PDF Limitations**: The ingestion engine relies on digital text layers; documents with low-resolution scanned tables require OCR preprocessing.

---

## 20. Future Architectural Improvements

1. **Contextual Document Header Prepending**: Prepending hierarchical chapter titles (e.g. `[NDMA Floods > Chapter 6: Regulation > 6.1 Flood Plain Zoning]`) to chunk text before embedding generation.
2. **Two-Stage Retrieval Pipeline**: Integrating a dedicated Cross-Encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the top-25 hybrid candidates.
3. **Sentence-Window Retrieval**: Indexing small 1-sentence propositions for vector matching while returning a wider 3-sentence window to the LLM.
