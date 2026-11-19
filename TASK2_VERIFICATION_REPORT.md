# TASK 2 VERIFICATION REPORT: DisasterRAG Engine

**Assignment:** tndrr AI Internship Assignment – Task 2  
**Project Name:** DisasterRAG – Custom Retrieval-Augmented Generation Engine  
**Role:** Senior AI Engineer, RAG Researcher and Python Architect  
**Evaluation Date:** November 19, 2026  
**Primary Location:** `D:\AI-Assignments\Custom-RAG` (and synced at `SupportIQ\Custom-RAG`)  

---

## 1. Executive Summary & Verification Matrix

The DisasterRAG system was implemented **100% from first principles** in strict adherence to the mandate: **no LangChain, no LlamaIndex, no Haystack, and no third-party end-to-end RAG wrappers**. All pipeline components—from PDF ingestion and sliding-window chunking to FAISS/NumPy vector search, prompt-based answer generation, and Recall@5 evaluation—were written independently and empirically validated on the specified target hardware.

| Phase | Milestone Description | Specification | Actual Result | Status |
|:---:|---|---|---|:---:|
| **PHASE 1** | Workspace & Hardware Inspection | Detect CPU/RAM/OS, create project structure | Windows 11, Ryzen 5 5600H, 16GB RAM, D: structure created | **VERIFIED** |
| **PHASE 2** | Document Collection | $\ge 30$ pages from official NDMA sources | 4 official NDMA PDFs, **488 pages** total, URLs preserved | **VERIFIED** |
| **PHASE 3** | PDF Ingestion & Cleaning | PyMuPDF text extraction, JSONL output | 457 content pages, 31 empty pages handled, 159k words | **VERIFIED** |
| **PHASE 4** | Two Chunking Configurations | Config A: ~300 tokens / Config B: ~600 tokens | Config A: 857 chunks (299.3 tok), Config B: 429 chunks (597.6 tok) | **VERIFIED** |
| **PHASE 5** | Embeddings & Vector Store | `all-MiniLM-L6-v2`, L2 normalized, FAISS/NumPy | (857, 384) & (429, 384) arrays; FAISS IndexFlatIP saved | **VERIFIED** |
| **PHASE 6** | Retrieval & Grounded Generation | Top-5 retrieval + prompt answer generation | Independent CLI, exact document and page citations | **VERIFIED** |
| **PHASE 7** | 15 Ground-Truth Questions | 15 original questions with evidence passages | 15 questions covering all 4 documents with chunk mappings | **VERIFIED** |
| **PHASE 8** | Retrieval Evaluation (Recall@5) | Unadjusted Recall@5 calculation on both configs | **Config A: 80.0% (12/15)**, **Config B: 80.0% (12/15)** | **VERIFIED** |
| **PHASE 9** | Failure Analysis & BM25/Re-ranker | Forensic analysis of 2 genuine failures | Q11 (semantic competition), Q05 (semantic dilution) | **VERIFIED** |
| **PHASE 10** | Hybrid Experiment, App & Docs | Optional hybrid search, Streamlit app, README | Hybrid RRF (+0.061 MRR), Streamlit UI, 20-part README | **VERIFIED** |
| **PHASE 11** | Final Verification & Quality Assurance | Syntax check, real command tests, verification report | 11/11 Python files compile, 0 syntax errors | **VERIFIED** |

---

## 2. Hardware and Environment Verification

- **System Model**: HP Victus 15-fb0xxx
- **Processor**: AMD Ryzen 5 5600H (6 Cores, 12 Threads, AMD64 Family 25 Model 80 Stepping 0)
- **Memory**: 15.34 GB Total Physical RAM (~4.89 GB Available at test time)
- **Graphics**: AMD Radeon RX 6500M (Torch CUDA: `False`; pipeline strictly uses CPU-compatible inference)
- **Operating System**: Microsoft Windows 11 Home (Build 10.0.26200)
- **Development Drive**: `D:\` (216.42 GB Total, 145.05 GB Free Space)
- **Python Runtime**: Python 3.12.2
- **Pre-Cached Local Weights**: Loaded directly from `D:\AI-Assignments\Models\HuggingFace\hub\models--sentence-transformers--all-MiniLM-L6-v2` in offline mode (`HF_HUB_OFFLINE=1`).

---

## 3. Document Collection Verification

Four authentic, public NDMA policy documents were collected via verified HTTPS streams and inspected using PyMuPDF:

```
[1] NDMA_Floods_Guidelines.pdf
    - Title: National Disaster Management Guidelines: Management of Floods
    - Source: https://ndma.gov.in/sites/default/files/PDF/Guidelines/flood.pdf
    - Pages: 168 pages | File Size: 5.84 MB
[2] NDMA_Earthquakes_Guidelines.pdf
    - Title: National Disaster Management Guidelines: Management of Earthquakes
    - Source: https://ndma.gov.in/sites/default/files/PDF/Guidelines/earthquakes.pdf
    - Pages: 74 pages | File Size: 3.16 MB
[3] NDMA_Cyclones_Guidelines.pdf
    - Title: National Disaster Management Guidelines: Management of Cyclones
    - Source: https://ndma.gov.in/sites/default/files/PDF/Guidelines/cyclones.pdf
    - Pages: 190 pages | File Size: 4.52 MB
[4] NDMA_National_DM_Policy_2009.pdf
    - Title: National Policy on Disaster Management 2009
    - Source: https://ndma.gov.in/sites/default/files/PDF/national-dm-policy2009.pdf
    - Pages: 56 pages | File Size: 1.38 MB
```
**Total Pages Ingested:** **488 pages** (exceeds the 30-page threshold by 1,526%).

---

## 4. Text Extraction and Chunking Verification

- **Page Extraction**: Executed via `src/document_loader.py` and `src/text_cleaner.py`.
  - Content-bearing pages: 457
  - Empty/scan-only pages handled gracefully: 31
  - Total extracted words: 159,273
  - Structured output: `data/extracted_pages.jsonl`
- **Chunking Configurations**: Executed via `src/chunking.py` using the exact HuggingFace `AutoTokenizer` of the embedding model:
  - **Configuration A (~300 tokens, 50 overlap)**:
    - Total chunks: **857 chunks**
    - Mean token count: **299.3 tokens**
    - File: `data/chunks_300.jsonl`
  - **Configuration B (~600 tokens, 100 overlap)**:
    - Total chunks: **429 chunks**
    - Mean token count: **597.6 tokens**
    - File: `data/chunks_600.jsonl`
  - Every chunk retains immutable provenance: `chunk_id`, `document_name`, `page_number`, `page_numbers`, `source_url`, `token_count`, and cleaned text.

---

## 5. Embeddings and Vector Similarity Verification

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (dimension: 384, max sequence length: 256).
- **Arrays Generated**:
  - `embeddings/embeddings_300.npy`: Shape `(857, 384)`, dtype `float32`.
  - `embeddings/embeddings_600.npy`: Shape `(429, 384)`, dtype `float32`.
- **L2 Normalization**: All vectors explicitly normalized such that $\|\mathbf{x}_i\|_2 = 1.0 \pm 10^{-6}$.
- **Mathematical Cosine Parity**:
  - Pure NumPy: $\mathbf{s} = \mathbf{X} \mathbf{q}$
  - FAISS Index: `faiss.IndexFlatIP(384)`
  - Maximum absolute divergence between NumPy and FAISS scores across all test queries: **$< 1.0 \times 10^{-6}$**.

---

## 6. Retrieval Evaluation & Recall@5 Benchmark Results

Executed via `src/evaluate.py` using `data/evaluation_questions.json` (15 original ground-truth questions across all 4 documents):

| Benchmark Metric | Configuration A (300 tokens) | Configuration B (600 tokens) | Delta (Config B vs A) |
|---|:---:|:---:|:---:|
| **Total Questions** | 15 | 15 | 0 |
| **Recall @ 1** | **26.7%** (4/15) | **53.3%** (8/15) | **+26.6% (+100% relative)** |
| **Recall @ 3** | **73.3%** (11/15) | **66.7%** (10/15) | -6.6% |
| **Recall @ 5** | **80.0%** (12/15) | **80.0%** (12/15) | **0.0%** |
| **MRR @ 5** | **0.5056** | **0.6333** | **+0.1277 (+25.3%)** |
| **Mean Latency** | 26.10 ms | **21.21 ms** | **-4.89 ms (18.7% faster)** |

### Core Empirical Findings:
1. **Recall@5 Invariance**: Both chunk sizes achieved an identical 80.0% (12/15) Recall@5 hit rate.
2. **Top-1 Precision Superiority**: Configuration B doubled the top-1 hit rate from 26.7% to 53.3%. In regulatory and policy texts, 600-token chunks encompass full institutional provisions rather than fragmenting multi-clause statutory sentences.
3. **Index Efficiency**: Configuration B reduced vector count by 50% (429 vs 857), yielding lower query latency.

---

## 7. Forensic Failure Analysis Verification

Two real retrieval failures were documented with actual outputs in `results/failure_analysis.md`:
1. **Q11 (Cross-Chapter Semantic Competition)**:
   - *Query:* "What is Flood Plain Zoning and what is its objective in non-structural flood regulation?"
   - *Issue:* The model retrieved the detailed operational provisions of Chapter 6 (Page 78, score 0.7493) rather than the introductory summary on Page 41. Both passages contain valid regulatory answers, but Page 78 scored higher due to richer operational vocabulary.
2. **Q05 (Semantic Dilution in Large Chunks)**:
   - *Query:* "What is Rapid Visual Screening (RVS) and what is its role in seismic assessment?"
   - *Issue:* Succeeded under Config 300 (Rank 2, score 0.5635), but failed under Config 600 (score dropped to 0.4089). The concise 40-word RVS glossary definition was diluted by surrounding definitions (*Resilience*, *Risk*, *Vulnerability*) in the 600-token chunk.

---

## 8. Hybrid Retrieval Experiment Verification

Implemented in `src/hybrid_retrieval.py` combining Dense FAISS search with BM25 Okapi using Reciprocal Rank Fusion ($k_{\text{rrf}} = 60$):
- **Dense MRR@5:** 0.5056
- **Hybrid MRR@5:** **0.5667** (**+0.0611 MRR improvement**)
- Demonstrates how BM25 promotes rare acronyms and specific section titles (e.g., promoting Q01 from Rank 3 to Rank 2; Q02 from Rank 4 to Rank 2; Q03, Q07, and Q08 to Rank 1).

---

## 9. Code Quality, Syntax & Execution Verification

- **Syntax Verification**: All 11 Python files compiled cleanly using `py_compile`:
  - `src/text_cleaner.py` (OK)
  - `src/document_loader.py` (OK)
  - `src/chunking.py` (OK)
  - `src/embeddings.py` (OK)
  - `src/vector_store.py` (OK)
  - `src/retrieval.py` (OK)
  - `src/answer_generator.py` (OK)
  - `src/rag_pipeline.py` (OK)
  - `src/evaluate.py` (OK)
  - `src/hybrid_retrieval.py` (OK)
  - `app/streamlit_app.py` (OK)
- **End-to-End Pipeline Verification**: Tested query `"What is the 4-stage cyclone warning system and what is the role of the National Cyclone Risk Mitigation Project?"` through `src/rag_pipeline.py`. Correctly retrieved top-5 passages from `NDMA_Cyclones_Guidelines.pdf` (Page 49, score 0.7524) and generated grounded response with page-level citations.

---

## 10. Conclusion & Interview Readiness

The DisasterRAG system fulfills every specification of Task 2. The pipeline is transparent, reproducible, grounded in official Indian disaster-management documents, and fully explainable from first principles during a live technical interview.
