# Applied-AI-Lab – Applied AI & LLM Engineering Projects

A collection of three applied AI engineering projects developed as part of the Tndrr AI Internship Assignment.

The repository demonstrates practical implementation and evaluation of customer support automation, Retrieval-Augmented Generation (RAG), and Large Language Model (LLM) optimisation.

## Project Overview

| Task | Project | Core Focus |
|---|---|---|
| Task 1 | SupportIQ | Customer support classification using LoRA fine-tuning |
| Task 2 | DisasterRAG | Document-grounded question answering using RAG |
| Task 3 | LLM Optimisation | Quantisation, inference benchmarking and efficient LLM serving |

---

# Task 1 – SupportIQ

**Customer Support Ticket Classification using LoRA Fine-Tuning**

SupportIQ is a customer support triage system that classifies incoming customer messages across three dimensions:

- Category – identifies the type of customer issue.
- Priority – determines the urgency of the issue.
- Sentiment – identifies the customer's emotional tone.

### Technologies

- Python
- Qwen2.5-0.5B-Instruct
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Pandas and Scikit-learn

### Important Files

| File | Purpose |
|---|---|
| `train.py` | LoRA fine-tuning pipeline |
| `inference.py` | Runs model inference on support tickets |
| `evaluate.py` | Evaluates classification performance |
| `generate_data.py` | Generates or prepares training data |
| `validate_data.py` | Validates dataset structure and labels |
| `memory_estimate.py` | Estimates training memory requirements |
| `train.jsonl` | Training dataset |
| `validation.jsonl` | Validation dataset |
| `test.jsonl` | Test dataset |
| `predictions.csv` | Model predictions and evaluation comparisons |
| `evaluation.json` | Evaluation metrics |
| `failure_analysis.md` | Analysis of classification errors |
| `priority_errors.csv` | Priority classification errors |
| `priority_errors_detailed.csv` | Detailed priority error analysis |
| `combined_priority_audit.csv` | Combined priority review candidates |
| `manual_priority_review.csv` | Manual priority review records |
| `TASK1_VERIFICATION_REPORT.md` | Task 1 verification report |

### Evaluation Results

| Metric | Result |
|---|---:|
| Category accuracy | 94.89% |
| Priority accuracy | 68.61% |
| Sentiment accuracy | 91.24% |
| Mean field accuracy | 84.91% |
| Exact-match accuracy | 59.12% |
| JSON parsing success | 100% |

The results demonstrate strong category and sentiment classification, while priority prediction remains an area for improvement.

---

# Task 2 – DisasterRAG

**Retrieval-Augmented Generation for Disaster Management Guidelines**

DisasterRAG is a document-grounded question-answering system designed to retrieve relevant information from disaster management guidelines and provide context-aware responses.

The implementation uses NDMA documents as its knowledge source.

### Technologies

- Python
- Retrieval-Augmented Generation
- Text embeddings
- FAISS / NumPy
- Hybrid retrieval
- Streamlit

### Important Source Files

| File | Purpose |
|---|---|
| `document_loader.py` | Loads and extracts source documents |
| `text_cleaner.py` | Cleans extracted document text |
| `chunking.py` | Splits documents into retrieval chunks |
| `embeddings.py` | Generates vector embeddings |
| `vector_store.py` | Manages vector storage and similarity search |
| `retrieval.py` | Performs document retrieval |
| `hybrid_retrieval.py` | Implements hybrid retrieval |
| `answer_generator.py` | Generates answers using retrieved context |
| `rag_pipeline.py` | Connects the RAG processing stages |
| `streamlit_app.py` | Provides the interactive application |

### Important Data and Evaluation Files

| File | Purpose |
|---|---|
| `extracted_pages.jsonl` | Extracted document page content |
| `chunks_300.jsonl` | Document chunks using a 300-token configuration |
| `chunks_600.jsonl` | Document chunks using a 600-token configuration |
| `embeddings_300.npy` | Embeddings for the 300-token chunks |
| `embeddings_600.npy` | Embeddings for the 600-token chunks |
| `evaluation_questions.json` | Evaluation questions |
| `question_results.csv` | Question-level evaluation results |
| `recall_comparison.json` | Retrieval recall comparison |
| `hybrid_comparison.json` | Hybrid retrieval comparison |
| `recall_comparison.png` | Retrieval performance visualisation |
| `TASK2_VERIFICATION_REPORT.md` | Task 2 verification report |

### Evaluation Results

| Metric | Result |
|---|---:|
| Source documents | 4 |
| Total document pages | 488 |
| Evaluation questions | 15 |
| Recall@5 | 80% |

The project also includes retrieval failure analysis and comparisons of retrieval configurations.

---

# Task 3 – LLM Optimisation

**Local Model Inference and Quantisation Optimisation**

This project explores practical techniques for improving local LLM inference efficiency while balancing model quality, storage requirements, memory consumption and generation speed.

The experiments compare two GGUF quantisation configurations of Qwen2.5-1.5B-Instruct.

### Technologies

- Python
- Qwen2.5-1.5B-Instruct
- GGUF
- llama.cpp
- Q4_K_M and Q8_0 quantisation
- CPU/Vulkan inference
- Hardware and memory monitoring

### Important Source Files

| File | Purpose |
|---|---|
| `hardware_detection.py` | Detects CPU, GPU and system hardware |
| `model_manager.py` | Manages model acquisition and verification |
| `inference.py` | Runs local LLM inference |
| `memory_monitor.py` | Tracks memory consumption |
| `quality_evaluation.py` | Evaluates model output quality |
| `benchmark.py` | Executes performance benchmarks |
| `benchmark_analysis.ipynb` | Interactive benchmark analysis |

### Configuration and Evaluation Files

| File | Purpose |
|---|---|
| `benchmark_prompts.json` | Ten model evaluation prompts |
| `hardware_profile.json` | Hardware profile and system information |
| `benchmark_results.csv` | Tabular benchmark results |
| `benchmark_results.json` | Detailed benchmark measurements |
| `quality_comparison.csv` | Model output and quality comparison |
| `quality_comparison.json` | Structured quality evaluation |
| `requirements.txt` | Python dependencies |

### Reports and Visualisations

| File | Purpose |
|---|---|
| `benchmark_analysis.md` | Performance and memory analysis |
| `kv_cache_explanation.md` | KV-cache and quantisation explanation |
| `concurrent_serving_plan.md` | Proposed architecture for 20 concurrent users |
| `performance_comparison.png` | Throughput comparison |
| `memory_comparison.png` | Memory comparison |
| `quality_comparison.png` | Quality comparison |
| `TASK3_VERIFICATION_REPORT.md` | Task 3 verification report |

### Recorded Benchmark Results

| Metric | Q4_K_M | Q8_0 |
|---|---:|---:|
| Model size | 1.041 GB | 1.764 GB |
| Generation speed | 26.57 tok/s | 18.23 tok/s |
| Prompt processing | 90.89 tok/s | 45.81 tok/s |
| Average latency | 11.78 s | 16.34 s |
| Peak RAM | 1.65 GB | 1.68 GB |
| Quality score | 9.20/10 | 8.95/10 |

These are recorded experimental results. The quality scores are based on the project's evaluation rubric and should not be interpreted as general-purpose model-quality scores.

### Optimisation Areas

1. Quantisation and model-size reduction.
2. Inference throughput and latency measurement.
3. Memory monitoring and KV-cache analysis.
4. Quality comparison across quantisation configurations.
5. A proposed 20-user serving architecture on a 24-GB GPU.

The concurrent-serving architecture is a design proposal, not a claim of completed 20-user load testing.

---

# Repository Structure

The intended final repository organisation is:

```text
RAG-TAG/
│
├── README.md
├── .gitignore
│
├── Task-1-SupportIQ/
│   ├── src/
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── evaluate.py
│   │   ├── generate_data.py
│   │   ├── validate_data.py
│   │   └── memory_estimate.py
│   ├── data/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── outputs/
│   ├── results/
│   ├── reports/
│   └── README.md
│
├── Task-2-DisasterRAG/
│   ├── src/
│   │   ├── document_loader.py
│   │   ├── text_cleaner.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retrieval.py
│   │   ├── hybrid_retrieval.py
│   │   ├── answer_generator.py
│   │   └── rag_pipeline.py
│   ├── app/
│   │   └── streamlit_app.py
│   ├── data/
│   ├── results/
│   ├── reports/
│   └── README.md
│
└── Task-3-LLM-Optimisation/
    ├── src/
    │   ├── hardware_detection.py
    │   ├── model_manager.py
    │   ├── inference.py
    │   ├── memory_monitor.py
    │   ├── quality_evaluation.py
    │   └── benchmark.py
    ├── data/
    │   └── benchmark_prompts.json
    ├── notebooks/
    │   └── benchmark_analysis.ipynb
    ├── results/
    ├── reports/
    ├── tools/
    ├── requirements.txt
    └── README.md
```

*This is the target structure. File paths should be updated after the repository has been reorganised.*

---

# Installation and Usage

## Clone the Repository

```bash
git clone https://github.com/Ajayso03/RAG-TAG.git
cd RAG-TAG
```

## Navigate to a Project

For SupportIQ:

```bash
cd Task-1-SupportIQ
```

For DisasterRAG:

```bash
cd Task-2-DisasterRAG
```

For LLM Optimisation:

```bash
cd Task-3-LLM-Optimisation
```

## Install Dependencies

Each task should maintain its own `requirements.txt` to avoid dependency conflicts.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install the task-specific dependencies:

```bash
pip install -r requirements.txt
```

Refer to each task's README for exact model downloads, dataset preparation, environment configuration and execution commands.

---

# Reproducibility and Project Integrity

- Evaluation datasets and test splits should be preserved.
- Benchmark results should be traceable to their scripts and configurations.
- Large model weights and temporary checkpoints should be excluded from Git when appropriate.
- API keys, credentials and private configuration files must never be committed.
- Generated Python cache files (`__pycache__` and `.pyc`) should be excluded.
- Measured results must be distinguished from proposed deployment designs.

---

# Key Learning Outcomes

These projects demonstrate practical experience in:

- Fine-tuning language models using LoRA.
- Building and evaluating classification systems.
- Designing document-grounded RAG pipelines.
- Implementing embedding-based and hybrid retrieval.
- Comparing LLM quantisation configurations.
- Benchmarking inference throughput, latency and memory.
- Analysing KV-cache requirements.
- Designing scalable LLM serving architectures.

---

# Author

**Ajay Soren**

B.Tech – Computer Science and Engineering

- GitHub: https://github.com/Ajayso03
- Repository: https://github.com/Ajayso03/RAG-TAG

---

*Applied AI and LLM engineering projects focused on implementation, evaluation, optimisation and reproducibility.*
