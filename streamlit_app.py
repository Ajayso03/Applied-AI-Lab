"""
streamlit_app.py - Interactive DisasterRAG Web Interface
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Provides an interactive user interface to explore document ingestion,
switch between 300-token and 600-token vector indexes, test dense vs hybrid
retrieval, query official disaster management guidelines, and inspect citations.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Set page config
st.set_page_config(
    page_title="DisasterRAG – Official Guidelines Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F3F4F6;
        border-left: 4px solid #2563EB;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        border-radius: 4px;
    }
    .metric-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_engine(config: str, use_hybrid: bool = False):
    """Loads and caches retriever and generator pipelines."""
    from src.retrieval import load_retriever_for_config
    from src.answer_generator import AnswerGenerator
    from src.hybrid_retrieval import HybridRetriever

    retriever = load_retriever_for_config(config, base_dir=PROJECT_ROOT)
    if use_hybrid:
        active_retriever = HybridRetriever(retriever)
    else:
        active_retriever = retriever

    generator = AnswerGenerator(backend="auto")
    return active_retriever, generator


@st.cache_data
def load_evaluation_data():
    """Loads evaluation dataset and summary results."""
    eval_path = PROJECT_ROOT / "data" / "evaluation_questions.json"
    results_path = PROJECT_ROOT / "results" / "recall_comparison.json"
    
    questions = []
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

    results = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    return questions, results


# Sidebar controls
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/200px-Emblem_of_India.svg.png", width=70)
st.sidebar.title("DisasterRAG Configuration")
st.sidebar.markdown("**Engine Settings**")

config_choice = st.sidebar.selectbox(
    "Chunk Configuration",
    ["300", "600"],
    format_func=lambda x: f"Config {x} (~{x} tokens, {'50' if x=='300' else '100'} overlap)"
)

retrieval_mode = st.sidebar.radio(
    "Retrieval Engine",
    ["Dense Vector (FAISS / Cosine)", "Hybrid (BM25 + Dense RRF)"]
)
is_hybrid = "Hybrid" in retrieval_mode

top_k = st.sidebar.slider("Top Passages (k)", min_value=1, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Document Manifest")
st.sidebar.caption("• NDMA Management of Floods (168 pp)\n• NDMA Management of Earthquakes (74 pp)\n• NDMA Management of Cyclones (190 pp)\n• National DM Policy 2009 (56 pp)\n**Total: 488 Pages**")


# Main UI
st.markdown('<div class="main-title">DisasterRAG: Retrieval-Augmented Generation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Authoritative Question-Answering over Public National Disaster Management Authority (NDMA) Guidelines</div>', unsafe_allow_html=True)

tab_qa, tab_eval, tab_docs = st.tabs(["💬 Ask Guidelines", "📊 Benchmark & Evaluation", "📚 Document Collection"])

eval_questions, eval_summary = load_evaluation_data()

with tab_qa:
    st.markdown("#### Enter your question or select from pre-defined evaluation benchmarks:")
    
    preset_options = ["-- Select a benchmark question --"] + [f"Q{q['q_id']:02d}: {q['question']}" for q in eval_questions]
    selected_preset = st.selectbox("Benchmark Presets", preset_options)

    default_query = ""
    if selected_preset != "-- Select a benchmark question --":
        qid = int(selected_preset.split(":")[0][1:])
        q_obj = next((q for q in eval_questions if q["q_id"] == qid), None)
        if q_obj:
            default_query = q_obj["question"]

    user_query = st.text_input("Disaster Management Query", value=default_query, placeholder="e.g. What are the key structural measures for flood management?")

    if st.button("Search & Generate Answer", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Retrieving relevant passages and synthesizing grounded response..."):
                t_start = time.perf_counter()
                retriever, generator = load_rag_engine(config_choice, use_hybrid=is_hybrid)
                
                # Retrieve
                t_ret = time.perf_counter()
                passages = retriever.retrieve(user_query, top_k=top_k)
                ret_latency = (time.perf_counter() - t_ret) * 1000

                # Generate
                t_gen = time.perf_counter()
                gen_res = generator.generate_answer(user_query, passages)
                gen_latency = (time.perf_counter() - t_gen) * 1000
                total_latency = (time.perf_counter() - t_start) * 1000

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Retrieval Latency", f"{ret_latency:.1f} ms")
            col_b.metric("Generation Latency", f"{gen_latency:.1f} ms")
            col_c.metric("Total Latency", f"{total_latency:.1f} ms")

            st.markdown("### Grounded Answer")
            st.markdown(gen_res["answer"])

            st.markdown("---")
            st.markdown("### Retrieved Evidence Passages (Top-5)")
            for p in passages:
                score_label = f"RRF Score: {p['score']:.5f}" if is_hybrid else f"Cosine Similarity: {p['score']:.4f}"
                with st.expander(f"Rank {p['rank']} | {p['document_name']} (Page {p['page_number']}) | {score_label}"):
                    st.markdown(f"**Document Title:** {p.get('document_title', p['document_name'])}")
                    st.markdown(f"**Source URL:** [{p.get('source_url', '')}]({p.get('source_url', '')})")
                    st.markdown(f"**Chunk ID:** `{p.get('chunk_id')}` | **Token Count:** {p.get('token_count', 'N/A')}")
                    st.info(p["text"])


with tab_eval:
    st.markdown("### Retrieval Evaluation & Recall@5 Benchmark")
    st.markdown("Evaluated across **15 original ground-truth questions** over all 488 pages using **sentence-transformers/all-MiniLM-L6-v2**.")

    if eval_summary:
        s300 = eval_summary.get("config_300", {})
        s600 = eval_summary.get("config_600", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Config A (300) Recall@5", f"{s300.get('recall_at_5', 0)*100:.1f}%", f"{s300.get('hits_at_5')}/{s300.get('total_questions')} Hits")
        col2.metric("Config B (600) Recall@5", f"{s600.get('recall_at_5', 0)*100:.1f}%", f"{s600.get('hits_at_5')}/{s600.get('total_questions')} Hits")
        col3.metric("Config A MRR@5", f"{s300.get('mrr_at_5', 0):.4f}")
        col4.metric("Config B MRR@5", f"{s600.get('mrr_at_5', 0):.4f}")

        chart_path = PROJECT_ROOT / "results" / "recall_comparison.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="Retrieval Performance Comparison (Recall@1, Recall@3, Recall@5, MRR@5)", use_column_width=True)

    st.markdown("### Ground-Truth Evaluation Dataset (15 Questions)")
    st.dataframe(
        [
            {
                "QID": q["q_id"],
                "Question": q["question"],
                "Expected Document": q["expected_document"],
                "Page": q["expected_page"],
                "Ground Truth Answer": q["ground_truth_answer"]
            }
            for q in eval_questions
        ],
        use_container_width=True
    )


with tab_docs:
    st.markdown("### Document Manifest and Provenance")
    st.markdown("""
    | # | Document File Name | Official Document Title | Pages | Source URL |
    |---|---|---|:---:|---|
    | 1 | `NDMA_Floods_Guidelines.pdf` | Management of Floods | 168 | [ndma.gov.in Guidelines](https://ndma.gov.in/sites/default/files/PDF/Guidelines/flood.pdf) |
    | 2 | `NDMA_Earthquakes_Guidelines.pdf` | Management of Earthquakes | 74 | [ndma.gov.in Guidelines](https://ndma.gov.in/sites/default/files/PDF/Guidelines/earthquakes.pdf) |
    | 3 | `NDMA_Cyclones_Guidelines.pdf` | Management of Cyclones | 190 | [ndma.gov.in Guidelines](https://ndma.gov.in/sites/default/files/PDF/Guidelines/cyclones.pdf) |
    | 4 | `NDMA_National_DM_Policy_2009.pdf` | National Policy on Disaster Management 2009 | 56 | [ndma.gov.in Policy](https://ndma.gov.in/sites/default/files/PDF/national-dm-policy2009.pdf) |
    """)
    st.info("Total Document Pages Ingested: **488 pages** (requirement: >= 30 pages). Extracted into structured JSONL format preserving document name, page number, character count, and canonical URLs.")
