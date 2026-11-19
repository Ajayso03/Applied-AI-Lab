"""
evaluate.py - Retrieval Evaluation Engine & Recall@5 Benchmark
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Evaluates both chunking configurations (300 tokens vs 600 tokens) using
the same 15 ground-truth evaluation questions. Computes Recall@1, Recall@3,
Recall@5, MRR@5, saves per-question retrieval outputs to CSV, and generates
visual comparison plots. No fabricated data.
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

try:
    from src.retrieval import SemanticRetriever, load_retriever_for_config
except ImportError:
    from retrieval import SemanticRetriever, load_retriever_for_config


def judge_relevance(
    retrieved_item: Dict[str, Any],
    expected_doc: str,
    expected_page: int,
    expected_chunk_id: Optional[str] = None,
    ground_truth_passage: str = ""
) -> Tuple[bool, str]:
    """
    Documented Relevance Judgement Methodology:
      A retrieved chunk is considered relevant if:
      1. Exact chunk ID matches the ground-truth chunk ID; OR
      2. Document name matches AND page number matches expected page (or expected page is in page_numbers list); OR
      3. Document name matches AND page is within adjacent page window (+/- 1) with key evidence overlap.
    """
    ret_doc = retrieved_item.get("document_name", "")
    ret_page = retrieved_item.get("page_number", -1)
    ret_pages = retrieved_item.get("page_numbers", [ret_page])
    ret_cid = retrieved_item.get("chunk_id", "")
    ret_text = retrieved_item.get("text", "").lower()

    # Rule 1: Exact chunk ID match
    if expected_chunk_id and ret_cid == expected_chunk_id:
        return True, "exact_chunk_id"

    # Rule 2: Document match + exact page match
    if ret_doc == expected_doc and (ret_page == expected_page or expected_page in ret_pages):
        return True, "exact_doc_and_page"

    # Rule 3: Document match + contiguous page window (+/- 1) AND key textual overlap
    if ret_doc == expected_doc and abs(ret_page - expected_page) <= 1:
        # Check if significant ground truth tokens appear in text
        gt_terms = [t.lower() for t in ground_truth_passage.split() if len(t) > 4]
        overlap = sum(1 for t in gt_terms if t in ret_text)
        if len(gt_terms) > 0 and (overlap / len(gt_terms)) >= 0.35:
            return True, "adjacent_page_with_evidence"

    return False, "not_relevant"


def evaluate_configuration(
    retriever: SemanticRetriever,
    questions: List[Dict[str, Any]],
    config_label: str,
    top_k: int = 5
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Runs full retrieval evaluation over all 15 questions for a given retriever configuration.
    """
    per_question_records = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    latencies = []

    print(f"\n--- Running Evaluation for Configuration: {config_label} (top_k={top_k}) ---")

    for q in questions:
        qid = q["q_id"]
        query_text = q["question"]
        exp_doc = q["expected_document"]
        exp_page = q["expected_page"]
        exp_cid = q.get(f"chunk_id_{config_label.replace('config_', '')}", None)
        gt_passage = q.get("relevant_passage", "")

        t0 = time.perf_counter()
        retrieved = retriever.retrieve(query_text, top_k=top_k)
        duration_ms = (time.perf_counter() - t0) * 1000
        latencies.append(duration_ms)

        first_relevant_rank = None
        relevance_judgements = []

        for item in retrieved:
            is_rel, reason = judge_relevance(item, exp_doc, exp_page, exp_cid, gt_passage)
            relevance_judgements.append({
                "rank": item["rank"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "doc": item["document_name"],
                "page": item["page_number"],
                "is_relevant": is_rel,
                "reason": reason
            })
            if is_rel and first_relevant_rank is None:
                first_relevant_rank = item["rank"]

        # Calculate hits
        has_hit_5 = first_relevant_rank is not None and first_relevant_rank <= 5
        has_hit_3 = first_relevant_rank is not None and first_relevant_rank <= 3
        has_hit_1 = first_relevant_rank is not None and first_relevant_rank == 1

        if has_hit_1:
            hits_at_1 += 1
        if has_hit_3:
            hits_at_3 += 1
        if has_hit_5:
            hits_at_5 += 1

        rr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        top_1 = retrieved[0] if retrieved else {}
        record = {
            "q_id": qid,
            "config": config_label,
            "question": query_text,
            "expected_doc": exp_doc,
            "expected_page": exp_page,
            "recall_at_5_hit": 1 if has_hit_5 else 0,
            "recall_at_3_hit": 1 if has_hit_3 else 0,
            "recall_at_1_hit": 1 if has_hit_1 else 0,
            "first_relevant_rank": first_relevant_rank if first_relevant_rank else "N/A",
            "reciprocal_rank": round(rr, 4),
            "top1_doc": top_1.get("document_name", "N/A"),
            "top1_page": top_1.get("page_number", "N/A"),
            "top1_score": round(top_1.get("score", 0.0), 4),
            "top1_chunk_id": top_1.get("chunk_id", "N/A"),
            "latency_ms": round(duration_ms, 2),
            "relevance_details": relevance_judgements
        }
        per_question_records.append(record)

        status_str = f"HIT (Rank {first_relevant_rank})" if has_hit_5 else "MISS"
        print(f"  Q{qid:02d}: {status_str} | Top-1 Score: {top_1.get('score', 0):.4f} | Latency: {duration_ms:.1f}ms")

    total_q = len(questions)
    recall_at_1 = hits_at_1 / total_q
    recall_at_3 = hits_at_3 / total_q
    recall_at_5 = hits_at_5 / total_q
    mrr_5 = float(np.mean(reciprocal_ranks))
    avg_latency = float(np.mean(latencies))

    summary = {
        "configuration": config_label,
        "total_questions": total_q,
        "hits_at_1": hits_at_1,
        "recall_at_1": round(recall_at_1, 4),
        "hits_at_3": hits_at_3,
        "recall_at_3": round(recall_at_3, 4),
        "hits_at_5": hits_at_5,
        "recall_at_5": round(recall_at_5, 4),
        "mrr_at_5": round(mrr_5, 4),
        "avg_latency_ms": round(avg_latency, 2)
    }

    return summary, per_question_records


def plot_comparison_chart(summary_300: Dict[str, Any], summary_600: Dict[str, Any], output_path: Path):
    """
    Renders comparative bar chart for Recall@1, Recall@3, Recall@5, and MRR@5.
    """
    metrics = ["Recall@1", "Recall@3", "Recall@5", "MRR@5"]
    vals_300 = [
        summary_300["recall_at_1"],
        summary_300["recall_at_3"],
        summary_300["recall_at_5"],
        summary_300["mrr_at_5"]
    ]
    vals_600 = [
        summary_600["recall_at_1"],
        summary_600["recall_at_3"],
        summary_600["recall_at_5"],
        summary_600["mrr_at_5"]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    rects1 = ax.bar(x - width/2, vals_300, width, label="Config A (~300 tokens, 50 overlap)", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, vals_600, width, label="Config B (~600 tokens, 100 overlap)", color="#ff7f0e", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight="bold")
    ax.set_title("DisasterRAG Retrieval Performance: 300 vs 600 Token Chunks (N=15 Questions)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=10)

    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Comparison chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG Retrieval Evaluation Benchmark")
    parser.add_argument("--questions", type=str, default="data/evaluation_questions.json", help="Path to evaluation questions")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for evaluation results")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    q_path = Path(args.questions)
    if not q_path.is_absolute():
        q_path = base_dir / args.questions

    res_dir = Path(args.results_dir)
    if not res_dir.is_absolute():
        res_dir = base_dir / args.results_dir

    if not q_path.exists():
        print(f"Error: Questions file not found at {q_path}")
        sys.exit(1)

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("==================================================")
    print("DisasterRAG: Retrieval Evaluation Pipeline")
    print(f"Total Questions: {len(questions)}")
    print(f"Questions File:  {q_path}")
    print(f"Results Dir:     {res_dir}")
    print("==================================================")

    # 1. Evaluate Configuration A: 300 tokens
    retriever_300 = load_retriever_for_config("300", base_dir=base_dir)
    summary_300, records_300 = evaluate_configuration(retriever_300, questions, "config_300", top_k=5)

    # 2. Evaluate Configuration B: 600 tokens
    retriever_600 = load_retriever_for_config("600", base_dir=base_dir)
    summary_600, records_600 = evaluate_configuration(retriever_600, questions, "config_600", top_k=5)

    # Combine summaries
    comparison_summary = {
        "evaluation_dataset": {
            "file": q_path.name,
            "num_questions": len(questions),
            "documents_evaluated": list(set(q["expected_document"] for q in questions))
        },
        "config_300": summary_300,
        "config_600": summary_600,
        "comparison": {
            "recall_at_5_diff": round(summary_300["recall_at_5"] - summary_600["recall_at_5"], 4),
            "mrr_diff": round(summary_300["mrr_at_5"] - summary_600["mrr_at_5"], 4),
            "faster_config": "config_600" if summary_600["avg_latency_ms"] < summary_300["avg_latency_ms"] else "config_300"
        }
    }

    # Save summary JSON
    json_out = res_dir / "recall_comparison.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nSaved comparison JSON to: {json_out}")

    # Save CSV with question-by-question breakdown
    csv_out = res_dir / "question_results.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question_ID", "Config", "Question", "Expected_Doc", "Expected_Page",
            "Recall@5_Hit", "First_Relevant_Rank", "Reciprocal_Rank",
            "Top1_Doc", "Top1_Page", "Top1_Score", "Latency_ms"
        ])
        for rec in records_300 + records_600:
            writer.writerow([
                rec["q_id"], rec["config"], rec["question"], rec["expected_doc"], rec["expected_page"],
                rec["recall_at_5_hit"], rec["first_relevant_rank"], rec["reciprocal_rank"],
                rec["top1_doc"], rec["top1_page"], rec["top1_score"], rec["latency_ms"]
            ])
    print(f"Saved question results CSV to: {csv_out}")

    # Generate Chart
    png_out = res_dir / "recall_comparison.png"
    plot_comparison_chart(summary_300, summary_600, png_out)

    # Print Terminal Comparison Table
    print("\n==========================================================================")
    print("                    RETRIEVAL EVALUATION SUMMARY TABLE                   ")
    print("==========================================================================")
    print(f"{'Metric':<25} | {'Config A (300 tokens)':<22} | {'Config B (600 tokens)':<22}")
    print("--------------------------------------------------------------------------")
    print(f"{'Total Questions':<25} | {summary_300['total_questions']:<22} | {summary_600['total_questions']:<22}")
    print(f"{'Recall@1':<25} | {summary_300['recall_at_1']*100:.1f}% ({summary_300['hits_at_1']}/{summary_300['total_questions']}){'':<10} | {summary_600['recall_at_1']*100:.1f}% ({summary_600['hits_at_1']}/{summary_600['total_questions']})")
    print(f"{'Recall@3':<25} | {summary_300['recall_at_3']*100:.1f}% ({summary_300['hits_at_3']}/{summary_300['total_questions']}){'':<10} | {summary_600['recall_at_3']*100:.1f}% ({summary_600['hits_at_3']}/{summary_600['total_questions']})")
    print(f"{'Recall@5':<25} | {summary_300['recall_at_5']*100:.1f}% ({summary_300['hits_at_5']}/{summary_300['total_questions']}){'':<10} | {summary_600['recall_at_5']*100:.1f}% ({summary_600['hits_at_5']}/{summary_600['total_questions']})")
    print(f"{'MRR@5':<25} | {summary_300['mrr_at_5']:<22.4f} | {summary_600['mrr_at_5']:<22.4f}")
    print(f"{'Avg Latency (ms)':<25} | {summary_300['avg_latency_ms']:<22.2f} | {summary_600['avg_latency_ms']:<22.2f}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
