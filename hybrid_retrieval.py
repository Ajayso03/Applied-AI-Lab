"""
hybrid_retrieval.py - Hybrid Retrieval Engine (BM25 + Dense + Reciprocal Rank Fusion)
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Implements sparse keyword retrieval (BM25 Okapi) combined with dense vector
retrieval (Sentence Transformers) using Reciprocal Rank Fusion (RRF).
Demonstrates how hybrid retrieval resolves keyword/acronym misses (e.g. RVS, IS 1893).
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank-bm25 is required. Install via 'pip install rank-bm25'")

try:
    from src.retrieval import SemanticRetriever, load_retriever_for_config
    from src.evaluate import judge_relevance
except ImportError:
    from retrieval import SemanticRetriever, load_retriever_for_config
    from evaluate import judge_relevance


def tokenize_for_bm25(text: str) -> List[str]:
    """Simple lowercase alphanumeric word tokenizer for BM25."""
    return re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', text.lower())


class BM25Retriever:
    """
    In-memory sparse keyword retriever using BM25 Okapi algorithm.
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = list(chunks)
        print(f"Tokenizing {len(self.chunks)} chunks for BM25 corpus...")
        self.corpus_tokens = [tokenize_for_bm25(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("BM25 index built successfully.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Scores all chunks against query tokens and returns top_k."""
        q_tokens = tokenize_for_bm25(query)
        if not q_tokens:
            return []

        doc_scores = self.bm25.get_scores(q_tokens)
        k = min(top_k, len(self.chunks))
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(doc_scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "document_title": chunk.get("document_title", chunk["document_name"]),
                "page_number": chunk["page_number"],
                "page_numbers": chunk.get("page_numbers", [chunk["page_number"]]),
                "source_url": chunk.get("source_url", ""),
                "token_count": chunk.get("token_count", 0),
                "text": chunk["text"],
                "engine": "bm25"
            })

        return results


class HybridRetriever:
    """
    Hybrid retriever fusing Dense Vector Search and BM25 Sparse Search
    via Reciprocal Rank Fusion (RRF).

    RRF Formulation:
      RRF_Score(d) = sum_{m in {dense, bm25}} ( 1 / (k_rrf + rank_m(d)) )
      Standard smoothing constant: k_rrf = 60
    """

    def __init__(self, dense_retriever: SemanticRetriever, k_rrf: int = 60):
        self.dense = dense_retriever
        self.k_rrf = k_rrf
        # Build BM25 index over the same chunks metadata
        self.bm25 = BM25Retriever(self.dense.vector_store.chunks_metadata)

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
          1. Fetches top candidate_pool results from Dense retriever.
          2. Fetches top candidate_pool results from BM25 retriever.
          3. Applies Reciprocal Rank Fusion to compute blended rankings.
          4. Returns top-k passages.
        """
        dense_results = self.dense.retrieve(query, top_k=candidate_pool)
        bm25_results = self.bm25.retrieve(query, top_k=candidate_pool)

        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        # 1. Process dense candidates
        for item in dense_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            dense_ranks[cid] = rank
            chunk_lookup[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_rrf + rank))

        # 2. Process BM25 candidates
        for item in bm25_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            bm25_ranks[cid] = rank
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k_rrf + rank))

        # 3. Sort by fused RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[:top_k]

        fused_results = []
        for rank, cid in enumerate(sorted_cids, start=1):
            base_item = dict(chunk_lookup[cid])
            base_item["rank"] = rank
            base_item["score"] = round(rrf_scores[cid], 5)
            base_item["rrf_score"] = round(rrf_scores[cid], 5)
            base_item["dense_rank"] = dense_ranks.get(cid, "N/A")
            base_item["bm25_rank"] = bm25_ranks.get(cid, "N/A")
            base_item["engine"] = "hybrid_rrf"
            fused_results.append(base_item)

        return fused_results


def evaluate_hybrid_vs_dense(config: str = "300", top_k: int = 5) -> Dict[str, Any]:
    """Runs comparative evaluation of Dense vs Hybrid on the 15 evaluation questions."""
    base_dir = Path(__file__).resolve().parent.parent
    q_path = base_dir / "data" / "evaluation_questions.json"
    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    dense_retriever = load_retriever_for_config(config, base_dir=base_dir)
    hybrid_retriever = HybridRetriever(dense_retriever)

    dense_hits = 0
    hybrid_hits = 0
    dense_rr = []
    hybrid_rr = []

    print(f"\n================ HYBRID VS DENSE RETRIEVAL BENCHMARK (Config {config}) ================")
    for q in questions:
        qid = q["q_id"]
        exp_doc = q["expected_document"]
        exp_page = q["expected_page"]
        exp_cid = q.get(f"chunk_id_{config}", None)
        gt_passage = q.get("relevant_passage", "")

        # Dense
        d_res = dense_retriever.retrieve(q["question"], top_k=top_k)
        d_first = next((r["rank"] for r in d_res if judge_relevance(r, exp_doc, exp_page, exp_cid, gt_passage)[0]), None)
        if d_first and d_first <= top_k:
            dense_hits += 1
            dense_rr.append(1.0 / d_first)
        else:
            dense_rr.append(0.0)

        # Hybrid
        h_res = hybrid_retriever.retrieve(q["question"], top_k=top_k)
        h_first = next((r["rank"] for r in h_res if judge_relevance(r, exp_doc, exp_page, exp_cid, gt_passage)[0]), None)
        if h_first and h_first <= top_k:
            hybrid_hits += 1
            hybrid_rr.append(1.0 / h_first)
        else:
            hybrid_rr.append(0.0)

        print(f"Q{qid:02d}: Dense={'HIT (' + str(d_first) + ')' if d_first else 'MISS'} | Hybrid={'HIT (' + str(h_first) + ')' if h_first else 'MISS'}")

    n = len(questions)
    results = {
        "config": config,
        "num_questions": n,
        "dense_recall_at_5": round(dense_hits / n, 4),
        "dense_mrr_at_5": round(float(np.mean(dense_rr)), 4),
        "hybrid_recall_at_5": round(hybrid_hits / n, 4),
        "hybrid_mrr_at_5": round(float(np.mean(hybrid_rr)), 4),
        "delta_recall": round((hybrid_hits - dense_hits) / n, 4),
        "delta_mrr": round(float(np.mean(hybrid_rr) - np.mean(dense_rr)), 4)
    }

    print("\n---------------- COMPARISON RESULTS ----------------")
    print(f"Dense Recall@5:  {results['dense_recall_at_5']*100:.1f}% | Dense MRR@5:  {results['dense_mrr_at_5']:.4f}")
    print(f"Hybrid Recall@5: {results['hybrid_recall_at_5']*100:.1f}% | Hybrid MRR@5: {results['hybrid_mrr_at_5']:.4f}")
    print(f"Recall Gain:     {results['delta_recall']*100:+.1f}% | MRR Gain:     {results['delta_mrr']:+.4f}")
    print("====================================================")

    out_file = base_dir / "results" / "hybrid_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved hybrid benchmark results to: {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG Hybrid Retrieval CLI")
    parser.add_argument("--query", type=str, default="", help="Query to retrieve with hybrid search")
    parser.add_argument("--config", type=str, default="300", choices=["300", "600"], help="Chunk configuration")
    parser.add_argument("--benchmark", action="store_true", help="Run full 15-question benchmark comparing Dense vs Hybrid")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent

    if args.benchmark or not args.query:
        evaluate_hybrid_vs_dense(config=args.config)
    else:
        dense = load_retriever_for_config(args.config, base_dir=base_dir)
        hybrid = HybridRetriever(dense)
        results = hybrid.retrieve(args.query, top_k=5)
        print(f"\n================ HYBRID RETRIEVAL RESULTS: '{args.query}' ================")
        for r in results:
            print(f"[Rank {r['rank']} | RRF Score: {r['score']:.5f}] {r['document_name']} (Page {r['page_number']})")
            print(f"  Dense Rank: {r['dense_rank']} | BM25 Rank: {r['bm25_rank']}")
            print(f"  Preview: {r['text'][:250].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
