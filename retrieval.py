"""
retrieval.py - Semantic Vector Retrieval Engine
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Handles natural-language query ingestion, embedding generation,
vector similarity search (FAISS & NumPy), result ranking, and metadata enrichment.
Completely standalone and decoupled from LLM answer generation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Force offline mode for local reproducibility
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from src.embeddings import EmbeddingGenerator
    from src.vector_store import VectorStore
except ImportError:
    from embeddings import EmbeddingGenerator
    from vector_store import VectorStore


class SemanticRetriever:
    """
    Independent Semantic Retrieval module.
    Encodes query string, searches pre-built vector index, and ranks top passages.
    """

    def __init__(
        self,
        index_dir: Path,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        use_faiss: bool = True
    ):
        self.index_dir = Path(index_dir)
        self.use_faiss = use_faiss
        print(f"Initializing SemanticRetriever from: {self.index_dir}")
        self.vector_store = VectorStore.load(self.index_dir)
        self.generator = embedding_generator or EmbeddingGenerator(device="cpu")
        print(f"Retriever ready. Indexed vectors: {len(self.vector_store.chunks_metadata)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes semantic retrieval:
          1. Encodes query into unit-normalized dense embedding q.
          2. Calculates cosine similarity with indexed chunk vectors X.
          3. Returns top-k passages ranked by descending similarity score.
        """
        query = query.strip()
        if not query:
            return []

        start_time = time.perf_counter()
        q_emb = self.generator.embed_query(query)
        embed_time = (time.perf_counter() - start_time) * 1000

        search_start = time.perf_counter()
        results = self.vector_store.search(q_emb, top_k=top_k, use_faiss=self.use_faiss)
        search_time = (time.perf_counter() - search_start) * 1000

        # Attach retrieval diagnostic metrics to results
        for r in results:
            r["query"] = query
            r["embed_latency_ms"] = round(embed_time, 2)
            r["search_latency_ms"] = round(search_time, 2)
            r["total_latency_ms"] = round(embed_time + search_time, 2)

        return results

    def format_passages(self, results: List[Dict[str, Any]], max_chars: int = 350) -> str:
        """Helper to pretty-print retrieved passages."""
        lines = []
        for r in results:
            doc = r["document_name"]
            page = r["page_number"]
            score = r["score"]
            cid = r["chunk_id"]
            preview = r["text"][:max_chars].replace("\n", " ") + ("..." if len(r["text"]) > max_chars else "")
            lines.append(f"[Rank {r['rank']} | Score: {score:.4f}] {doc} (Page {page}) - ID: {cid}\n  \"{preview}\"\n")
        return "\n".join(lines)


def load_retriever_for_config(config: str = "300", base_dir: Optional[Path] = None) -> SemanticRetriever:
    """Factory function to load retriever for '300' or '600' config."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    idx_folder = f"index_{config}"
    index_path = base_dir / "indexes" / idx_folder
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory does not exist: {index_path}. Build index first.")

    return SemanticRetriever(index_dir=index_path)


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG Semantic Retrieval CLI")
    parser.add_argument("--query", type=str, required=True, help="Natural language query")
    parser.add_argument("--config", type=str, default="300", choices=["300", "600"], help="Chunk configuration (300 or 600)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve")
    parser.add_argument("--numpy-only", action="store_true", help="Force pure NumPy search instead of FAISS")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    index_path = base_dir / "indexes" / f"index_{args.config}"

    retriever = SemanticRetriever(index_dir=index_path, use_faiss=not args.numpy_only)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print("\n==================================================")
    print(f"QUERY: {args.query}")
    print(f"Configuration: {args.config} tokens | Engine: {'NumPy' if args.numpy_only else 'FAISS'}")
    print(f"Top-{args.top_k} Retrieved Passages:")
    print("==================================================\n")
    print(retriever.format_passages(results))


if __name__ == "__main__":
    main()
