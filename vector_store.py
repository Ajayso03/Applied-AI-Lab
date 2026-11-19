"""
vector_store.py - Vector Index and Similarity Search Engine (NumPy & FAISS)
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Implements from-scratch vector similarity search with explicit mathematical
formulations for both pure NumPy dot-product cosine similarity and C++ FAISS IndexFlatIP.
No third-party RAG framework abstractions are used.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    """
    Dual-engine vector store supporting explicit NumPy matrix algebra and
    high-performance FAISS IndexFlatIP.

    Mathematical Basis:
      Given query vector q in R^D and document embeddings X in R^(N x D):
      Cosine similarity:
        cos(q, x_i) = (q . x_i) / (||q||_2 * ||x_i||_2)
      When ||q||_2 = 1 and ||x_i||_2 = 1 (L2 normalized):
        cos(q, x_i) = q . x_i
      Matrix formulation:
        scores = X @ q in R^N
        top_k = argsort(scores)[::-1][:k]
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape (N, D)
        self.faiss_index = None

    def build_index(self, embeddings: np.ndarray, chunks_metadata: List[Dict[str, Any]]) -> None:
        """
        Initializes store with pre-normalized embeddings and corresponding chunk metadata.
        """
        if len(embeddings) != len(chunks_metadata):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks_metadata)} metadata records.")

        # Ensure float32 and shape
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.chunks_metadata = list(chunks_metadata)
        self.dimension = self.embeddings.shape[1]

        # Verify unit norm (tolerance 1e-4)
        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print("Notice: Embeddings are not unit normalized. Performing L2 normalization in-place...")
            faiss_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(faiss_norms, 1e-12)

        # Build FAISS IndexFlatIP (Inner Product)
        if faiss is not None:
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(self.embeddings)
            print(f"FAISS IndexFlatIP initialized with {self.faiss_index.ntotal} vectors (dim={self.dimension}).")
        else:
            print("FAISS not installed. Using pure NumPy vector store.")

    def search_numpy(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Pure NumPy implementation of cosine similarity search.
        Calculates scores = X @ q, sorts descending, and formats top-k records.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        q = query_embedding.astype(np.float32).flatten()
        # Ensure query is unit normalized
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Matrix-vector multiplication: (N, D) @ (D,) -> (N,)
        scores = np.dot(self.embeddings, q)

        # Efficient partial sort for top_k
        k = min(top_k, len(scores))
        if k == 0:
            return []

        # argsort descending
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks_metadata[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "document_title": chunk.get("document_title", chunk["document_name"]),
                "page_number": chunk["page_number"],
                "page_numbers": chunk.get("page_numbers", [chunk["page_number"]]),
                "source_url": chunk.get("source_url", ""),
                "token_count": chunk.get("token_count", 0),
                "text": chunk["text"],
                "engine": "numpy"
            })

        return results

    def search_faiss(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Accelerated FAISS similarity search.
        """
        if self.faiss_index is None:
            return self.search_numpy(query_embedding, top_k)

        q = query_embedding.astype(np.float32).reshape(1, -1)
        # Normalize
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(q, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            chunk = self.chunks_metadata[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "document_name": chunk["document_name"],
                "document_title": chunk.get("document_title", chunk["document_name"]),
                "page_number": chunk["page_number"],
                "page_numbers": chunk.get("page_numbers", [chunk["page_number"]]),
                "source_url": chunk.get("source_url", ""),
                "token_count": chunk.get("token_count", 0),
                "text": chunk["text"],
                "engine": "faiss"
            })

        return results

    def search(self, query_embedding: np.ndarray, top_k: int = 5, use_faiss: bool = True) -> List[Dict[str, Any]]:
        """
        Unified search endpoint. Defaults to FAISS with fallback to NumPy.
        """
        if use_faiss and self.faiss_index is not None:
            return self.search_faiss(query_embedding, top_k)
        return self.search_numpy(query_embedding, top_k)

    def save(self, index_dir: Path) -> None:
        """
        Persists FAISS index, embeddings array, and metadata to directory.
        """
        index_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save metadata JSON
        meta_path = index_dir / "chunks_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, ensure_ascii=False, indent=2)

        # 2. Save raw embeddings
        np.save(index_dir / "embeddings.npy", self.embeddings)

        # 3. Save FAISS index
        if self.faiss_index is not None and faiss is not None:
            faiss.write_index(self.faiss_index, str(index_dir / "index.faiss"))

        # 4. Save index metadata summary
        summary = {
            "num_vectors": len(self.chunks_metadata),
            "dimension": self.dimension,
            "has_faiss_index": self.faiss_index is not None
        }
        with open(index_dir / "index_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Successfully saved vector store ({len(self.chunks_metadata)} entries) to: {index_dir}")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        """
        Loads vector store from disk.
        """
        meta_path = index_dir / "chunks_metadata.json"
        emb_path = index_dir / "embeddings.npy"
        faiss_path = index_dir / "index.faiss"

        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata in {index_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        store = cls()
        store.chunks_metadata = metadata

        if emb_path.exists():
            store.embeddings = np.load(emb_path)
            store.dimension = store.embeddings.shape[1]

        if faiss_path.exists() and faiss is not None:
            store.faiss_index = faiss.read_index(str(faiss_path))
            store.dimension = store.faiss_index.d

        return store


def build_and_save_index(chunks_jsonl: Path, embeddings_npy: Path, index_dir: Path) -> VectorStore:
    """Helper to build and persist an index from chunks and embeddings."""
    # Load metadata
    chunks = []
    with open(chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Load embeddings
    embeddings = np.load(embeddings_npy)

    store = VectorStore(dimension=embeddings.shape[1])
    store.build_index(embeddings, chunks)
    store.save(index_dir)
    return store


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG Vector Index Builder")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with chunk JSONL files")
    parser.add_argument("--embeddings-dir", type=str, default="embeddings", help="Directory with .npy embeddings")
    parser.add_argument("--indexes-dir", type=str, default="indexes", help="Directory to save vector indexes")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / args.data_dir

    embeddings_dir = Path(args.embeddings_dir)
    if not embeddings_dir.is_absolute():
        embeddings_dir = base_dir / args.embeddings_dir

    indexes_dir = Path(args.indexes_dir)
    if not indexes_dir.is_absolute():
        indexes_dir = base_dir / args.indexes_dir

    print("==================================================")
    print("DisasterRAG: Vector Index Builder")
    print(f"Data Directory:       {data_dir}")
    print(f"Embeddings Directory: {embeddings_dir}")
    print(f"Indexes Directory:    {indexes_dir}")
    print("==================================================")

    # Build Index 300
    print("\nBuilding Index for Configuration A (300 tokens)...")
    build_and_save_index(
        data_dir / "chunks_300.jsonl",
        embeddings_dir / "embeddings_300.npy",
        indexes_dir / "index_300"
    )

    # Build Index 600
    print("\nBuilding Index for Configuration B (600 tokens)...")
    build_and_save_index(
        data_dir / "chunks_600.jsonl",
        embeddings_dir / "embeddings_600.npy",
        indexes_dir / "index_600"
    )

    print("\nVector index build complete.")


if __name__ == "__main__":
    main()
