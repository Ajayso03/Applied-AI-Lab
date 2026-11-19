"""
embeddings.py - Embedding Generation and Normalization Module
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Uses sentence-transformers/all-MiniLM-L6-v2 to encode text chunks into
dense 384-dimensional vector embeddings, applies L2 normalization for
exact cosine similarity, and caches output locally to avoid redundant compute.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Force offline mode for local reproducibility
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install via 'pip install sentence-transformers'")


DEFAULT_MODEL_LOCAL_PATH = Path(r"D:\AI-Assignments\Models\HuggingFace\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def compute_file_hash(filepath: Path) -> str:
    """Calculates SHA256 hash of a file for cache invalidation."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class EmbeddingGenerator:
    """
    Manages SentenceTransformer embedding inference, batching, L2 normalization,
    and disk caching.
    """

    def __init__(self, model_name_or_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        target_path = model_name_or_path
        if not target_path:
            if DEFAULT_MODEL_LOCAL_PATH.exists():
                target_path = str(DEFAULT_MODEL_LOCAL_PATH)
            else:
                target_path = FALLBACK_MODEL_NAME

        print(f"Loading embedding model from: {target_path} (device: {self.device})")
        self.model = SentenceTransformer(target_path, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully. Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Encodes list of strings into dense embeddings with L2 normalization.
        Cosine similarity between unit vectors u and v simplifies to their dot product:
            cos(u, v) = (u . v) / (||u|| * ||v||) = u . v
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        raw_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )

        return raw_embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a normalized 1D vector (dim,).
        """
        q_emb = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return q_emb[0].astype(np.float32)


def generate_and_cache_embeddings(
    chunks_path: Path,
    output_npy_path: Path,
    generator: EmbeddingGenerator,
    batch_size: int = 64,
    force_recompute: bool = False
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads chunks from JSONL, checks hash cache, and generates embeddings if needed.
    """
    meta_cache_path = output_npy_path.with_suffix(".meta.json")

    # Load chunks
    chunks = []
    texts = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                chunks.append(item)
                texts.append(item["text"])

    current_hash = compute_file_hash(chunks_path)

    # Check cache validity
    if output_npy_path.exists() and meta_cache_path.exists() and not force_recompute:
        try:
            with open(meta_cache_path, "r", encoding="utf-8") as mf:
                meta_info = json.load(mf)
            if meta_info.get("chunks_hash") == current_hash and meta_info.get("num_chunks") == len(chunks):
                print(f"Cache hit: {output_npy_path.name} is up to date ({len(chunks)} chunks). Skipping generation.")
                embeddings = np.load(output_npy_path)
                return embeddings, chunks
        except Exception as e:
            print(f"Warning: Cache check failed ({e}). Recomputing...")

    print(f"Generating embeddings for {len(texts)} chunks from {chunks_path.name} ...")
    embeddings = generator.embed_texts(texts, batch_size=batch_size)

    # Save to disk
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy_path, embeddings)

    # Save cache metadata
    meta_info = {
        "chunks_file": str(chunks_path.name),
        "chunks_hash": current_hash,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": FALLBACK_MODEL_NAME,
        "normalized": True
    }
    with open(meta_cache_path, "w", encoding="utf-8") as mf:
        json.dump(meta_info, mf, indent=2)

    print(f"Saved embeddings array {embeddings.shape} to: {output_npy_path}")
    print(f"Saved metadata cache to: {meta_cache_path}")

    return embeddings, chunks


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG Embedding Generator")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with chunk JSONL files")
    parser.add_argument("--output-dir", type=str, default="embeddings", help="Directory to store .npy embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--force", action="store_true", help="Force recomputation ignoring cache")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / args.data_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / args.output_dir

    print("==================================================")
    print("DisasterRAG: Vector Embedding Pipeline")
    print(f"Data Directory:       {data_dir}")
    print(f"Output Directory:     {output_dir}")
    print(f"Batch Size:           {args.batch_size}")
    print("==================================================")

    generator = EmbeddingGenerator(device="cpu")

    # 1. Generate embeddings for 300-token chunks
    chunks_300 = data_dir / "chunks_300.jsonl"
    emb_300 = output_dir / "embeddings_300.npy"
    if chunks_300.exists():
        generate_and_cache_embeddings(chunks_300, emb_300, generator, batch_size=args.batch_size, force_recompute=args.force)
    else:
        print(f"Error: {chunks_300} not found. Run chunking.py first.")

    # 2. Generate embeddings for 600-token chunks
    chunks_600 = data_dir / "chunks_600.jsonl"
    emb_600 = output_dir / "embeddings_600.npy"
    if chunks_600.exists():
        generate_and_cache_embeddings(chunks_600, emb_600, generator, batch_size=args.batch_size, force_recompute=args.force)
    else:
        print(f"Error: {chunks_600} not found. Run chunking.py first.")

    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()
