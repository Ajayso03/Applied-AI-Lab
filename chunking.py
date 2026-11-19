"""
chunking.py - Tokenizer-Aware Text Chunking Module
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Implements two deterministic chunking configurations:
  - Configuration A: ~300 tokens, 50 tokens overlap
  - Configuration B: ~600 tokens, 100 tokens overlap

Preserves exact chunk metadata (Chunk ID, Document Name, Page Number,
Source URL, Token Count, and Chunk Text) and saves to structured JSONL.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Attempt HuggingFace tokenizer first, fallback to tiktoken or whitespace
TOKENIZER_BACKEND = None
try:
    from transformers import AutoTokenizer
    # Check known local cache paths
    SNAPSHOT_DIR = Path(r"D:\AI-Assignments\Models\HuggingFace\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots")
    if SNAPSHOT_DIR.exists():
        subdirs = [d for d in SNAPSHOT_DIR.iterdir() if d.is_dir()]
        if subdirs:
            HF_TOKENIZER = AutoTokenizer.from_pretrained(str(subdirs[0]), local_files_only=True)
            TOKENIZER_BACKEND = "transformers"
        else:
            HF_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            TOKENIZER_BACKEND = "transformers"
    else:
        HF_TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        TOKENIZER_BACKEND = "transformers"
except Exception as e:
    try:
        import tiktoken
        TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        TOKENIZER_BACKEND = "tiktoken"
    except Exception:
        TOKENIZER_BACKEND = "whitespace"


class TokenizerWrapper:
    """
    Unified tokenizer interface providing accurate tokenization, counting,
    and decoding across HuggingFace, tiktoken, or whitespace fallbacks.
    """

    def __init__(self):
        self.backend = TOKENIZER_BACKEND

    def encode(self, text: str) -> List[int]:
        if self.backend == "transformers":
            return HF_TOKENIZER.encode(text, add_special_tokens=False)
        elif self.backend == "tiktoken":
            return TIKTOKEN_ENC.encode(text)
        else:
            # Approximate tokenization via word/punctuation split
            words = text.split()
            return list(range(len(words)))

    def decode(self, token_ids: List[int]) -> str:
        if self.backend == "transformers":
            return HF_TOKENIZER.decode(token_ids, skip_special_tokens=True)
        elif self.backend == "tiktoken":
            return TIKTOKEN_ENC.decode(token_ids)
        else:
            # Fallback when token_ids represents words
            return " ".join([str(t) for t in token_ids])

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


class ChunkingEngine:
    """
    Engine to segment document page streams into sliding token windows
    while preserving document boundaries and page-level provenance.
    """

    def __init__(self, tokenizer: Optional[TokenizerWrapper] = None):
        self.tokenizer = tokenizer or TokenizerWrapper()

    def chunk_document_pages(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        config_label: str
    ) -> List[Dict[str, Any]]:
        """
        Chunks pages belonging to a single document using a token-aware sliding window.
        Preserves page attribution for every chunk.
        """
        if not pages:
            return []

        doc_name = pages[0]["document_name"]
        doc_title = pages[0].get("document_title", doc_name)
        source_url = pages[0].get("source_url", "")

        # Build an index of token blocks tagged with their origin page number
        token_stream = []
        token_page_map = []  # Maps each token index to page_number

        for p in pages:
            if p.get("is_empty", False):
                continue
            text = p.get("text", "").strip()
            if not text:
                continue
            
            page_tokens = self.tokenizer.encode(text)
            p_num = p["page_number"]
            for tok in page_tokens:
                token_stream.append(tok)
                token_page_map.append(p_num)

        if not token_stream:
            return []

        step_size = chunk_size - chunk_overlap
        if step_size <= 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be strictly greater than chunk_overlap ({chunk_overlap})")

        chunks = []
        total_tokens = len(token_stream)
        chunk_idx = 0

        for start_idx in range(0, total_tokens, step_size):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk_tokens = token_stream[start_idx:end_idx]

            # If the final remnant chunk is tiny (< 35 tokens) and we already have chunks, append or discard
            if len(chunk_tokens) < 35 and chunks:
                break

            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if not chunk_text:
                continue

            # Compute page attribution
            pages_in_chunk = token_page_map[start_idx:end_idx]
            # Primary page is the most frequent page in this chunk
            from collections import Counter
            primary_page = Counter(pages_in_chunk).most_common(1)[0][0]
            unique_pages = sorted(list(set(pages_in_chunk)))

            chunk_id = f"{Path(doc_name).stem}_cfg{chunk_size}_c{chunk_idx+1:04d}"

            chunk_record = {
                "chunk_id": chunk_id,
                "document_name": doc_name,
                "document_title": doc_title,
                "page_number": primary_page,
                "page_numbers": unique_pages,
                "source_url": source_url,
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
                "config": config_label,
                "text": chunk_text
            }
            chunks.append(chunk_record)
            chunk_idx += 1

            if end_idx >= total_tokens:
                break

        return chunks

    def process_all_documents(
        self,
        extracted_pages: List[Dict[str, Any]],
        chunk_size: int,
        chunk_overlap: int,
        config_label: str
    ) -> List[Dict[str, Any]]:
        """
        Groups extracted pages by document and executes chunking per document.
        """
        from collections import defaultdict
        grouped = defaultdict(list)
        for page in extracted_pages:
            grouped[page["document_name"]].append(page)

        all_chunks = []
        for doc_name, pages in sorted(grouped.items()):
            # Sort pages strictly by page_number
            sorted_pages = sorted(pages, key=lambda x: x["page_number"])
            doc_chunks = self.chunk_document_pages(
                sorted_pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                config_label=config_label
            )
            print(f"  [{config_label}] {doc_name}: {len(sorted_pages)} pages -> {len(doc_chunks)} chunks.")
            all_chunks.extend(doc_chunks)

        return all_chunks


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG Chunking Engine (300 & 600 token configurations)")
    parser.add_argument("--input-file", type=str, default="data/extracted_pages.jsonl", help="Input pages JSONL")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for chunk datasets")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        input_file = base_dir / args.input_file

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / args.output_dir

    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist. Run document_loader.py first.")
        sys.exit(1)

    print("==================================================")
    print("DisasterRAG: Text Chunking Pipeline")
    print(f"Tokenizer Backend: {TOKENIZER_BACKEND}")
    print(f"Input Pages File:  {input_file}")
    print(f"Output Directory:  {output_dir}")
    print("==================================================")

    # Load extracted pages
    pages = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))

    print(f"Loaded {len(pages)} extracted page records.")

    engine = ChunkingEngine()

    # Configuration A: 300 tokens, 50 overlap
    print("\nExecuting Configuration A: chunk_size=300, overlap=50 ...")
    chunks_300 = engine.process_all_documents(
        pages,
        chunk_size=300,
        chunk_overlap=50,
        config_label="config_300"
    )
    out_300 = output_dir / "chunks_300.jsonl"
    with open(out_300, "w", encoding="utf-8") as f:
        for c in chunks_300:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks_300)} chunks to {out_300}")

    # Configuration B: 600 tokens, 100 overlap
    print("\nExecuting Configuration B: chunk_size=600, overlap=100 ...")
    chunks_600 = engine.process_all_documents(
        pages,
        chunk_size=600,
        chunk_overlap=100,
        config_label="config_600"
    )
    out_600 = output_dir / "chunks_600.jsonl"
    with open(out_600, "w", encoding="utf-8") as f:
        for c in chunks_600:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks_600)} chunks to {out_600}")

    # Print comparative statistics
    print("\n================ CHUNKING COMPARISON ================")
    print(f"Configuration A (300 tokens, 50 overlap):  Total Chunks = {len(chunks_300):,}")
    avg_tokens_300 = sum(c["token_count"] for c in chunks_300) / max(len(chunks_300), 1)
    print(f"  Average token count: {avg_tokens_300:.1f} tokens")
    
    print(f"Configuration B (600 tokens, 100 overlap): Total Chunks = {len(chunks_600):,}")
    avg_tokens_600 = sum(c["token_count"] for c in chunks_600) / max(len(chunks_600), 1)
    print(f"  Average token count: {avg_tokens_600:.1f} tokens")
    print("=====================================================")


if __name__ == "__main__":
    main()
