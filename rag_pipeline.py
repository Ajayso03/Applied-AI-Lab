"""
rag_pipeline.py - End-to-End DisasterRAG Pipeline
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Connects document retrieval and grounded answer generation into a unified,
production-grade interface without any external RAG framework dependencies.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from src.retrieval import SemanticRetriever, load_retriever_for_config
    from src.answer_generator import AnswerGenerator
except ImportError:
    from retrieval import SemanticRetriever, load_retriever_for_config
    from answer_generator import AnswerGenerator


class DisasterRAGPipeline:
    """
    Unified DisasterRAG Engine:
      1. Receives natural language user question.
      2. Performs dense vector retrieval over indexed disaster management guidelines.
      3. Passes retrieved top-k passages to AnswerGenerator.
      4. Returns structured grounded response with page-level citations.
    """

    def __init__(self, config: str = "300", llm_backend: str = "auto", base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.retriever = load_retriever_for_config(config=config, base_dir=self.base_dir)
        self.generator = AnswerGenerator(backend=llm_backend)

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Executes full RAG workflow for a user question."""
        total_start = time.perf_counter()

        # Step 1: Retrieval
        retrieval_start = time.perf_counter()
        passages = self.retriever.retrieve(question, top_k=top_k)
        retrieval_duration_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 2: Generation
        gen_start = time.perf_counter()
        gen_output = self.generator.generate_answer(question, passages)
        gen_duration_ms = (time.perf_counter() - gen_start) * 1000

        total_duration_ms = (time.perf_counter() - total_start) * 1000

        return {
            "query": question,
            "answer": gen_output["answer"],
            "citations": gen_output["citations"],
            "backend": gen_output["backend"],
            "retrieved_passages": passages,
            "timings_ms": {
                "retrieval": round(retrieval_duration_ms, 2),
                "generation": round(gen_duration_ms, 2),
                "total": round(total_duration_ms, 2)
            },
            "config": self.config
        }


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG End-to-End Pipeline CLI")
    parser.add_argument("--query", type=str, required=True, help="User query about disaster guidelines")
    parser.add_argument("--config", type=str, default="300", choices=["300", "600"], help="Chunking configuration")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context passages")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "openai", "gemini", "grounded_extractive"])
    args = parser.parse_args()

    pipeline = DisasterRAGPipeline(config=args.config, llm_backend=args.backend)
    result = pipeline.query(args.query, top_k=args.top_k)

    print("\n==================================================")
    print("DisasterRAG: Grounded Response")
    print(f"Query:         {args.query}")
    print(f"Configuration: {args.config} tokens | Backend: {result['backend']}")
    print(f"Latency:       Retrieval: {result['timings_ms']['retrieval']} ms | Total: {result['timings_ms']['total']} ms")
    print("==================================================\n")
    print(result["answer"])
    print("\n--------------------------------------------------")
    print("Top Retrieved Passages:")
    for idx, p in enumerate(result["retrieved_passages"], start=1):
        print(f"[{idx}] {p['document_name']} (Page {p['page_number']}) - Score: {p['score']:.4f}")


if __name__ == "__main__":
    main()
