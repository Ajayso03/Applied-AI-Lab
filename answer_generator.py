"""
answer_generator.py - Grounded Answer Generation Module
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Formats retrieved passages into a strict contextual prompt, enforces
anti-hallucination guardrails, supports external LLM APIs (OpenAI/Gemini/Groq)
via environment variables, and provides a deterministic grounded extractive
synthesis fallback for offline execution without API keys.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional


RAG_SYSTEM_PROMPT = """You are DisasterRAG, an authoritative AI assistant specializing in public disaster-management guidelines.
Your goal is to provide accurate, grounded answers based EXCLUSIVELY on the provided official context passages.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the provided context passages. Do NOT invent, extrapolate, or introduce external knowledge.
2. If the context passages do not contain enough information to answer the question, state:
   "The provided documents do not contain sufficient information to answer this question."
3. Every factual statement must cite its source document and page number using: [Doc: <filename>, Page: <page_number>].
4. At the end of your answer, provide a dedicated "Citations" section listing the unique source documents, page numbers, and original URLs."""


def build_context_block(passages: List[Dict[str, Any]]) -> str:
    """Formats retrieved passages into a clean contextual block for LLM prompts."""
    blocks = []
    for idx, p in enumerate(passages, start=1):
        doc = p.get("document_name", "Unknown Document")
        page = p.get("page_number", "N/A")
        cid = p.get("chunk_id", f"c{idx}")
        score = p.get("score", 0.0)
        url = p.get("source_url", "")
        text = p.get("text", "").strip()

        header = f"[Passage {idx}] Document: {doc} | Page: {page} | Chunk ID: {cid} | Similarity: {score:.4f}"
        if url:
            header += f" | URL: {url}"
        blocks.append(f"{header}\n{text}\n")

    return "\n".join(blocks)


def build_rag_prompt(query: str, passages: List[Dict[str, Any]]) -> str:
    """Constructs complete prompt string with context and query."""
    context_text = build_context_block(passages)
    prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"=================== CONTEXT PASSAGES ===================\n"
        f"{context_text}\n"
        f"========================================================\n\n"
        f"USER QUESTION: {query}\n\n"
        f"GROUNDED ANSWER:"
    )
    return prompt


class AnswerGenerator:
    """
    Decoupled Answer Generation engine.
    Supports external API keys (OpenAI / Gemini / Groq) and offline fallback.
    """

    def __init__(self, backend: str = "auto", model_name: Optional[str] = None):
        self.backend = backend
        self.model_name = model_name
        self.client = None
        self._init_backend()

    def _init_backend(self):
        """Auto-detects available LLM credentials or falls back to local grounded synthesis."""
        if self.backend in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
            try:
                import openai
                self.client = openai.OpenAI()
                self.backend = "openai"
                self.model_name = self.model_name or "gpt-4o-mini"
                print(f"AnswerGenerator: Using OpenAI API backend ({self.model_name}).")
                return
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")

        if self.backend in ("auto", "gemini") and os.environ.get("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ["GEMINI_API_KEY"])
                self.backend = "gemini"
                self.model_name = self.model_name or "gemini-1.5-flash"
                print(f"AnswerGenerator: Using Gemini API backend ({self.model_name}).")
                return
            except Exception as e:
                print(f"Failed to initialize Gemini client: {e}")

        # Fallback to local deterministic grounded synthesis
        self.backend = "grounded_extractive"
        print("AnswerGenerator: Operating in Grounded Extractive mode (CPU-friendly, 100% offline).")

    def _extractive_synthesis(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """
        Deterministic, offline grounded answer synthesis.
        Extracts top-ranking sentences containing query keywords, ensures strict factual
        grounding directly from the official passages, and formats citations.
        """
        if not passages:
            return "The provided documents do not contain sufficient information to answer this question.\n\nCitations: None."

        query_terms = set(re.findall(r'\b[A-Za-z]{3,}\b', query.lower()))
        stopwords = {"what", "which", "when", "where", "how", "why", "who", "are", "the", "and", "for", "with", "from", "that", "this", "they", "does", "been", "have", "were"}
        filtered_terms = query_terms - stopwords

        supported_sentences = []
        cited_sources = {}

        for p in passages:
            doc = p["document_name"]
            page = p["page_number"]
            url = p.get("source_url", "")
            text = p["text"]

            # Split text into sentences
            raw_sents = re.split(r'(?<=[.!?])\s+', text)
            for s in raw_sents:
                s_clean = s.strip()
                if len(s_clean) < 30:
                    continue
                s_lower = s_clean.lower()
                # Count matching terms
                matches = sum(1 for term in filtered_terms if term in s_lower)
                if matches >= 1:
                    score = matches * (p["score"])
                    supported_sentences.append({
                        "sentence": s_clean,
                        "doc": doc,
                        "page": page,
                        "url": url,
                        "score": score
                    })
                    cited_sources[(doc, page)] = url

        if not supported_sentences:
            return (
                "The provided documents do not contain sufficient information to answer this question.\n\n"
                "Citations: None."
            )

        # Sort sentences by relevance score and deduplicate
        supported_sentences.sort(key=lambda x: x["score"], reverse=True)
        seen_stems = set()
        chosen = []
        for item in supported_sentences:
            stem = item["sentence"][:50].lower()
            if stem not in seen_stems:
                seen_stems.add(stem)
                chosen.append(item)
            if len(chosen) >= 4:
                break

        # Assemble grounded response
        ans_lines = ["Based on the official disaster management guidelines:"]
        for c in chosen:
            ans_lines.append(f"• {c['sentence']} [Doc: {c['doc']}, Page: {c['page']}]")

        ans_lines.append("\nCitations:")
        for (doc, page), url in sorted(cited_sources.items()):
            url_str = f" ({url})" if url else ""
            ans_lines.append(f"- {doc}, Page {page}{url_str}")

        return "\n".join(ans_lines)

    def generate_answer(self, query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates grounded answer from query and retrieved passages.
        Returns dictionary containing prompt, answer, citations, and backend info.
        """
        prompt = build_rag_prompt(query, passages)

        if self.backend == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{build_context_block(passages)}\n\nQuestion: {query}"}
                    ],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI generation error: {e}. Falling back to extractive synthesis.")
                answer = self._extractive_synthesis(query, passages)

        elif self.backend == "gemini":
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel(self.model_name)
                resp = model.generate_content(prompt)
                answer = resp.text.strip()
            except Exception as e:
                print(f"Gemini generation error: {e}. Falling back to extractive synthesis.")
                answer = self._extractive_synthesis(query, passages)

        else:
            answer = self._extractive_synthesis(query, passages)

        # Extract structured citations
        citations = []
        for p in passages:
            citations.append({
                "document_name": p["document_name"],
                "document_title": p.get("document_title"),
                "page_number": p["page_number"],
                "source_url": p.get("source_url"),
                "chunk_id": p.get("chunk_id"),
                "similarity_score": p.get("score")
            })

        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "backend": self.backend,
            "prompt": prompt
        }
