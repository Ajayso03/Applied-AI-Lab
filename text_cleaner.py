"""
text_cleaner.py - Text Cleaning and Normalization Module
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).
"""

import re
import unicodedata
from typing import Dict, Any, Tuple


class TextCleaner:
    """
    Production-grade text cleaning and normalization for document ingestion in RAG pipelines.
    Preserves semantic structure, technical numbering, and section hierarchies while
    eliminating OCR/PDF extraction noise, excessive whitespace, and line-break artifacts.
    """

    def __init__(self, fix_hyphenation: bool = True, normalize_unicode: bool = True):
        self.fix_hyphenation = fix_hyphenation
        self.normalize_unicode = normalize_unicode

    def clean_text(self, text: str) -> str:
        """
        Applies a pipeline of deterministic cleaning transformations on extracted raw page text.
        """
        if not text:
            return ""

        # 1. Unicode Normalization (converts ligatures like 'fi', 'fl', accents to standard forms)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Normalize Windows/Mac line endings to standard Unix \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Fix hyphenated line breaks (e.g. "disas-\nter" -> "disaster")
        if self.fix_hyphenation:
            # Matches word character followed by hyphen, optional space, newline, and word character
            text = re.sub(r'(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)', r'\1\2', text)

        # 4. Remove form feeds, null bytes, and non-printable control characters (except newline, tab)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 5. Fix common repetitive headers/footers in Indian government documents (e.g., standalone page numbers)
        # e.g., "\n\n12\n\n" or "Page 12 of 168"
        text = re.sub(r'\n+\s*(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*\n+', '\n\n', text, flags=re.IGNORECASE)

        # 6. Normalize multiple horizontal whitespaces (spaces, tabs) to a single space
        text = re.sub(r'[ \t]+', ' ', text)

        # 7. Collapse more than two consecutive newlines into a standard paragraph break (\n\n)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 8. Strip trailing and leading whitespace
        cleaned = text.strip()

        return cleaned

    def compute_stats(self, raw_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Computes extraction and cleaning diagnostic statistics.
        """
        raw_words = len(raw_text.split())
        cleaned_words = len(cleaned_text.split())
        raw_chars = len(raw_text)
        cleaned_chars = len(cleaned_text)

        compression_ratio = (cleaned_chars / raw_chars) if raw_chars > 0 else 0.0

        return {
            "raw_characters": raw_chars,
            "cleaned_characters": cleaned_chars,
            "raw_words": raw_words,
            "cleaned_words": cleaned_words,
            "compression_ratio": round(compression_ratio, 4)
        }


def clean_page_text(text: str) -> str:
    """Convenience function for page-level cleaning."""
    cleaner = TextCleaner()
    return cleaner.clean_text(text)
