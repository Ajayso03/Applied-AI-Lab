"""
document_loader.py - PDF Ingestion and Metadata Preservation Pipeline
Part of DisasterRAG (Custom Retrieval-Augmented Generation Engine).

Extracts text page-by-page from PDF collections using PyMuPDF, applies
deterministic text cleaning, preserves page-level metadata and source URLs,
handles empty/scanned pages gracefully, and exports structured JSONL datasets.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. Install via 'pip install pymupdf'")

# Local text cleaner import
try:
    from src.text_cleaner import TextCleaner
except ImportError:
    from text_cleaner import TextCleaner


DOCUMENT_REGISTRY = {
    "NDMA_Floods_Guidelines.pdf": {
        "title": "National Disaster Management Guidelines: Management of Floods",
        "agency": "National Disaster Management Authority (NDMA), Government of India",
        "url": "https://ndma.gov.in/sites/default/files/PDF/Guidelines/flood.pdf"
    },
    "NDMA_Earthquakes_Guidelines.pdf": {
        "title": "National Disaster Management Guidelines: Management of Earthquakes",
        "agency": "National Disaster Management Authority (NDMA), Government of India",
        "url": "https://ndma.gov.in/sites/default/files/PDF/Guidelines/earthquakes.pdf"
    },
    "NDMA_Cyclones_Guidelines.pdf": {
        "title": "National Disaster Management Guidelines: Management of Cyclones",
        "agency": "National Disaster Management Authority (NDMA), Government of India",
        "url": "https://ndma.gov.in/sites/default/files/PDF/Guidelines/cyclones.pdf"
    },
    "NDMA_National_DM_Policy_2009.pdf": {
        "title": "National Policy on Disaster Management 2009",
        "agency": "Ministry of Home Affairs / NDMA, Government of India",
        "url": "https://ndma.gov.in/sites/default/files/PDF/national-dm-policy2009.pdf"
    }
}


class DocumentLoader:
    """
    Robust document loader using PyMuPDF to extract text and preserve page-level metadata.
    """

    def __init__(self, cleaner: Optional[TextCleaner] = None, min_char_threshold: int = 25):
        self.cleaner = cleaner or TextCleaner()
        self.min_char_threshold = min_char_threshold

    def extract_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts all pages from a single PDF document.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_name = path_obj.name
        metadata_info = DOCUMENT_REGISTRY.get(doc_name, {
            "title": doc_name,
            "agency": "NDMA / Public Disaster Management",
            "url": f"https://ndma.gov.in/sites/default/files/PDF/Guidelines/{doc_name}"
        })
        source_url = metadata_info["url"]

        pages_data = []

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page_num = page_idx + 1  # 1-indexed for human readability
                try:
                    page = doc[page_idx]
                    raw_text = page.get_text("text") or ""
                    cleaned_text = self.cleaner.clean_text(raw_text)

                    is_empty = len(cleaned_text.strip()) < self.min_char_threshold

                    page_record = {
                        "document_name": doc_name,
                        "document_title": metadata_info["title"],
                        "page_number": page_num,
                        "source_url": source_url,
                        "raw_char_count": len(raw_text),
                        "cleaned_char_count": len(cleaned_text),
                        "word_count": len(cleaned_text.split()),
                        "is_empty": is_empty,
                        "text": cleaned_text
                    }
                    pages_data.append(page_record)

                except Exception as page_err:
                    print(f"Warning: Failed to extract page {page_num} of {doc_name}: {page_err}")
                    pages_data.append({
                        "document_name": doc_name,
                        "document_title": metadata_info["title"],
                        "page_number": page_num,
                        "source_url": source_url,
                        "raw_char_count": 0,
                        "cleaned_char_count": 0,
                        "word_count": 0,
                        "is_empty": True,
                        "extraction_error": str(page_err),
                        "text": ""
                    })

            doc.close()

        except Exception as doc_err:
            print(f"Fatal error reading PDF {pdf_path}: {doc_err}")
            raise doc_err

        return pages_data

    def process_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        Scans directory for PDF files and extracts all pages.
        """
        input_path = Path(input_dir)
        pdf_files = sorted(list(input_path.glob("*.pdf")))

        if not pdf_files:
            print(f"No PDF files found in directory: {input_dir}")
            return []

        all_pages = []
        print(f"Found {len(pdf_files)} PDF document(s) in {input_dir}:")
        for pdf_file in pdf_files:
            print(f"  - Ingesting: {pdf_file.name} ...")
            records = self.extract_document(str(pdf_file))
            valid_pages = sum(1 for r in records if not r["is_empty"])
            print(f"    Extracted {len(records)} total pages ({valid_pages} content-bearing pages).")
            all_pages.extend(records)

        return all_pages

    def save_jsonl(self, pages: List[Dict[str, Any]], output_path: str) -> None:
        """
        Saves extracted page records to JSONL file.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            for item in pages:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Successfully saved {len(pages)} page records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DisasterRAG PDF Document Loader & Ingestion Pipeline")
    parser.add_argument("--input-dir", type=str, default="documents", help="Directory containing PDF files")
    parser.add_argument("--output-file", type=str, default="data/extracted_pages.jsonl", help="Output JSONL file path")
    parser.add_argument("--verify-only", action="store_true", help="Verify documents without full export")
    args = parser.parse_args()

    # Determine real path relative to project root or current working dir
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = base_dir / args.input_dir

    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = base_dir / args.output_file

    print("==================================================")
    print("DisasterRAG: Document Ingestion Pipeline")
    print(f"Input Directory:  {input_dir}")
    print(f"Output File:      {output_file}")
    print("==================================================")

    loader = DocumentLoader()
    pages = loader.process_directory(str(input_dir))

    if not pages:
        print("Error: No pages were extracted.")
        sys.exit(1)

    total_pages = len(pages)
    valid_pages = sum(1 for p in pages if not p["is_empty"])
    empty_pages = total_pages - valid_pages
    total_words = sum(p["word_count"] for p in pages)
    total_chars = sum(p["cleaned_char_count"] for p in pages)

    print("\n--- INGESTION SUMMARY ---")
    print(f"Total documents processed: {len(set(p['document_name'] for p in pages))}")
    print(f"Total pages extracted:     {total_pages}")
    print(f"Content-bearing pages:    {valid_pages}")
    print(f"Empty/low-text pages:      {empty_pages}")
    print(f"Total words extracted:     {total_words:,}")
    print(f"Total characters:          {total_chars:,}")
    print(f"Average words/page:        {round(total_words / max(valid_pages, 1), 1)}")

    if total_pages < 30:
        print(f"WARNING: Extracted {total_pages} pages, which is less than the required 30 pages threshold!")
    else:
        print(f"SUCCESS: Page requirement verified ({total_pages} pages >= 30 required).")

    if not args.verify_only:
        loader.save_jsonl(pages, str(output_file))


if __name__ == "__main__":
    main()
