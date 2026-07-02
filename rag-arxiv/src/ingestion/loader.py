"""
src/ingestion/loader.py

Loads PDFs downloaded from ArXiv and extracts clean text + rich metadata.
Uses PyMuPDF (fitz) for high-quality extraction.
"""
import fitz          # PyMuPDF
import json
import re
from pathlib import Path
from loguru import logger
from config import DATA_RAW


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    PDFs often have hyphenation, weird whitespace, and artifacts.
    """
    # Fix hyphenated line breaks (e.g., "trans-\nformer" → "transformer")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form feeds and other control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_sections(text: str) -> dict[str, str]:
    """
    Attempt to extract major paper sections (Abstract, Introduction, etc.)
    Returns a dict mapping section_name → text content.
    """
    # Common section headings in ML papers
    section_pattern = re.compile(
        r"^(Abstract|Introduction|Related Work|Background|Method(?:ology)?|"
        r"Approach|Model|Experiment(?:s)?|Results?|Discussion|"
        r"Conclusion(?:s)?|References?|Appendix)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    sections = {}
    matches = list(section_pattern.finditer(text))

    for i, match in enumerate(matches):
        section_name = match.group(1).title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 100:   # skip near-empty sections
            sections[section_name] = section_text

    return sections


def load_pdf(pdf_path: str, paper_metadata: dict = None) -> list[dict]:
    """
    Load a single PDF and return a list of page dicts.

    Each dict:
        {
            "text":     str,      # cleaned page text
            "metadata": {
                "source":       str,   # filename
                "arxiv_id":     str,
                "title":        str,
                "authors":      list,
                "published":    str,
                "url":          str,
                "page":         int,
                "total_pages":  int,
                "section":      str,   # best-guess section name
            }
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.warning(f"File not found: {pdf_path}")
        return []

    meta_base = paper_metadata or {}

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        logger.error(f"Cannot open {pdf_path}: {e}")
        return []

    pages = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        # Skip pages with very little content (cover pages, blank pages)
        if len(text.strip()) < 100:
            continue

        # Skip reference-only pages (mostly citations)
        if text.count("[") > 20 and len(text) < 1000:
            continue

        pages.append({
            "text": text,
            "metadata": {
                "source":      path.name,
                "arxiv_id":    meta_base.get("arxiv_id", ""),
                "title":       meta_base.get("title", path.stem),
                "authors":     meta_base.get("authors", []),
                "published":   meta_base.get("published", ""),
                "url":         meta_base.get("url", ""),
                "page":        page_num + 1,
                "total_pages": total_pages,
                "section":     "",   # filled in by chunker
            },
        })

    doc.close()
    logger.debug(f"Loaded {len(pages)} pages from {path.name}")
    return pages


def load_all_papers(data_dir: Path = DATA_RAW) -> list[dict]:
    """
    Load ALL downloaded papers from the data/raw directory.
    Uses papers_metadata.json to enrich each page with paper metadata.

    Returns flat list of page dicts across all papers.
    """
    metadata_path = data_dir / "papers_metadata.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            papers_meta = {p["arxiv_id"]: p for p in json.load(f)}
    else:
        papers_meta = {}
        logger.warning("No metadata file found — loading PDFs without paper metadata")

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {data_dir}")
        return []

    logger.info(f"Loading {len(pdf_files)} PDFs...")

    all_pages = []
    for pdf_path in pdf_files:
        # Try to match to metadata by filename
        arxiv_id = pdf_path.name.split("_")[0]
        meta = papers_meta.get(arxiv_id, {})
        pages = load_pdf(str(pdf_path), meta)
        all_pages.extend(pages)

    logger.success(f"Loaded {len(all_pages)} pages from {len(pdf_files)} papers")
    return all_pages
