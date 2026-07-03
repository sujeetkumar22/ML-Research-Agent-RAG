"""
scripts/build_index.py

Run this ONCE to:
1. Download ArXiv papers
2. Load and parse PDFs
3. Chunk text
4. Build + save the vector index

After this, the app loads instantly from the saved index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --skip-download   # if papers already downloaded
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "rag-arxiv" / ".cache" / "huggingface")

from loguru import logger
from src.ingestion.arxiv_downloader import download_papers
from src.ingestion.loader import load_all_papers
from src.ingestion.chunker import chunk_pages, print_chunk_stats
from src.retrieval.vector_store import VectorStore
from config import DATA_RAW, VECTOR_STORE_PATH


def build_index(skip_download: bool = False):
    logger.info("=" * 60)
    logger.info("  RAG ArXiv — Building Index")
    logger.info("=" * 60)

    # ── STEP 1: Download papers ──────────────────────────────────────────────
    if skip_download:
        logger.info("Skipping download (--skip-download flag set)")
    else:
        logger.info("\n📥 STEP 1: Downloading ArXiv papers…")
        papers = download_papers()
        logger.success(f"Downloaded {len(papers)} papers")

    # ── STEP 2: Load PDFs ────────────────────────────────────────────────────
    logger.info("\n📄 STEP 2: Loading and parsing PDFs…")
    pages = load_all_papers(DATA_RAW)

    if not pages:
        logger.error("No pages loaded! Check that PDFs are in data/raw/")
        return False

    logger.success(f"Loaded {len(pages)} pages from PDFs")

    # ── STEP 3: Chunk ────────────────────────────────────────────────────────
    logger.info("\n✂️  STEP 3: Chunking pages…")
    chunks = chunk_pages(pages)
    print_chunk_stats(chunks)

    # ── STEP 4: Build vector index ───────────────────────────────────────────
    logger.info("\n🔢 STEP 4: Building embedding index…")
    logger.info("(This takes 2–5 minutes for 100 papers — runs once only)")

    store = VectorStore()
    store.build(chunks, batch_size=64)

    # ── STEP 5: Save ─────────────────────────────────────────────────────────
    logger.info("\n💾 STEP 5: Saving index to disk…")
    store.save(VECTOR_STORE_PATH)

    logger.success("\n" + "=" * 60)
    logger.success("  ✅ INDEX BUILD COMPLETE!")
    logger.success("=" * 60)
    logger.success(f"  Papers processed: {len({c['metadata'].get('arxiv_id') for c in chunks})}")
    logger.success(f"  Total chunks:     {len(chunks)}")
    logger.success(f"  Index saved to:   {VECTOR_STORE_PATH}.*")
    logger.success("\n  Now run: streamlit run app/streamlit_app.py")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip paper download (use if papers already in data/raw/)",
    )
    args = parser.parse_args()
    build_index(skip_download=args.skip_download)
