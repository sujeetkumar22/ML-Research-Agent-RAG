"""
src/ingestion/chunker.py

Splits loaded pages into smaller chunks suitable for embedding.
Each chunk carries full metadata for citation.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split pages into overlapping chunks.

    Returns list of chunk dicts:
        {
            "text":     str,
            "metadata": { ...all page metadata..., "chunk_index": int }
        }

    Design choices (explain these in interviews!):
    - chunk_size=512 tokens: balances context richness vs retrieval precision
    - chunk_overlap=64: ensures context isn't lost at chunk boundaries
    - separators ordered by preference: paragraph > sentence > word > char
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    skipped = 0

    for page in pages:
        text = page["text"]
        page_meta = page["metadata"]

        # Skip if page is too short to be useful
        if len(text.strip()) < 50:
            skipped += 1
            continue

        splits = splitter.split_text(text)

        for i, split_text in enumerate(splits):
            # Skip tiny chunks (often page numbers, headers)
            if len(split_text.strip()) < 40:
                continue

            chunks.append({
                "text": split_text.strip(),
                "metadata": {
                    **page_meta,
                    "chunk_index": i,
                    "chunk_total": len(splits),
                    # Human-readable citation string
                    "citation": _build_citation(page_meta, i),
                },
            })

    logger.success(
        f"Created {len(chunks)} chunks from {len(pages)} pages "
        f"(skipped {skipped} short pages)"
    )
    return chunks


def _build_citation(meta: dict, chunk_idx: int) -> str:
    """Build a human-readable citation string for a chunk."""
    title = meta.get("title", "Unknown")
    authors = meta.get("authors", [])
    first_author = authors[0].split()[-1] if authors else "Unknown"  # last name
    published = meta.get("published", "")
    year = published[:4] if published else "?"
    page = meta.get("page", "?")
    arxiv_id = meta.get("arxiv_id", "")

    # Format: "Vaswani et al. (2017) — Attention Is All You Need, p.3"
    author_str = f"{first_author} et al." if len(authors) > 1 else first_author
    title_short = title[:50] + "…" if len(title) > 50 else title
    return f"{author_str} ({year}) — {title_short}, p.{page}"


def print_chunk_stats(chunks: list[dict]) -> None:
    """Print useful stats about your chunks — good for debugging."""
    if not chunks:
        logger.warning("No chunks to analyze")
        return

    lengths = [len(c["text"]) for c in chunks]
    papers = len({c["metadata"]["arxiv_id"] for c in chunks if c["metadata"].get("arxiv_id")})

    print(f"\n{'='*50}")
    print(f"  CHUNK STATISTICS")
    print(f"{'='*50}")
    print(f"  Total chunks:       {len(chunks)}")
    print(f"  Unique papers:      {papers}")
    print(f"  Avg chunk length:   {sum(lengths) // len(lengths)} chars")
    print(f"  Min chunk length:   {min(lengths)} chars")
    print(f"  Max chunk length:   {max(lengths)} chars")
    print(f"{'='*50}\n")
