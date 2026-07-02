"""
src/ingestion/arxiv_downloader.py

Downloads ML/AI papers from ArXiv and saves PDFs + metadata.
Uses the official arxiv Python client.

Usage:
    python -m src.ingestion.arxiv_downloader
"""
import arxiv
import json
import time
from pathlib import Path
from loguru import logger
from config import DATA_RAW, ARXIV_MAX_PAPERS


# ── Curated list of landmark + recent ML papers ──────────────────────────────
# These are the paper IDs (from arxiv.org/abs/XXXX.XXXXX) we'll download.
# Mix of foundational + modern papers across NLP, CV, RL, GenAI.

LANDMARK_PAPER_IDS = [
    # ── Transformers & Attention ────────────────────────
    "1706.03762",   # Attention Is All You Need (Transformer)
    "1810.04805",   # BERT
    "2005.14165",   # GPT-3
    "2204.02311",   # GPT-4 Technical Report
    "2302.13971",   # LLaMA
    "2307.09288",   # LLaMA 2
    "2309.12307",   # Mistral 7B

    # ── Fine-tuning & Alignment ──────────────────────────
    "2106.09685",   # LoRA: Low-Rank Adaptation
    "2305.14314",   # QLoRA
    "2203.02155",   # InstructGPT (RLHF)
    "2305.18290",   # Direct Preference Optimization (DPO)
    "2210.11610",   # Scaling Instruction-Finetuned Language Models

    # ── RAG & Retrieval ──────────────────────────────────
    "2005.11401",   # RAG: Retrieval-Augmented Generation
    "2212.10560",   # Self-RAG
    "2310.11511",   # RAGAS: Automated RAG Evaluation
    "2312.10997",   # Corrective RAG (CRAG)

    # ── Embeddings & Representation ─────────────────────
    "1301.3666",    # Word2Vec
    "1408.5882",    # Sentence Embeddings (Skip-Thought)
    "1908.10084",   # Sentence-BERT (SBERT)
    "2201.10005",   # text-embedding-ada-002 / E5

    # ── Computer Vision ──────────────────────────────────
    "1512.03385",   # ResNet
    "2010.11929",   # Vision Transformer (ViT)
    "2112.10752",   # Stable Diffusion (LDM)
    "2301.13188",   # BLIP-2

    # ── Agents & Reasoning ──────────────────────────────
    "2201.11903",   # Chain-of-Thought Prompting
    "2303.11366",   # ReAct: Reasoning + Acting
    "2308.08155",   # AutoGen
    "2210.03629",   # Toolformer

    # ── MLOps & Evaluation ──────────────────────────────
    "2209.07753",   # Holistic Evaluation of Language Models (HELM)
    "2306.05685",   # BIG-Bench Hard
    "1803.09010",   # Datasheets for Datasets

    # ── Diffusion & Generative ───────────────────────────
    "2006.11239",   # DDPM: Denoising Diffusion
    "2207.12598",   # Classifier-Free Guidance
    "2103.00020",   # CLIP

    # ── Efficiency & Compression ─────────────────────────
    "1910.01108",   # DistilBERT
    "2004.11981",   # ALBERT
    "2306.07929",   # Flash Attention 2
    "2309.06180",   # Mistral MoE (Mixtral)

    # ── Classic ML ───────────────────────────────────────
    "1603.02754",   # XGBoost
    "1211.5063",    # Dropout
    "1502.03167",   # Batch Normalization
    "1412.6980",    # Adam optimizer
]


def download_papers(
    paper_ids: list[str] = LANDMARK_PAPER_IDS,
    output_dir: Path = DATA_RAW,
    delay: float = 3.0,
) -> list[dict]:
    """
    Download PDFs and metadata for a list of ArXiv paper IDs.

    Returns list of metadata dicts for successfully downloaded papers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "papers_metadata.json"

    # Load existing metadata if resuming
    if metadata_path.exists():
        with open(metadata_path) as f:
            existing = json.load(f)
        existing_ids = {p["arxiv_id"] for p in existing}
        logger.info(f"Resuming: {len(existing)} papers already downloaded")
    else:
        existing = []
        existing_ids = set()

    client = arxiv.Client(
        page_size=10,
        delay_seconds=delay,
        num_retries=3,
    )

    downloaded = list(existing)
    to_download = [pid for pid in paper_ids if pid not in existing_ids]
    logger.info(f"Papers to download: {len(to_download)}")

    for i, paper_id in enumerate(to_download):
        try:
            search = arxiv.Search(id_list=[paper_id])
            results = list(client.results(search))

            if not results:
                logger.warning(f"[{i+1}/{len(to_download)}] Not found: {paper_id}")
                continue

            paper = results[0]
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_"
                for c in paper.title
            )[:80]
            filename = f"{paper_id}_{safe_title}.pdf"
            filepath = output_dir / filename

            if not filepath.exists():
                paper.download_pdf(dirpath=str(output_dir), filename=filename)
                logger.success(f"[{i+1}/{len(to_download)}] ✓ {paper.title[:60]}…")
            else:
                logger.info(f"[{i+1}/{len(to_download)}] Already exists: {filename}")

            meta = {
                "arxiv_id":   paper_id,
                "title":      paper.title,
                "authors":    [str(a) for a in paper.authors[:5]],
                "abstract":   paper.summary,
                "published":  str(paper.published.date()),
                "categories": paper.categories,
                "pdf_path":   str(filepath),
                "filename":   filename,
                "url":        f"https://arxiv.org/abs/{paper_id}",
            }
            downloaded.append(meta)

            # Save metadata incrementally (safe if interrupted)
            with open(metadata_path, "w") as f:
                json.dump(downloaded, f, indent=2)

            time.sleep(delay)

        except Exception as e:
            logger.error(f"Failed {paper_id}: {e}")
            continue

    logger.success(f"\n✅ Done! {len(downloaded)} papers ready in {output_dir}")
    return downloaded


def search_and_download(
    query: str,
    max_results: int = 20,
    output_dir: Path = DATA_RAW,
) -> list[dict]:
    """
    Search ArXiv by keyword and download results.
    Use this to add topic-specific papers beyond the curated list.

    Example:
        search_and_download("mixture of experts language models", max_results=10)
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    paper_ids = [r.entry_id.split("/")[-1] for r in client.results(search)]
    logger.info(f"Found {len(paper_ids)} papers for query: '{query}'")
    return download_papers(paper_ids, output_dir)


if __name__ == "__main__":
    logger.info("Starting ArXiv paper download...")
    papers = download_papers()
    logger.success(f"Downloaded {len(papers)} papers total")
