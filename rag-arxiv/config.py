"""
config.py — Single source of truth for all settings.
Load this everywhere instead of reading .env directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────
ROOT_DIR         = Path(__file__).parent
DATA_RAW         = ROOT_DIR / "data" / "raw"
DATA_PROCESSED   = ROOT_DIR / "data" / "processed"
VECTOR_STORE_PATH = str(DATA_PROCESSED / "vector_store")

# Create dirs if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── LLM ─────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ── Embeddings ───────────────────────────────────
EMBED_MODEL  = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Retrieval ────────────────────────────────────
TOP_K        = int(os.getenv("TOP_K", 5))

# ── Chunking ─────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

# ── ArXiv download settings ───────────────────────
ARXIV_MAX_PAPERS = 100
ARXIV_CATEGORIES = [
    "cs.LG",   # Machine Learning
    "cs.AI",   # Artificial Intelligence
    "cs.CL",   # Computation and Language (NLP)
    "cs.CV",   # Computer Vision
    "stat.ML", # Statistics - Machine Learning
]
