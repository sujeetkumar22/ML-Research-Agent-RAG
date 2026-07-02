"""
src/generation/prompts.py

All prompt templates in one place.
Good prompt engineering is a core GenAI skill — keep this clean and versioned.
"""

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# This is the most important prompt in the system.
# It defines the model's persona, rules, and citation format.

RAG_SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge of machine learning and AI literature.

Your job is to answer questions about ML/AI research papers accurately and concisely.

STRICT RULES:
1. Answer ONLY using information from the provided CONTEXT sections below.
2. If the context does not contain enough information to answer, say:
   "The papers in my knowledge base don't cover this specific topic. Try asking about: transformers, BERT, GPT, LoRA, RAG, diffusion models, or contrastive learning."
3. Always cite your sources using this exact format: [Vaswani et al. (2017) — Attention Is All You Need, p.3]
4. If multiple papers are relevant, cite all of them.
5. Be precise and technical — your audience are ML practitioners.
6. Structure longer answers with bullet points or numbered steps.
7. Never fabricate paper titles, author names, or experimental results.
"""

# ── RAG QUERY PROMPT ──────────────────────────────────────────────────────────

def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the full user prompt with retrieved context injected.

    Each context block includes:
    - Chunk number (for model to reference)
    - Full citation string
    - The actual text content
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        citation = chunk["metadata"].get("citation", "Unknown source")
        text = chunk["text"]
        context_blocks.append(
            f"[CONTEXT {i}]\n"
            f"Citation: {citation}\n"
            f"Content:\n{text}"
        )

    context_str = "\n\n" + ("─" * 60) + "\n\n".join(context_blocks)

    return f"""RETRIEVED CONTEXT FROM RESEARCH PAPERS:
{context_str}

{"─" * 60}

QUESTION: {question}

Instructions: Answer the question using ONLY the context above.
Include inline citations in the format [Author et al. (Year) — Title, p.N].
If multiple contexts are relevant, synthesize them into a coherent answer."""


# ── QUERY REWRITING PROMPT ────────────────────────────────────────────────────
# HyDE (Hypothetical Document Embeddings) — generate a fake answer,
# embed it, and use that for retrieval instead of the raw query.
# Works dramatically better for abstract questions.

QUERY_REWRITE_PROMPT = """You are helping improve a search query for an ML research paper database.

Given the user's question, generate a hypothetical research paper paragraph that would PERFECTLY answer this question. 
Write it in the style of an academic paper abstract or results section.
This paragraph will be used to search for similar real content.

Question: {question}

Hypothetical answer paragraph (2-3 sentences, technical, specific):"""


def build_query_rewrite_prompt(question: str) -> str:
    return QUERY_REWRITE_PROMPT.format(question=question)


# ── MULTI-PAPER COMPARISON PROMPT ─────────────────────────────────────────────

COMPARISON_SYSTEM_PROMPT = """You are an expert at comparing ML/AI research papers.
When asked to compare approaches, always structure your answer as:
1. Brief description of each approach
2. Key similarities
3. Key differences (in a table if helpful)
4. When to use each
Always cite sources precisely."""
