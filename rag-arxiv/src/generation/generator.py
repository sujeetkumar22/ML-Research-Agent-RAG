"""
src/generation/generator.py

Groq LLM integration for answer generation.
Supports both streaming (for UI) and non-streaming (for evaluation).

Groq is used because:
- Free tier: 14,400 requests/day, 500,000 tokens/min
- Fast: sub-second TTFT (time to first token)
- Llama 3.1 8B is an excellent open model for RAG
"""
import os
from groq import Groq
from loguru import logger
from config import GROQ_API_KEY, LLM_MODEL
from src.generation.prompts import (
    RAG_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    build_rag_prompt,
    build_query_rewrite_prompt,
)


class RAGGenerator:
    """
    Wraps the Groq LLM for RAG answer generation.

    Interview talking points:
    - temperature=0.1: low for factual answers (more deterministic)
    - max_tokens=1024: enough for detailed answers, prevents runaway costs
    - Streaming: users see tokens appear immediately — much better UX
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = LLM_MODEL,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Generator ready — model: {model}")

    # ── MAIN ANSWER GENERATION ─────────────────────────────────────────────

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
        stream: bool = True,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        is_comparison: bool = False,
    ):
        """
        Generate an answer from retrieved chunks.

        If stream=True, yields tokens one by one (use in Streamlit with st.write_stream).
        If stream=False, returns the complete answer string (use in evaluation).
        """
        system_prompt = (
            COMPARISON_SYSTEM_PROMPT if is_comparison else RAG_SYSTEM_PROMPT
        )
        user_prompt = build_rag_prompt(question, retrieved_chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )

            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if stream:
                def error_gen():
                    yield f"⚠️ Generation error: {str(e)}"
                return error_gen()
            return f"⚠️ Generation error: {str(e)}"

    def _stream_response(self, response):
        """Yield tokens from a streaming Groq response."""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── QUERY REWRITING (HyDE) ─────────────────────────────────────────────

    def rewrite_query(self, question: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE):
        Generate a hypothetical answer and use it for retrieval.

        This significantly improves retrieval for abstract questions like:
        "What are the advantages of attention over recurrence?"
        because it gives the retriever a content-rich query instead of
        a short natural language question.
        """
        prompt = build_query_rewrite_prompt(question)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
                stream=False,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.debug(f"Query rewritten: {question[:40]}… → {rewritten[:60]}…")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            return question   # fallback to original query

    # ── FULL RAG PIPELINE ──────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        retriever,
        top_k: int = 5,
        use_hyde: bool = True,
        stream: bool = True,
    ):
        """
        Full end-to-end RAG pipeline:
        1. (Optional) Rewrite query with HyDE
        2. Retrieve relevant chunks
        3. Generate grounded answer

        Returns: (stream_or_answer, retrieved_chunks)
        """
        # Step 1: Query rewriting
        retrieval_query = self.rewrite_query(question) if use_hyde else question

        # Step 2: Retrieve
        chunks = retriever.retrieve(retrieval_query, top_k=top_k)
        if not chunks:
            logger.warning("No chunks retrieved — returning fallback message")
            def _empty():
                yield "I couldn't find relevant information in the research papers. Please try rephrasing your question."
            return (_empty() if stream else "No relevant context found."), []

        # Step 3: Generate
        is_comparison = any(
            word in question.lower()
            for word in ["compare", "difference", "vs", "versus", "better than"]
        )
        answer = self.generate(
            question, chunks,
            stream=stream,
            is_comparison=is_comparison,
        )

        return answer, chunks
