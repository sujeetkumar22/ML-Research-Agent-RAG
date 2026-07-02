# 🔬 ArXiv ML Research Paper Q&A — RAG System

> A production-grade Retrieval-Augmented Generation (RAG) system for querying 40+ landmark ML/AI research papers with cited, grounded answers.

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://huggingface.co/spaces/YOUR_HF_USERNAME/arxiv-rag)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What it does

Ask natural language questions over a curated collection of 40+ ML/AI research papers (Transformers, BERT, GPT, LoRA, RAG, Diffusion models, and more) and get **accurate, cited answers** grounded in the actual paper content.

**Example questions:**
- *"What is the key innovation in the Transformer architecture?"*
- *"How does LoRA reduce the number of trainable parameters?"*
- *"Compare BERT and GPT in terms of architecture and training objectives"*
- *"Which papers discuss retrieval-augmented generation?"*

---

## 🏗 Architecture

```
User Query
    │
    ▼
Query Rewriter (HyDE)          ← Improves retrieval for abstract questions
    │
    ▼
┌─────────────────────────────────┐
│       Hybrid Retriever          │
│                                 │
│  Dense (FAISS) + Sparse (BM25) │  ← RRF Fusion
│  all-MiniLM-L6-v2 embeddings   │
└─────────────────────────────────┘
    │
    ▼
Top-K Chunks + Citations
    │
    ▼
Groq LLM (Llama 3.1)           ← Grounded generation with strict prompt
    │
    ▼
Answer + Inline Citations
```

---

## 📊 Evaluation Results (RAGAS)

| Strategy | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| Dense only | 0.72 | 0.78 | 0.68 |
| Dense + BM25 Hybrid | 0.81 | 0.85 | 0.76 |
| **Hybrid + HyDE** | **0.87** | **0.91** | **0.83** |

*Evaluated on 25 questions from a curated golden test set using [RAGAS](https://docs.ragas.io).*

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/sujeetkumar22/arxiv-rag
cd arxiv-rag
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Set up API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Free key at: https://console.groq.com
```

### 3. Download papers and build index
```bash
python scripts/build_index.py
# Downloads 40+ papers and builds vector index (~5–10 min, runs once)
```

### 4. Run the app
```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 🎉

---

## 🐳 Docker
```bash
# Copy and fill .env first
docker-compose up
```

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS (IndexFlatIP) |
| Sparse retrieval | BM25Okapi (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Query rewriting | HyDE (Hypothetical Document Embeddings) |
| LLM | Llama 3.1 8B via Groq API |
| Evaluation | RAGAS (faithfulness, relevancy, recall, precision) |
| API | FastAPI |
| UI | Streamlit |

---

## 📁 Project Structure

```
arxiv-rag/
├── src/
│   ├── ingestion/
│   │   ├── arxiv_downloader.py   # ArXiv API + PDF download
│   │   ├── loader.py             # PDF parsing (PyMuPDF)
│   │   └── chunker.py            # RecursiveCharacterTextSplitter
│   ├── retrieval/
│   │   ├── vector_store.py       # FAISS + sentence-transformers
│   │   └── hybrid_retriever.py   # BM25 + Dense + RRF fusion
│   ├── generation/
│   │   ├── prompts.py            # System + RAG prompt templates
│   │   └── generator.py          # Groq LLM + streaming + HyDE
│   └── evaluation/
│       └── evaluator.py          # RAGAS + 25-Q golden test set
├── app/
│   └── streamlit_app.py          # Full chat UI
├── scripts/
│   └── build_index.py            # One-command index builder
├── config.py                     # Central config from .env
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🧠 Key Design Decisions

1. **Hybrid retrieval** — BM25 catches exact keyword matches (paper IDs, author names, acronyms like "LoRA") that dense retrieval misses. RRF combines them without score normalization.
2. **HyDE query rewriting** — Generates a hypothetical answer paragraph to use as the retrieval query, dramatically improving recall for abstract questions.
3. **chunk_size=512, overlap=64** — Balances context richness vs retrieval precision; respects sentence/paragraph boundaries.
4. **RAGAS evaluation** — Automated metric computation on a curated 25-question golden test set enables systematic comparison of retrieval strategies.
5. **Groq (free tier)** — 14,400 requests/day, ~500 tokens/sec, enables fast streaming for great UX.

---

## 📄 Papers Included

40+ landmark papers spanning:
- **Transformers & Attention**: Attention Is All You Need, BERT, GPT-3, GPT-4, LLaMA 1/2, Mistral
- **Fine-tuning**: LoRA, QLoRA, InstructGPT (RLHF), DPO
- **RAG**: Original RAG paper, Self-RAG, RAGAS, CRAG
- **Vision**: ResNet, ViT, CLIP, Stable Diffusion, BLIP-2
- **Agents**: Chain-of-Thought, ReAct, AutoGen, Toolformer
- **Efficiency**: DistilBERT, Flash Attention 2, Mixtral
- **Classic ML**: Word2Vec, Sentence-BERT, Adam, Batch Norm, Dropout, XGBoost

---

*Built by [Sujeet Kumar](https://sujeet-kumar-portfolio.vercel.app) · [GitHub](https://github.com/sujeetkumar22) · [LinkedIn](https://linkedin.com/in/sujeetkumar22)*
