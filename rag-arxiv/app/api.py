from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import HTMLResponse, StreamingResponse
import json

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import RAGGenerator
from config import VECTOR_STORE_PATH, TOP_K

app = FastAPI(title="ArXiv RAG API", description="API for the ArXiv RAG system")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ml-research-agent-rag.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (lazy loaded)
_retriever = None
_generator = None

def get_rag_system():
    global _retriever, _generator
    if _retriever is None or _generator is None:
        try:
            store = VectorStore()
            loaded = store.load(VECTOR_STORE_PATH)
            if not loaded:
                raise RuntimeError("Vector store not found. Please run the build script.")
            _retriever = HybridRetriever(store, store.chunks)
            _generator = RAGGenerator()
        except Exception as e:
            raise RuntimeError(f"Error loading RAG system: {str(e)}")
    return _retriever, _generator


class ChatRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    use_hyde: bool = True
    stream: bool = True

class ChatResponseChunk(BaseModel):
    content: str
    sources: Optional[List[Dict[str, Any]]] = None

@app.on_event("startup")
async def startup_event():
    # Attempt to load on startup, but don't crash if index is currently building
    try:
        get_rag_system()
    except Exception as e:
        print(f"Warning during startup: {e}")

@app.get("/", response_class=HTMLResponse)
def root_endpoint():
    return """
    <html>
        <head>
            <title>ML Research Agent API</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    background-color: #0b0c10;
                    color: #ffffff;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 2.5rem;
                    border-radius: 16px;
                    background-color: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
                    max-width: 480px;
                }
                h1 {
                    color: #8a63f7;
                    margin-bottom: 0.75rem;
                    font-size: 1.8rem;
                }
                p {
                    color: #9ca3af;
                    line-height: 1.6;
                    margin-bottom: 2rem;
                    font-size: 0.95rem;
                }
                .status {
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    padding: 0.4rem 1rem;
                    border-radius: 20px;
                    background-color: rgba(45, 212, 191, 0.1);
                    color: #2dd4bf;
                    font-size: 0.85rem;
                    font-weight: 600;
                }
                .dot {
                    width: 8px;
                    height: 8px;
                    background-color: #2dd4bf;
                    border-radius: 50%;
                    display: inline-block;
                    animation: pulse 1.8s infinite ease-in-out;
                }
                @keyframes pulse {
                    0%, 100% { transform: scale(0.8); opacity: 0.5; }
                    50% { transform: scale(1.2); opacity: 1; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ML Research Agent RAG API</h1>
                <p>The backend API is running successfully on Hugging Face Spaces! Point your Next.js Vercel frontend to this space URL to query papers.</p>
                <div class="status"><span class="dot"></span> API Status: Live</div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        retriever, generator = get_rag_system()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if request.stream:
        # Generate the stream of tokens
        # We need to send sources first, then tokens
        def event_stream():
            try:
                answer_stream, chunks = generator.answer(
                    request.query,
                    retriever,
                    top_k=request.top_k,
                    use_hyde=request.use_hyde,
                    stream=True,
                )
                
                # Send sources as the first message
                sources_data = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"
                
                # Send chunks
                for token in answer_stream:
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
                
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        # Non-streaming is just for fallback if needed
        answer_stream, chunks = generator.answer(
            request.query,
            retriever,
            top_k=request.top_k,
            use_hyde=request.use_hyde,
            stream=False,
        )
        sources_data = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
        return {"answer": "".join(list(answer_stream)), "sources": sources_data}
