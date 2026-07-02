from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.responses import StreamingResponse
import json

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import RAGGenerator
from config import VECTOR_STORE_PATH, TOP_K

app = FastAPI(title="ArXiv RAG API", description="API for the ArXiv RAG system")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
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
