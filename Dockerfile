FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/rag-arxiv:$PYTHONPATH

WORKDIR $HOME/app

# Copy requirements files first
COPY --chown=user:user requirements.txt $HOME/app/
COPY --chown=user:user rag-arxiv/requirements.txt $HOME/app/rag-arxiv/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r rag-arxiv/requirements.txt

# Copy the entire project
COPY --chown=user:user . $HOME/app

# Hugging Face Spaces port
EXPOSE 7860

# Set working directory to where the api is located
WORKDIR $HOME/app/rag-arxiv

# Run FastAPI via Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
