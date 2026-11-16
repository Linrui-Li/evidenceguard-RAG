
# EvidenceGuard

An evidence-grounded Retrieval-Augmented Generation (RAG)
system for intelligent document question answering.

## Project Status

Under active development.

## Planned Features

- Document ingestion and text chunking
- Semantic vector search
- Hybrid retrieval and reranking
- Evidence-grounded answer generation
- Source citations and controlled refusal
- Automated RAG evaluation
- Interactive web interface

## Planned Tech Stack

- Backend: Python, FastAPI
- Frontend: React
- Database: PostgreSQL, pgvector
- Embeddings: Sentence Transformers
- LLM: Mistral via Ollama
- Evaluation: RAGAS
- Containerisation: Docker Compose

## Getting Started

Requirements: Python 3.11

Install dependencies:

    pip install -r backend/requirements.txt

Start the backend:

    cd backend
    uvicorn app.main:app --reload

API documentation:

http://127.0.0.1:8000/docs
