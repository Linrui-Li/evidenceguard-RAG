
from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="EvidenceGuard API",
    description="An evidence-grounded RAG system",
    version="0.1.0"
)


@app.get("/")
def root():
    return {"message": "Welcome to EvidenceGuard"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
