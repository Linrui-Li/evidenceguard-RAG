
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "app"))

from document_loader import extract_text_from_pdf
from text_splitter import split_text
from retriever import retrieve, model
from generator import generate_answer

# 1. Load questions
with open(project_root / "evaluation/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 2. Prepare document
text = extract_text_from_pdf(str(project_root / "data/sample.pdf"))
chunks = split_text(text, chunk_size=400, overlap=50)
chunk_embeddings = model.encode(chunks)

# 3. Compare two retrieval strategies
for threshold in [None, 0.25]:

    print(f"\n===== Threshold: {threshold} =====")

    for item in questions:

        question = item["question"]

        results = retrieve(
            question,
            chunks,
            chunk_embeddings,
            k=3,
            threshold=threshold
        )

        if not results:
            answer = "I don't have enough evidence to answer."

        else:
            context = "\n\n".join(
                f"[Chunk {r['chunk_id']}]\n{r['text']}"
                for r in results
            )

            answer = generate_answer(question, context)

        print(f"\nQuestion: {question}")
        print(f"Expected answerable: {item['answerable']}")
        print(f"Retrieved chunks: {len(results)}")
        print(f"Answer: {answer}")
        print("-" * 50)
