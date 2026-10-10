
import json
import sys
from pathlib import Path

# Allow importing modules from app/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "app"))

from document_loader import extract_text_from_pdf
from text_splitter import split_text
from retriever import retrieve, model


# 1. Load evaluation questions
with open(project_root / "evaluation/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)


# 2. Prepare document and embeddings
text = extract_text_from_pdf(str(project_root / "data/sample.pdf"))
chunks = split_text(text, chunk_size=400, overlap=50)
chunk_embeddings = model.encode(chunks)


# 3. Evaluate different thresholds
thresholds = [0.15, 0.25, 0.35]

for threshold in thresholds:
    correct = 0

    print(f"\n===== Threshold: {threshold} =====")

    for item in questions:
        results = retrieve(
            item["question"],
            chunks,
            chunk_embeddings,
            k=3,
            threshold=threshold
        )

        predicted_answerable = len(results) > 0
        expected_answerable = item["answerable"]

        is_correct = predicted_answerable == expected_answerable

        if is_correct:
            correct += 1

        print(
            f"{item['id']} | "
            f"Expected: {expected_answerable} | "
            f"Predicted: {predicted_answerable} | "
            f"Correct: {is_correct}"
        )

    accuracy = correct / len(questions)

    print(f"Accuracy: {accuracy:.2%}")

