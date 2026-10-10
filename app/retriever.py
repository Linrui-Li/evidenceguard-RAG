from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from document_loader import extract_text_from_pdf
from text_splitter import split_text


model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(question, chunks, chunk_embeddings, k=3, threshold=None):

    # Convert question into embedding
    question_embedding = model.encode(question)

    # Calculate similarity scores
    scores = cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    # Select Top-K chunks
    top_results = scores.topk(k=min(k, len(chunks)))

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        if threshold is not None and score.item() < threshold:
            continue

        results.append({
            "text": chunks[index.item()],
            "score": score.item(),
            "chunk_id": index.item()
        })

    return results


# if __name__ == "__main__":

#     text = extract_text_from_pdf("data/sample.pdf")

#     chunks = split_text(
#         text,
#         chunk_size=400,
#         overlap=50
#     )

#     chunk_embeddings = model.encode(chunks)

#     question = "What are the possible states of weather?"

#     results = retrieve(
#         question,
#         chunks,
#         chunk_embeddings,
#         k=3
#     )

#     for result in results:
#         print("\n--- Retrieved Chunk ---")
#         print("Score:", result["score"])
#         print(result["text"])