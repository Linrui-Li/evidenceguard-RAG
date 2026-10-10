from document_loader import extract_text_from_pdf
from text_splitter import split_text
from retriever import retrieve, model
from generator import generate_answer


# 1. Load document
text = extract_text_from_pdf("data/sample.pdf")

# 2. Split document into chunks
chunks = split_text(text, chunk_size=400, overlap=50)

# 3. Generate document embeddings
chunk_embeddings = model.encode(chunks)

# 4. User question
question = "What are the possible states of weather?"

# 5. Retrieve relevant chunks
results = retrieve(question, chunks, chunk_embeddings, k=3, threshold=0.25)

print("\nQuestion:", question)

if not results:
    print("\nAnswer: I don't have enough evidence to answer.")
    print("\nNo relevant evidence found.")

else:
    # 6. Build context
    context = "\n\n".join(
        f"[Chunk {result['chunk_id']}]\n{result['text']}"
        for result in results
    )

    # 7. Generate answer
    answer = generate_answer(question, context)

    # 8. Display result
    print("\nAnswer:", answer)

    print("\n--- Retrieved Evidence ---")

for result in results:
    print(f"\nChunk ID: {result['chunk_id']}")
    print(f"Score: {result['score']:.4f}")
    print(result["text"][:300])