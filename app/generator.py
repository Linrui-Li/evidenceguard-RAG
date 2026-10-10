from ollama import chat

def generate_answer(question, context):
    response = chat(
        model="mistral:latest",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded question answering assistant. "
                    "Answer questions using only the provided context. "
                    "Cite the supporting chunk IDs in your answer, "
                    "using the format [Chunk N]. "
                    "Only cite chunk IDs that appear in the context. "
                    "If the context does not contain enough evidence, "
                    "say 'I don't have enough evidence to answer.' "
                    "Do not invent facts or sources."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )

    return response.message.content


if __name__ == "__main__":
    question = "What are the three possible states of weather?"

    context = (
        "There are three states of weather conditions "
        "in a city: sunny, cloudy and rainy."
    )

    answer = generate_answer(question, context)

    print("Question:", question)
    print("Answer:", answer)