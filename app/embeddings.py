from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Machine learning models learn patterns from data.",
    "AI systems can learn from training data.",
    "I ate pizza for dinner yesterday."
]

embeddings = model.encode(sentences)

print(embeddings.shape)
print(embeddings[0][:10])

similarity_1_2 = cos_sim(embeddings[0], embeddings[1])
similarity_1_3 = cos_sim(embeddings[0], embeddings[2])

print("Sentence 1 vs Sentence 2:", similarity_1_2.item())
print("Sentence 1 vs Sentence 3:", similarity_1_3.item())