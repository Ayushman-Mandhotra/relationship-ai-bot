import json
from sentence_transformers import SentenceTransformer
import chromadb


with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)


model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(path="embeddings")
collection = client.get_or_create_collection("relationship_advice")


for chunk in chunks:
    embedding = model.encode(chunk["text"])
    collection.add(
        documents=[chunk["text"]],
        embeddings=[embedding],
        ids=[chunk["id"]],
        metadatas=[{"source": chunk["source"]}]
    )

print(f" Embedded {len(chunks)} chunks and saved to ChromaDB.")
