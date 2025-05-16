import os
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer


api_key = "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"  


client = chromadb.PersistentClient(path="embeddings")
collection = client.get_or_create_collection("relationship_advice")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


print("Ask a relationship question:")
query = input("> ")


query_vector = embedder.encode(query).tolist()
results = collection.query(query_embeddings=[query_vector], n_results=5)
chunks = results["documents"][0]
context = "\n\n".join(chunks)


system_prompt = "You are a helpful relationship advisor. Answer questions using the following expert book passages:\n\n" + context


headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://openrouter.ai",  
    "X-Title": "Relationship AI Bot"
}

data = {
    "model": "deepseek/deepseek-chat:free",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)


if response.status_code == 200:
    reply = response.json()['choices'][0]['message']['content']
    print("\n Advice:")
    print(reply.strip())
else:
    print(" Error:", response.status_code)
    print(response.text)
