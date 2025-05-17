import os
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure it's in the .env file as OPENROUTER_API_KEY.")


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
    print("\nAdvice:")
    print(reply.strip())
else:
    print("Error:", response.status_code)
    print(response.text)
