import os
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


client_ai = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


client = chromadb.PersistentClient(path="embeddings")
collection = client.get_or_create_collection("relationship_advice")


embedder = SentenceTransformer("all-MiniLM-L6-v2")


st.title("Relationship Advice Chatbot")
user_query = st.text_input("Ask your relationship question:")

if user_query:
    with st.spinner("Thinking..."):
        query_vector = embedder.encode(user_query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=5)
        chunks = results["documents"][0]
        context = "\n\n".join(chunks)

        system_prompt = "You are a helpful relationship advisor. Use the insights from expert books:\n\n" + context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = client_ai.chat.completions.create(
            model="deepseek/deepseek-chat:free",
            messages=messages
        )

        st.markdown("Advice:")
        st.write(response.choices[0].message.content.strip())
