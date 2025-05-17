import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, PodSpec


load_dotenv()
openrouter_key = os.getenv("OPENROUTER_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
region = os.getenv("PINECONE_REGION")


pc = Pinecone(api_key=pinecone_key)
index = pc.Index(index_name)


embedder = SentenceTransformer("all-MiniLM-L6-v2")


client_ai = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")


st.title(" Relationship Advice Chatbot")
user_query = st.text_input("Ask your relationship question:")

if user_query:
    with st.spinner("Thinking..."):
        query_vector = embedder.encode(user_query).tolist()
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        chunks = [match["metadata"]["text"] for match in results["matches"]]
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
