import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import openai


load_dotenv()
openai_api_key = os.getenv("OPENROUTER_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
pinecone_region = os.getenv("PINECONE_REGION")


pc = Pinecone(api_key=pinecone_key)
index = pc.Index(pinecone_index)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


system_prompt = "You are a helpful and emotionally intelligent assistant specialized in relationship advice."


st.set_page_config(page_title="Relationship Advice AI", layout="centered")
st.title("Relationship Advice AI")
st.write("Ask me anything about relationships. I'm here to help!")

question = st.text_input("Your question:", max_chars=200)

if st.button("Get Advice") and question:
    with st.spinner("Thinking..."):
        
        embedded_question = embed_model.encode([question]).tolist()

        
        search_result = index.query(vector=embedded_question[0], top_k=5, include_metadata=True)

        
        context = "\n".join([item["metadata"]["text"] for item in search_result["matches"]])

      
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]

        
        client = openai.OpenAI(api_key=openai_api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",  
            messages=messages
        )

        answer = response.choices[0].message.content
        st.markdown("### Advice:")
        st.write(answer)
