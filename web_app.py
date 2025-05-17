import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai


load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_region = os.getenv("PINECONE_REGION")


pc = Pinecone(api_key=pinecone_key)
index = pc.Index(
    name=pinecone_index_name,
    spec=ServerlessSpec(cloud="aws", region=pinecone_region)
)


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


st.set_page_config(page_title="Relationship AI Bot")
st.title(" Relationship Advice AI")
st.write("Ask me anything about relationships. I'm here to help!")


user_question = st.text_input("Your question:", placeholder="e.g., How do I fix communication in a long-distance relationship?")


if st.button("Get Advice") and user_question:
   
    user_embedding = embed_model.encode(user_question).tolist()

   
    search_results = index.query(
        vector=user_embedding,
        top_k=5,
        include_metadata=True
    )

    
    context_chunks = [match["metadata"]["text"] for match in search_results["matches"] if "text" in match["metadata"]]
    context_text = "\n".join(context_chunks)

   
    prompt = f"""You are an AI trained to give advice on relationships.

Context:
{context_text}

Question: {user_question}
Answer:"""

  
    response = openai.ChatCompletion.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful relationship advisor."},
            {"role": "user", "content": prompt}
        ]
    )

   
    ai_answer = response['choices'][0]['message']['content']
    st.markdown("### 💡 Advice")
    st.write(ai_answer)
