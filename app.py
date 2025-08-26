import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
SECRET_KEY = "TACHUMINT"  # change this to your private key
DATA_FOLDER = "my_training_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Embedding model for Q&A
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Private Chatbot", layout="centered")
st.title("🔒 Private Chatbot (Only MyData Files Allowed)")

# 1. Ask for secret key
user_key = st.text_input("Enter Access Key", type="password")
if user_key != SECRET_KEY:
    st.warning("Access denied. Enter correct key.")
    st.stop()

# 2. File uploader (ONLY .mydata files allowed)
uploaded_file = st.file_uploader("Upload your .mydata file", type=["mydata"])
if uploaded_file is not None:
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Saved {uploaded_file.name}")

# 3. Load all .mydata files into memory
documents = []
for fname in os.listdir(DATA_FOLDER):
    if fname.endswith(".mydata"):
        with open(os.path.join(DATA_FOLDER, fname), "r", encoding="utf-8") as f:
            try:
                pages = json.load(f)
                for p in pages:
                    if "content" in p and p["content"].strip():
                        documents.append(p["content"])
            except:
                st.error(f"❌ Could not read {fname} (invalid format)")

# 4. If we have data, create embeddings
if documents:
    doc_embeddings = embedder.encode(documents)
else:
    st.warning("No data loaded yet. Please upload your .mydata files.")
    st.stop()

# 5. Chat interface
st.subheader("💬 Ask a Question")
query = st.text_input("Your question:")
if query:
    q_emb = embedder.encode([query])
    sims = cosine_similarity(q_emb, doc_embeddings)[0]
    best_idx = np.argmax(sims)
    best_answer = documents[best_idx]

    st.markdown("**Answer:**")
    st.write(best_answer)
