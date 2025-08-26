import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF reading

# ---------------- CONFIG ----------------
SECRET_KEY = st.secrets["app_key"]  # Stored in Streamlit Secrets
DATA_FOLDER = "my_training_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Embedding model for Q&A
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Private PDF Chatbot", layout="centered")
st.title("🔒 Private PDF Chatbot")

# 1. Ask for secret key
user_key = st.text_input("Enter Access Key", type="password")
if user_key != SECRET_KEY:
    st.warning("Access denied. Enter correct key.")
    st.stop()

# 2. PDF uploader (only available if key correct)
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_pdf is not None:
    pdf_path = os.path.join(DATA_FOLDER, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    # Convert PDF → .mydata
    doc = fitz.open(pdf_path)
    pages_data = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages_data.append({"content": text})
    doc.close()

    # Save .mydata
    mydata_name = uploaded_pdf.name.replace(".pdf", ".mydata")
    mydata_path = os.path.join(DATA_FOLDER, mydata_name)
    with open(mydata_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)

    st.success(f"✅ PDF converted and saved as {mydata_name}")

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
    st.warning("No data loaded yet. Please upload a PDF.")
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
