import streamlit as st
st.write("DEBUG: This is train_pdf.py running")
import streamlit as st
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- CONFIG ----------------
DB_PATH = "faiss_store"

st.title("📘 Train PDF Knowledge Base")

uploaded_file = st.file_uploader("Upload a PDF to train", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create or update FAISS vector store
    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore
    vectorstore.save_local(DB_PATH)

    st.success("✅ PDF trained and stored in FAISS database!")
