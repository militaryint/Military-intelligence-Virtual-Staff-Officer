import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ---------------- CONFIG ----------------
SECRET_KEY = st.secrets["app_key"]  # Stored securely in Streamlit Cloud
DATA_FOLDER = "my_training_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Secure PDF Chatbot", layout="centered")
st.title("🔒 Secure PDF Chatbot")

# 1. Ask for secret key
user_key = st.text_input("Enter Access Key", type="password")
if user_key != SECRET_KEY:
    st.warning("Access denied. Enter correct key.")
    st.stop()

# 2. PDF uploader (only visible after correct key)
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save vector store for later use
        vectorstore.save_local(os.path.join(DATA_FOLDER, "pdf_vectors"))
        st.success(f"✅ PDF '{uploaded_file.name}' uploaded and processed successfully!")

    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")

# 3. Chat interface
query = st.text_input("Ask a question about your PDF:")
if query:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(os.path.join(DATA_FOLDER, "pdf_vectors"), embeddings, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=1)
        if results:
            st.markdown("**Answer:**")
            st.write(results[0].page_content)
        else:
            st.write("No matching content found.")
    except Exception as e:
        st.error(f"❌ Error during search: {e}")
