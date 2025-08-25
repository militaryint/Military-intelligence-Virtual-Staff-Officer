import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- APP ----------------
st.title("📘 Military Intelligence Virtual Staff Officer")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Load PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("✅ PDF processed and stored in vector database!")

    # Chat input
    query = st.text_input("Ask a question about your PDF:")
    if query:
        results = vectorstore.similarity_search(query, k=3)
        st.subheader("🔍 Results")
        for i, res in enumerate(results, start=1):
            st.write(f"**{i}.** {res.page_content}")
