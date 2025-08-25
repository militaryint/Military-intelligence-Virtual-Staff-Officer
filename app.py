import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title="Virtual Teacher", page_icon="📘")

st.title("📘 Virtual PDF Teacher")
st.write("Upload a PDF and ask questions. Powered by Hugging Face + LangChain.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Load PDF
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    # Hugging Face model (free)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

    st.success("✅ PDF processed successfully! Ask away.")

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask a question:")
    if query:
        result = qa({"question": query, "chat_history": st.session_state.history})
        st.session_state.history.append((query, result["answer"]))
        st.write("**Answer:**", result["answer"])
