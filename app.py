import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
DB_PATH = "faiss_store"

st.set_page_config(page_title="PDF Trainer & Chat", page_icon="📘")
st.title("📘 Military Intelligence Virtual Staff Officer")

# ---------------- FUNCTIONS ----------------
@st.cache_resource
def load_llm():
    """Load local Flan-T5 model."""
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

def train_pdf(uploaded_file):
    """Train FAISS DB with uploaded PDF."""
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(DB_PATH)
    return vectorstore

def ask_llm(question, context):
    """Generate answer using Flan-T5."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = llm_pipeline(prompt, max_length=256, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"]

# ---------------- APP UI ----------------
st.sidebar.header("📂 Upload & Train PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded")
    vectorstore = train_pdf(uploaded_file)
else:
    if os.path.exists(DB_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = None

llm_pipeline = load_llm()

st.header("💬 Chat with your trained PDFs")
if vectorstore is None:
    st.warning("Please upload and train a PDF first.")
else:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    query = st.text_input("Ask a question:")
    if query:
        results = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in results])
        answer = ask_llm(query, context)

        st.subheader("🤖 Answer")
        st.write(answer)

        with st.expander("📄 Retrieved context"):
            for i, res in enumerate(results, start=1):
                st.write(f"**{i}.** {res.page_content}")
