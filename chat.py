import streamlit as st
st.write("DEBUG: This is chat.py running")
import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
DB_PATH = "faiss_store"

st.title("💬 Chat with your Trained PDFs")

# Load FAISS database
if not os.path.exists(DB_PATH):
    st.error("❌ No trained database found! Please run train_pdf.py first.")
    st.stop()

# Load embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load local model (Flan-T5)
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

llm_pipeline = load_llm()

def ask_llm(question, context):
    """Feed question + context to Flan-T5."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = llm_pipeline(prompt, max_length=256, clean_up_tokenization_spaces=True)
    return result[0]["generated_text"]

# Chat input
query = st.text_input("💡 Ask a question about your trained PDFs:")

if query:
    # Get relevant chunks
    results = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in results])

    # Generate answer
    answer = ask_llm(query, context)

    # Show answer
    st.subheader("🤖 Answer")
    st.write(answer)

    # Show retrieved chunks (optional)
    with st.expander("📄 Retrieved context"):
        for i, res in enumerate(results, start=1):
            st.write(f"**{i}.** {res.page_content}")
