import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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

    # Build retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load a local model (Flan-T5)
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
    query = st.text_input("💬 Ask a question about your PDF:")
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
