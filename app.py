import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load HuggingFace token if needed
load_dotenv()

# --- Load QA Model ---
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

qa_pipeline = load_model()

# --- Helper Functions ---
def load_and_split_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = "uploaded_pdf"
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)

def ask_question(vector_store, question):
    docs = vector_store.similarity_search(question)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"""
    Answer the question based on the context below. Be concise and accurate.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    result = qa_pipeline(prompt, max_new_tokens=300, do_sample=False)
    return result[0]['generated_text'].strip()

# --- Streamlit App UI ---
st.set_page_config(page_title="📘 Search Q&A from the PDF", layout="wide")
st.markdown("<h1 style='color:#39FF14;'>📘 Search Q&A from the PDF</h1>", unsafe_allow_html=True)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload and question
pdf_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Ask a question about the PDF", placeholder="e.g., What is the main objective of this paper?")

if pdf_file:
    file_path = os.path.join(os.getcwd(), pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    chunks = load_and_split_pdf(file_path)
    vector_store = create_vector_store(chunks)

    if question:
        with st.spinner("Getting answer..."):
            answer = ask_question(vector_store, question)
            st.session_state.chat_history.append((question, answer))
            st.success(answer)

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for idx, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
        st.markdown(f"**Q{idx}:** {q}")
        st.markdown(f"**A{idx}:** {a}")
        st.markdown("---")

# Optional: Reset chat
if st.button("🔄 Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()
