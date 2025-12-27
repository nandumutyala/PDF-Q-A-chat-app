# 📄 PDF Question Answering Application (LLM + RAG)

## 📌 Overview
This project is an AI-powered PDF Question Answering application that allows users to upload PDF documents and ask natural language questions to retrieve precise, context-aware answers.  
It leverages **Large Language Models (LLMs)** combined with **Retrieval-Augmented Generation (RAG)** to ensure accurate and document-grounded responses.

---

## 🎯 Problem Statement
Traditional keyword-based search in PDFs is inefficient and fails to capture semantic meaning.  
This project solves that by enabling **conversational querying over PDF content**, making document exploration faster, smarter, and more intuitive.

---

## 🚀 Features
- Upload and process PDF documents
- Semantic search using vector embeddings
- Context-aware question answering
- Prevents hallucinations using document-based retrieval
- Simple and interactive user interface
- Scalable architecture for larger documents

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **LLM:** OpenAI / Compatible LLM API  
- **Embeddings:** Sentence Transformers / OpenAI Embeddings  
- **Vector Store:** FAISS  
- **Frameworks:** LangChain  
- **Frontend:** Streamlit  
- **Environment Management:** Python Virtual Environment (`venv`)

---

## 🧠 How It Works (Architecture)
1. PDF files are uploaded by the user.
2. Text is extracted and split into smaller chunks.
3. Each chunk is converted into vector embeddings.
4. Embeddings are stored in a FAISS vector database.
5. User questions are embedded and matched with relevant document chunks.
6. The retrieved context is passed to the LLM.
7. The LLM generates a grounded answer based strictly on retrieved content.

---

## 📂 Project Structure
pdf-qa-app/
├──README.md
├── .env
├── app.py
├── config.toml
├── f_PDF_QA_model.ipynb
├── requirements.txt


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/pdf-qa-app.git
cd pdf-qa-app


2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


3️⃣ Install Dependencies
pip install -r requirements.txt


4️⃣ Set Environment Variables

Create a .env file and add:

OPENAI_API_KEY=your_api_key_here

5️⃣ Run the Application
streamlit run app.py

📊 Output

Users can upload PDFs and ask questions in natural language.

The app returns accurate answers sourced directly from the document.

Responses are contextually relevant and explainable.
