
---

# 🏥 Clinical Diagnosis Assistant with Retrieval-Augmented Generation (RAG)

This project builds an intelligent, AI-powered medical assistant that integrates **OCR**, **medical knowledge graphs**, and **clinical notes** with **state-of-the-art LLMs** and **vector search** to answer complex clinical queries. It uses **LangChain**, **FAISS**, **Hugging Face models**, and **Mistral-7B-Instruct** for efficient information retrieval and natural language generation.

---

## 🔍 Key Features

* 🔤 **OCR Processing** using Tesseract to extract and clean medical text from images or scanned documents.
* 📚 **Knowledge Graph & Clinical Notes Loader**: Reads, parses, and cleans structured and unstructured JSON medical data.
* 🤖 **Embedding & Vector Store**: Creates vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` and stores them in a FAISS index.
* 💬 **Conversational QA Pipeline**: Uses **Mistral 7B Instruct** model for answering natural language clinical queries with context from retrieved documents.
* 🔌 **LangChain Integration** for smooth handling of document objects and retrieval chaining.

---

## 🧠 How It Works

### Step 1: OCR and Text Cleaning

Extract text from images using `pytesseract`, then clean it using regex operations to remove unwanted characters and normalize it for processing.

### Step 2: Load Knowledge Graphs and Clinical Notes

Parse JSON files containing medical diagnosis steps and clinical case notes into structured text using recursive data traversal.

### Step 3: Clean & Embed Documents

* Clean all text using custom regex functions.
* Convert all cleaned text into `LangChain.Document` objects.
* Generate vector embeddings using `all-MiniLM-L6-v2`.
* Save them in a local **FAISS** index.

### Step 4: Query Answering with RAG

* Load FAISS index and use it as a retriever.
* Use the **Mistral-7B-Instruct** model with 4-bit quantization for efficient text generation.
* Combine the retriever context with the user query to generate rich, accurate answers.

---


## 📦 Dependencies

* `pytesseract`
* `Pillow`
* `PyPDF2`
* `transformers`
* `torch`, `bitsandbytes`
* `langchain`, `sentence-transformers`
* `faiss-cpu` or `faiss-gpu`
* `huggingface_hub`
* `google-generativeai` (optional for Gemini API testing)

---

## 🔐 API Keys and Secrets

* For Gemini API: Set your key in `genai.Client(api_key="YOUR_KEY")`.
* For HuggingFace: Login using `huggingface-cli login` or use `login(token="YOUR_TOKEN")`.

---

## ⚠️ Warnings

* **Large Models**: Mistral-7B is a heavy model and should ideally run on GPUs with at least 16GB VRAM.
* **Deserialization Warning**: `allow_dangerous_deserialization=True` is used when loading FAISS — avoid using it in production without safety checks.

---

## ✅ Future Enhancements

* UI via **Streamlit** or **Gradio**
* Integration with **medical ontologies**
* Real-time OCR from scanned PDF uploads
* Add support for other language models (e.g., LLaMA, Claude)

---
