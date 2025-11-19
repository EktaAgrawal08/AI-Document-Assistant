
# **AI Document Assistant — Multi-Format RAG Chatbot**

AI Document Assistant is a smart multi-format **document question-answering system** built using **Python, Streamlit, and Google Gemini API**.
It allows users to upload files (PDF, DOCX, TXT, Images) and ask questions directly from the document contents using a **Retrieval-Augmented Generation (RAG)** pipeline.

This project was created as part of my semester coursework.

---

## **Features**

### **Document Upload (Multi-format)**

Supports:

* PDF
* DOCX
* TXT
* Images (JPG/PNG with OCR)

### **Accurate Answers Using RAG**

* Extracts text from uploaded files
* Splits text into chunks
* Embeds chunks using **Gemini text-embedding-004**
* Retrieves relevant chunks based on cosine similarity
* Uses **Gemini 2.0 Flash** to generate final answers
* Responds *only* from the uploaded documents

### **Chat System**

* Modern dark UI
* Chat history maintained
* Follow-up questions supported
* Download chat as PDF

### **Local Processing**

* Files stored temporarily inside `chat-with-pdf/pdfs/`
* No cloud storage
* No external database

---

## **Tech Stack**

| Component        | Technology                |
| ---------------- | ------------------------- |
| User Interface   | Streamlit                 |
| Backend          | Python                    |
| LLM              | Gemini 2.0 Flash          |
| Embeddings       | text-embedding-004        |
| Document Parsing | PDFPlumber, python-docx   |
| OCR              | Pytesseract               |
| Vector Search    | NumPy + Cosine Similarity |
| PDF Export       | FPDF                      |

---

## **How It Works**

1. User uploads one or more documents
2. Text is extracted
3. Text is split into manageable chunks
4. Chunks are converted into embeddings
5. User query is embedded
6. Similar chunks are retrieved using cosine similarity
7. Gemini Flash generates an answer using only the retrieved context
8. The response is displayed in the chat interface

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/EktaAgrawal08/AI-Document-Assistant.git
cd AI-Document-Assistant
```

### **2. Create Virtual Environment (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Set Gemini API Key (Windows PowerShell)**

```powershell
setx GEMINI_API_KEY "your_api_key_here"
```

Close PowerShell after this once.

---

## **Run the Application**

```bash
streamlit run University_Assistant.py
```

---

## **Project Structure**

```
AI-Document-Assistant/
│
├─ University_Assistant.py
├─ requirements.txt
├─ README.md
│
└─ chat-with-pdf/
   └─ pdfs/        # Temporarily stored files
```

---

## **Future Enhancements**

* Add light mode toggle
* Implement FAISS/Chroma vector store
* Add document summarization
* Add multi-page Streamlit UI
* Add drag-and-drop PDF viewer

---

## **Author**

**Ekta Agrawal**


---



