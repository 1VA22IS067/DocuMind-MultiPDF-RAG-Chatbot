# 🧠 DocuMind
### A smarter way to find what matters.

DocuMind is a Retrieval-Augmented Generation (RAG) chatbot that helps users find information across multiple PDF documents without manually reading every page.

Upload your documents, ask questions in natural language, and receive answers based on the information retrieved from your files.

## ✨ Features

- **Multiple PDF Upload:** Upload and process multiple PDF documents together.
- **Beginner-Friendly Interface:** Simple interface with clear instructions for connecting to Groq.
- **Groq API Integration:** Connect using your Groq API key.
- **Semantic Search:** Find relevant information using FAISS vector search.
- **Document-Based Answers:** Generate responses using retrieved document content.
- **Hallucination Control:** The chatbot is instructed to respond when relevant information cannot be found rather than making up an answer.
- **Chat Interface:** Ask multiple questions and view conversation history.
- **Streaming Responses:** View AI-generated answers as they are produced.
- **New Chat:** Clear the current conversation and start fresh.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| LangChain | LLM and document retrieval integration |
| LangGraph | Controls the chatbot workflow |
| Groq | LLM inference |
| GPT-OSS 120B | Language model |
| Hugging Face | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector storage and similarity search |
| PyPDFLoader | Extracts text from PDF documents |

## ⚙️ How It Works

1. The user connects DocuMind using a Groq API key.
2. Multiple PDF documents are uploaded.
3. PyPDFLoader extracts text from the uploaded documents.
4. RecursiveCharacterTextSplitter divides the extracted text into smaller chunks.
5. Hugging Face converts the chunks into numerical embeddings.
6. FAISS creates a vector index containing the chunks from all uploaded PDFs.
7. When a user asks a question, LangGraph routes the question through the retrieval tool.
8. FAISS retrieves the four most relevant text chunks.
9. The retrieved information is passed to the Groq language model.
10. The model generates an answer based on the retrieved document content.

## 🔄 Application Workflow

```text
        User
         |
         v
    Streamlit UI
         |
         v
   Upload Multiple PDFs
         |
         v
     PyPDFLoader
         |
         v
   Text Extraction
         |
         v
  Text Chunking
         |
         v
 Hugging Face Embeddings
         |
         v
      FAISS
   Vector Index
         |
         v
   User Question
         |
         v
     LangGraph
         |
         v
      RAG Tool
         |
         v
  FAISS Similarity Search
         |
         v
 Relevant Text Chunks
         |
         v
    Groq LLM
         |
         v
   Generated Answer
```

## 📁 Project Structure

```text
DocuMind-MultiPDF-RAG-Chatbot/
│
├── .streamlit/
│   └── config.toml
│
├── rag_app.py
├── rag_backend.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/1VA22IS067/DocuMind-MultiPDF-RAG-Chatbot.git
```

```bash
cd DocuMind-MultiPDF-RAG-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows PowerShell:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run rag_app.py
```

The application will open at:

`http://localhost:8501`

## 🔑 Getting a Groq API Key

1. Visit [Groq Console](https://console.groq.com/).
2. Create an account or sign in.
3. Navigate to the API Keys section.
4. Create a new API key.
5. Copy the key and enter it in DocuMind.

Keep your API key private. Never commit API keys or secret files to GitHub.

## 💬 How to Use DocuMind

1. Open the DocuMind application.
2. Enter your Groq API key and connect.
3. Select one or multiple PDF documents.
4. Click **Process PDFs**.
5. Enter your question in the chat interface.
6. Receive an answer based on the uploaded documents.
7. Use **New Chat** to clear the conversation and start again.

## ⚠️ Current Limitations

- Supports PDF document uploads.
- Scanned or image-only PDFs may not provide readable text because OCR is not implemented.
- The application retrieves a limited number of relevant chunks for each question.
- Answers depend on the quality and content of the uploaded documents and retrieved passages.
- Uploaded document indexes and chat history are held in application memory and are not persistent across application restarts.

## 👩‍💻 Project Attribution

This project is an extended and modified version of the original DocuMind RAG Chatbot.

**Original Repository:** [Muqadas Ejaz – DocuMind RAG Chatbot](https://github.com/muqadasejaz/DocuMind-RAG-Chatbot)

The original project provided the foundation for the RAG chatbot. This repository extends it with multiple PDF processing and an improved, beginner-friendly user interface.

The original MIT License is retained.

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for details.
