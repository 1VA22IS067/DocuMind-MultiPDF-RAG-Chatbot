
from __future__ import annotations

import os
import tempfile
import logging
from typing import Annotated, Optional, Tuple, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------------------

retriever: Optional[object] = None
rag_ready: bool = False

# Store the names of all uploaded PDFs
loaded_filenames: list[str] = []

# Keep this variable for displaying uploaded filenames in the app
loaded_filename: Optional[str] = None

_current_api_key: Optional[str] = None
chatbot: Optional[object] = None


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(
    content="""You are a helpful assistant with access to a document
retrieval tool called rag_tool.

STRICT RULES — follow these without exception:

1. For ANY user question, you MUST call rag_tool first.
   Never answer from your own knowledge.

2. After receiving the tool result, check carefully:
   - If the tool returns useful content, answer the question
     based only on that content.
   - If the tool says no PDF is uploaded, politely ask
     the user to upload one.
   - If the tool returns content but it does not contain
     a relevant answer, respond with:
     "I couldn't find an answer to that in the uploaded documents."
   - Do NOT make up an answer or use general knowledge.

3. Never expose raw tool output, JSON, metadata,
   file paths, or page numbers in your response.

4. Keep answers clear, concise, and grounded strictly
   in the uploaded document content."""
)


# ---------------------------------------------------------------------------
# Chat State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Build LangGraph Workflow
# ---------------------------------------------------------------------------

def _build_graph(api_key: str):

    lm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key
    )

    lm_forced = lm.bind_tools(
        [rag_tool],
        tool_choice="any"
    )

    lm_free = lm.bind_tools([rag_tool])

    def chat_node(state: ChatState):

        from langchain_core.messages import ToolMessage

        messages = state["messages"]

        last_is_tool_result = (
            bool(messages)
            and isinstance(messages[-1], ToolMessage)
        )

        active_lm = lm_free if last_is_tool_result else lm_forced

        response = active_lm.invoke(
            [SYSTEM_PROMPT] + messages
        )

        return {"messages": [response]}

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode([rag_tool]))

    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge("tools", "chat_node")

    return graph.compile()


# ---------------------------------------------------------------------------
# Initialize Groq LLM
# ---------------------------------------------------------------------------

def init_llm(api_key: str) -> Tuple[bool, str]:

    global _current_api_key, chatbot

    if not api_key or not api_key.strip():
        return False, "API key must not be empty."

    api_key = api_key.strip()

    if api_key == _current_api_key and chatbot is not None:
        return True, "Already initialised."

    try:

        chatbot = _build_graph(api_key)
        _current_api_key = api_key

        logger.info("LLM graph initialised successfully.")

        return True, "LLM initialised successfully."

    except Exception as exc:

        logger.exception("Failed to initialise LLM.")

        chatbot = None
        _current_api_key = None

        return False, f"Failed to initialise LLM: {exc}"


# ---------------------------------------------------------------------------
# Load Multiple PDFs
# ---------------------------------------------------------------------------

def load_pdfs(files: list[tuple[bytes, str]]) -> Tuple[bool, str]:

    global retriever
    global rag_ready
    global loaded_filename
    global loaded_filenames

    temp_paths = []

    # Clear the previous document index
    retriever = None
    rag_ready = False
    loaded_filename = None
    loaded_filenames = []

    try:

        if not files:
            return False, "Please upload at least one PDF."

        all_docs = []
        successful_filenames = []

        # Read every uploaded PDF
        for file_bytes, filename in files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(file_bytes)
                tmp_path = tmp.name
                temp_paths.append(tmp_path)

            # Extract text from the PDF
            loader = PyPDFLoader(tmp_path)

            docs = loader.load()

            if docs:

                for doc in docs:
                    doc.metadata["source"] = filename

                all_docs.extend(docs)
                successful_filenames.append(filename)

        if not all_docs:
            return False, "No readable content was found in the uploaded PDFs."

        # Split the content of all PDFs into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(all_docs)

        if not chunks:
            return False, "No text chunks could be created from the PDFs."

        # Create ONE FAISS vector store using all PDF chunks
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        rag_ready = True

        loaded_filenames = successful_filenames
        loaded_filename = ", ".join(successful_filenames)

        logger.info(
            "Loaded %d PDFs containing %d chunks.",
            len(successful_filenames),
            len(chunks)
        )

        return True, (
            f"Successfully loaded {len(successful_filenames)} PDF(s) "
            f"and processed {len(chunks)} text chunks."
        )

    except Exception as exc:

        retriever = None
        rag_ready = False
        loaded_filename = None
        loaded_filenames = []

        logger.exception("PDF loading failed.")

        return False, f"Failed to process PDFs: {exc}"

    finally:

        # Delete temporary PDF files
        for path in temp_paths:

            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# Support Existing Single-PDF Function
# ---------------------------------------------------------------------------

def load_pdf(file_bytes: bytes, filename: str) -> Tuple[bool, str]:

    return load_pdfs([
        (file_bytes, filename)
    ])


# ---------------------------------------------------------------------------
# RAG Retrieval Tool
# ---------------------------------------------------------------------------

@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the uploaded PDF documents.
    Always call this tool before answering any user question."""

    if not rag_ready or retriever is None:
        return "NO_PDF: No PDF documents have been uploaded yet."

    try:

        results = retriever.invoke(query)

        if not results:
            return (
                "NO_MATCH: No relevant content was found "
                "in the uploaded documents."
            )

        passages = "\n\n---\n\n".join(
            doc.page_content for doc in results
        )

        return (
            "FOUND: Retrieved content from the uploaded documents:\n\n"
            f"{passages}"
        )

    except Exception as exc:

        logger.exception("rag_tool retrieval error.")

        return (
            "ERROR: An error occurred while retrieving information: "
            f"{exc}"
        )
