
import streamlit as st
import hashlib

from langchain_core.messages import AIMessage, HumanMessage

import rag_backend as backend


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .tagline {
        text-align: center;
        font-size: 1.15rem;
        color: #888;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    .section-gap {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

def init_state():

    defaults = {
        "message_history": [],
        "groq_api_key": "",
        "llm_ready": False,
        "processed_file_signature": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ---------------------------------------------------------------------------
# Welcome Screen
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-title">🧠 DocuMind</div>
    <div class="tagline">
        A smarter way to find what matters.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Step 1: Groq API Connection
# ---------------------------------------------------------------------------

st.header("🔑 Connect to DocuMind")

st.write(
    "To get started, connect your Groq API key. "
    "This allows DocuMind to understand your questions "
    "and generate answers from your documents."
)

api_key_input = st.text_input(
    "Enter your Groq API key",
    type="password",
    placeholder="Enter your Groq API key here",
    key="groq_api_key_input",
    help="Your API key is kept in session memory only.",
)


# ---------------------------------------------------------------------------
# Groq API Key Instructions
# ---------------------------------------------------------------------------

with st.expander("❓ How do I get a Groq API key?"):

    st.write(
        "Follow these simple steps to connect DocuMind."
    )

    st.markdown(
        """
        **Step 1:** Visit the
        [Groq Console](https://console.groq.com/).

        **Step 2:** Create an account or sign in.

        **Step 3:** Open the API Keys section.

        **Step 4:** Select Create API Key.

        **Step 5:** Copy your generated API key.

        **Step 6:** Return to DocuMind and paste the key
        into the input field above.
        """
    )

    st.info(
        "Keep your API key private. Never share it with anyone."
    )


# ---------------------------------------------------------------------------
# Connect Button
# ---------------------------------------------------------------------------

if st.button("Connect to Groq", use_container_width=True):

    if not api_key_input.strip():

        st.error("Please enter your Groq API key.")

        st.session_state["llm_ready"] = False

    else:

        with st.spinner("Connecting to Groq..."):

            ok, msg = backend.init_llm(api_key_input.strip())

        if ok:

            st.session_state["groq_api_key"] = api_key_input.strip()
            st.session_state["llm_ready"] = True

            st.success("Connected successfully! DocuMind is ready.")

        else:

            st.session_state["llm_ready"] = False

            st.error(msg)


if st.session_state["llm_ready"]:

    st.success("🟢 Groq AI is connected.")


st.divider()


# ---------------------------------------------------------------------------
# Step 2: Multiple PDF Upload
# ---------------------------------------------------------------------------

st.header("📄 Upload Your Documents")

st.write(
    "Choose one or multiple PDF documents. "
    "DocuMind will process them so you can find information easily."
)

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select multiple PDF documents.",
)


# ---------------------------------------------------------------------------
# Generate File Signature
# ---------------------------------------------------------------------------

def get_file_signature(files):

    signature = []

    for file in files:

        file_bytes = file.getvalue()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        signature.append(
            (file.name, len(file_bytes), file_hash)
        )

    return tuple(signature)


current_signature = (
    get_file_signature(uploaded_files)
    if uploaded_files
    else None
)


# ---------------------------------------------------------------------------
# Display Selected Files
# ---------------------------------------------------------------------------

if uploaded_files:

    st.write(f"**{len(uploaded_files)} PDF file(s) selected:**")

    for file in uploaded_files:

        st.write(f"📄 {file.name}")


# ---------------------------------------------------------------------------
# Process PDFs Button
# ---------------------------------------------------------------------------

if st.button(
    "Process PDFs",
    use_container_width=True,
    disabled=not uploaded_files or not st.session_state["llm_ready"],
):

    files_to_process = [
        (file.getvalue(), file.name)
        for file in uploaded_files
    ]

    with st.spinner("Processing your documents..."):

        ok, msg = backend.load_pdfs(files_to_process)

    if ok:

        st.session_state["processed_file_signature"] = current_signature

        # Clear old conversation when new documents are loaded
        st.session_state["message_history"] = []

        st.success(msg)

    else:

        st.session_state["processed_file_signature"] = None

        st.error(msg)


# ---------------------------------------------------------------------------
# Document Status
# ---------------------------------------------------------------------------

documents_ready = (
    backend.rag_ready
    and current_signature is not None
    and current_signature == st.session_state["processed_file_signature"]
)

if documents_ready:

    st.success(
        f"🟢 {len(backend.loaded_filenames)} PDF document(s) ready."
    )

elif uploaded_files:

    st.info(
        "Your selected PDFs are not processed yet. "
        "Click Process PDFs to continue."
    )

else:

    st.info("Upload your PDF documents to get started.")


st.divider()


# ---------------------------------------------------------------------------
# Step 3: Ask Questions
# ---------------------------------------------------------------------------

st.header("💬 Ask DocuMind")

st.write(
    "Ask questions about your uploaded documents "
    "and get answers based on their content."
)


# ---------------------------------------------------------------------------
# New Chat Button
# ---------------------------------------------------------------------------

if st.button("➕ New Chat"):

    st.session_state["message_history"] = []

    st.rerun()


# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------

ready = (
    st.session_state["llm_ready"]
    and documents_ready
)

placeholder = (
    "Ask a question about your documents..."
    if ready
    else "Connect Groq and process your PDFs to start chatting."
)

user_input = st.chat_input(
    placeholder,
    disabled=not ready,
)


# ---------------------------------------------------------------------------
# Generate AI Response
# ---------------------------------------------------------------------------

if user_input and ready:

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    with st.chat_message("assistant"):

        def stream_response():

            for chunk, metadata in backend.chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="messages",
            ):

                if (
                    isinstance(chunk, AIMessage)
                    and chunk.content
                    and not getattr(chunk, "tool_calls", None)
                ):

                    if isinstance(chunk.content, str):

                        yield chunk.content

        try:

            ai_response = st.write_stream(stream_response())

            if ai_response:

                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": ai_response,
                    }
                )

        except Exception as exc:

            st.error(
                f"Something went wrong while generating the answer: {exc}"
            )
