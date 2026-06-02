import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_mic_recorder import mic_recorder

# -------------------
# Config
# -------------------

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Assistant")
st.caption("Upload one or more PDFs and chat with them.")

with st.expander("🚀 Features"):
    st.markdown("""
    - Chat with multiple PDFs
    - Chat history
    - PDF chunking
    - Voice recording
    - OpenRouter AI integration
    """)

# -------------------
# Session State
# -------------------

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# -------------------
# Upload PDFs
# -------------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    text = ""
    total_pages = 0

    st.sidebar.header("📄 Uploaded PDFs")

    for uploaded_file in uploaded_files:

        pdf = PdfReader(uploaded_file)

        total_pages += len(pdf.pages)

        st.sidebar.write(f"📄 {uploaded_file.name}")

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += "\n\n" + page_text

    # Chunking

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    st.session_state.pdf_text = text
    st.session_state.chunks = chunks
    st.session_state.pdf_loaded = True

    st.success(
        f"✅ {len(uploaded_files)} PDF(s) Loaded Successfully"
    )

    st.sidebar.markdown("---")

    st.sidebar.success("🚀 Assistant Ready")

    st.sidebar.write(
        f"**Total PDFs:** {len(uploaded_files)}"
    )

    st.sidebar.write(
        f"**Total Pages:** {total_pages}"
    )

    st.sidebar.write(
        f"**Chunks Created:** {len(chunks)}"
    )

    st.sidebar.write(
        f"**Characters Loaded:** {len(text):,}"
    )

# -------------------
# Display Chat History
# -------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# -------------------
# Voice Recorder
# -------------------

if st.session_state.pdf_loaded:

    st.subheader("🎤 Voice Question")

    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=True
    )
    if audio:
       st.success("🎤 Voice recorded successfully!")

# -------------------
# Chat
# -------------------

if st.session_state.pdf_loaded:

    question = st.chat_input(
        "Ask a question about the PDFs..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):

            conversation = [
                {
                    "role": "system",
                    "content": f"""
You are an AI PDF Assistant.

Use ONLY the information contained in the uploaded PDF documents.

Rules:
1. Do not invent facts.
2. If information is missing, say:
   "I could not find that information in the uploaded documents."
3. Provide concise and accurate answers.
4. If multiple PDFs contain relevant information,
   compare and combine them.
5. Ask for clarification if needed.

PDF CONTENT:

{st.session_state.pdf_text[:15000]}
"""
                }
            ]

            conversation.extend(
                st.session_state.messages
            )

            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=conversation
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.write(answer)

# -------------------
# Sidebar Actions
# -------------------

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()