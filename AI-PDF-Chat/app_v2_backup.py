
import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

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

# -------------------
# Session State
# -------------------

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# -------------------
# Upload PDF
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

        st.sidebar.write(
            f"📄 {uploaded_file.name}"
        )

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += "\n\n" + page_text

    st.session_state.pdf_text = text
    st.session_state.pdf_loaded = True

    st.success(
        f"✅ {len(uploaded_files)} PDF(s) Loaded Successfully"
    )

    st.sidebar.markdown("---")

    st.sidebar.write(
        f"**Total PDFs:** {len(uploaded_files)}"
    )

    st.sidebar.write(
        f"**Total Pages:** {total_pages}"
    )

# -------------------
# Display Chat History
# -------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(
            message["content"]
        )

# -------------------
# Chat
# -------------------

if st.session_state.pdf_loaded:

    question = st.chat_input(
        "Ask a question about the PDF..."
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
You are an AI PDF assistant.

Answer ONLY using the PDF content below.

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
# Clear Chat
# -------------------

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()
