import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

st.set_page_config(
    page_title="AI PDF Chat",
    page_icon="📄"
)

st.title("📄 AI PDF Chat")

# -------------------
# Session State
# -------------------

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------
# Upload PDF
# -------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    st.session_state.pdf_text = text

    st.success("PDF Loaded!")

# -------------------
# Chat
# -------------------

if st.session_state.pdf_text:

    question = st.chat_input(
        "Ask about the PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only using the PDF content."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
PDF CONTENT:

{st.session_state.pdf_text[:15000]}

QUESTION:

{question}
"""
                    }
                ]
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

# -------------------
# Display Messages
# -------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

# -------------------
# Clear Chat
# -------------------

if st.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()