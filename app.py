import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from streamlit_mic_recorder import mic_recorder

# -----------------------
# CONFIG
# -----------------------

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

st.set_page_config(
    page_title="AI PDF Assistant Pro",
    page_icon="📄",
    layout="wide"
)

# -----------------------
# MODERN CSS
# -----------------------

st.markdown("""
<style>

.block-container {
    max-width: 1300px;
}

.stButton button {
    width:100%;
    border-radius:14px;
    height:55px;
    font-weight:bold;
}

.stChatMessage {
    border-radius:18px;
    padding:10px;
}

[data-testid="stSidebar"]{
    background:#111827;
}

div[data-testid="metric-container"]{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
    border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------

st.markdown("""
<div style="
padding:40px;
border-radius:25px;
background:linear-gradient(135deg,#2563eb,#7c3aed,#ec4899);
text-align:center;
margin-bottom:25px;
box-shadow:0 10px 30px rgba(0,0,0,0.3);
">

<h1 style="color:white;font-size:48px;">
📄 AI PDF Assistant Pro
</h1>

<h3 style="color:white;">
Your Personal AI Research Assistant
</h3>

<p style="color:white;font-size:18px;">
Upload PDFs • Search • Compare • Summarize • AI Powered
</p>

</div>
""", unsafe_allow_html=True)

# -----------------------
# SESSION STATE
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# -----------------------
# PDF UPLOAD
# -----------------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    text = ""

    total_pages = 0
    total_size = 0

    for uploaded_file in uploaded_files:

        pdf = PdfReader(uploaded_file)

        total_pages += len(pdf.pages)
        total_size += uploaded_file.size

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += "\n\n" + page_text

    with st.spinner("Building knowledge base..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma.from_texts(
            chunks,
            embeddings
        )

        st.session_state.db = db
        st.session_state.chunks = chunks
        st.session_state.pdf_loaded = True

    st.success(
        f"Loaded {len(uploaded_files)} PDF(s)"
    )

    # Dashboard

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "PDFs",
            len(uploaded_files)
        )

    with col2:
        st.metric(
            "Pages",
            total_pages
        )

    with col3:
        st.metric(
            "Chunks",
            len(chunks)
        )

    with col4:
        st.metric(
            "Size MB",
            round(total_size / 1024 / 1024, 2)
        )
    st.subheader("💡 Suggested Questions")

    s1,s2,s3,s4 = st.columns(4)

    with s1:
         st.info("📄 Summarize all PDFs")

    with s2:
         st.info("🧠 Extract Insights")

    with s3:
         st.info("📊 Compare Documents")

    with s4:
         st.info("🎯 Main Conclusions") 

# -----------------------
# CHAT HISTORY
# -----------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )

# -----------------------
# CHAT
# -----------------------

if st.session_state.pdf_loaded:

    st.subheader("⚡ Quick Actions")

    st.subheader("🎤 Voice Assistant")

    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=True
    )

    if audio:
        st.success("🎤 Voice Recorded Successfully")

    col1, col2, col3 = st.columns(3)

    quick_question = None

    with col1:

        if st.button("📄 Summarize"):

            quick_question = """
Create an executive summary.

Include:
- Main topics
- Findings
- Conclusions
- Action items
"""

    with col2:

        if st.button("🧠 Key Points"):

            quick_question = """
Extract the most important insights.

Rank them by importance.

Use bullet points.
"""

    with col3:

        if st.button("📊 Compare"):

            quick_question = """
Compare all uploaded PDFs.

Show:
- Similarities
- Differences
- Unique information
- Final assessment
"""

    user_input = st.chat_input(
        "Ask anything about your PDFs..."
    )

    question = (
        user_input
        if user_input
        else quick_question
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

        docs = st.session_state.db.similarity_search(
            question,
            k=4
        )

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        with st.expander(
            "📄 Retrieved Context"
        ):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(
                    doc.page_content[:500]
                )

        with st.spinner(
            "Thinking..."
        ):

            conversation = [
                {
                    "role":"system",
                    "content":f"""
You are AI PDF Assistant Pro.

Use ONLY the provided document context.

Rules:

- Never hallucinate.
- Never use outside knowledge.
- If information is unavailable, clearly say so.
- Use bullet points whenever possible.
- Use headings.
- Be concise and professional.
- Think like a research analyst.

Document Context:

{context}
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
                "role":"assistant",
                "content":answer
            }
        )

        with st.chat_message(
            "assistant"
        ):
            st.write(answer)

# -----------------------
# SIDEBAR
# -----------------------

st.sidebar.title("🚀 Control Center")

st.sidebar.success("🟢 AI Online")

st.sidebar.markdown("""
### Features

📚 Multi PDF Upload

🧠 RAG Search

📄 Summaries

📊 Document Comparison

🔍 Source Viewer

🎤 Voice Assistant

---

### Example Questions

• Summarize this document

• What are the key findings?

• Compare all PDFs

• What conclusions are mentioned?
""")

st.sidebar.write(
    f"💬 Messages: {len(st.session_state.messages)}"
)

if st.sidebar.button(
    "🗑 Clear Chat"
):

    st.session_state.messages = []
    st.rerun()