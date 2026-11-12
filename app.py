import csv
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from pdf_qa import build_vector_store, answer_question

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
UPLOAD_DIR = DATASET_DIR / "uploaded_files"
CHROMA_DIR = BASE_DIR / "chroma_db"
for folder in (DATASET_DIR, UPLOAD_DIR, CHROMA_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def append_csv(path, headers, values):
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(headers)
        writer.writerow(values)


st.set_page_config(page_title="PDF Chat Assistant", page_icon="📘", layout="wide")

with st.sidebar:
    logo = BASE_DIR / "assets" / "logo.png"
    if logo.exists():
        st.image(str(logo), width=90)
    st.title("PDF Chat Assistant")
    st.caption("Ask questions from your PDF documents.")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("📘 Chat with your PDF")
st.write("Upload a PDF in the sidebar, then ask a question.")

for key, default in [
    ("messages", []), ("vector_store", None),
    ("active_file", None), ("page_count", 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

if uploaded is not None:
    if not uploaded.name.lower().endswith(".pdf"):
        st.error("Only PDF files are allowed.")
        st.stop()

    if st.session_state.active_file != uploaded.name:
        safe_name = Path(uploaded.name).name
        pdf_path = UPLOAD_DIR / safe_name
        pdf_path.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Reading PDF and creating searchable chunks..."):
                page_count = len(PdfReader(str(pdf_path)).pages)
                db_path = CHROMA_DIR / Path(safe_name).stem
                vector_store, _, chunk_count = build_vector_store(pdf_path, db_path)
            st.session_state.vector_store = vector_store
            st.session_state.active_file = uploaded.name
            st.session_state.page_count = page_count
            st.session_state.messages = []
            st.success(f"PDF processed: {chunk_count} chunks created.")
        except Exception as error:
            st.error(f"PDF processing failed: {error}")
            st.info("Check that Ollama is running and both required models are installed.")
            st.stop()

if st.session_state.active_file:
    st.info(
        f"**Uploaded PDF:** {st.session_state.active_file}  |  "
        f"**Total pages:** {st.session_state.page_count}"
    )
else:
    st.warning("Upload a PDF to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Show retrieved document chunks"):
                for i, doc in enumerate(message["sources"], 1):
                    page = doc.metadata.get("page", "Unknown")
                    page = int(page) + 1 if str(page).isdigit() else page
                    st.markdown(f"**Chunk {i} — Page {page}**")
                    st.write(doc.page_content)

question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=st.session_state.vector_store is None
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                answer, sources = answer_question(st.session_state.vector_store, question)
            elapsed = time.perf_counter() - start
            st.markdown(answer)
            st.caption(f"Response generation time: {elapsed:.2f} seconds")
            with st.expander("Show retrieved document chunks"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "Unknown")
                    page = int(page) + 1 if str(page).isdigit() else page
                    st.markdown(f"**Chunk {i} — Page {page}**")
                    st.write(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })
            timestamp = datetime.now().isoformat(timespec="seconds")
            append_csv(DATASET_DIR / "queries.csv",
                       ["timestamp", "pdf_filename", "question"],
                       [timestamp, st.session_state.active_file, question])
            append_csv(DATASET_DIR / "answers.csv",
                       ["timestamp", "pdf_filename", "question", "answer", "time_seconds"],
                       [timestamp, st.session_state.active_file, question, answer, f"{elapsed:.2f}"])
        except Exception as error:
            st.error(f"Could not generate an answer: {error}")
