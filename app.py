import streamlit as st
import tempfile
import uuid

from file_loader import load_file
from chunker import split_documents
from vector_store import create_vector_store
from retrieval import HybridRetriever
from workflow import build_graph

st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("🧠 Agentic RAG AI Assistant")

# ✅ SESSION ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# Upload
uploaded_file = st.file_uploader(
    "Upload file",
    type=["pdf","docx","txt","csv","json","html","png","jpg","jpeg"]
)

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name

        docs = load_file(file_path, uploaded_file.name)
        docs = split_documents(docs)

        if not docs or all(not d.page_content.strip() for d in docs):
            st.error("No readable content found")
            st.stop()

        vector_db = create_vector_store(docs, uploaded_file.name)
        retriever = HybridRetriever(docs, vector_db)

        st.session_state.retriever = retriever
        st.success("File processed successfully")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Chat
query = st.chat_input("Ask something...")

if query:

    if st.session_state.retriever is None:
        st.warning("Upload a file first")
    else:
        st.chat_message("user").write(query)

        result = st.session_state.graph.invoke({
            "query": query,
            "retriever": st.session_state.retriever,
            "session_id": st.session_state.session_id
        })

        answer = result.get("answer", "No response")

        st.chat_message("assistant").write(answer)
