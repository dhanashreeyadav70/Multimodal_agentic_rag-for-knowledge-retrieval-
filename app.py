import streamlit as st
import tempfile

from file_loader import load_file
from chunker import split_documents
from vector_store import create_vector_store
from retrieval import HybridRetriever
from workflow import build_graph

st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("🧠 Agentic RAG PoC for Intelligent Knowledge Retrieval")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

uploaded_file = st.file_uploader(
    "Upload file",
    type=["pdf","docx","txt","csv","json","html","png","jpg","jpeg","mp3","wav","mp4","avi"]
)

if uploaded_file:

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name

        docs = load_file(file_path, uploaded_file.name)
        docs = split_documents(docs)

        # ✅ STRONG VALIDATION
        if not docs or all(not d.page_content.strip() for d in docs):
            st.error("❌ No readable content found (scanned/empty file)")
            st.stop()

        vector_db = create_vector_store(docs, uploaded_file.name)
        retriever = HybridRetriever(docs, vector_db)

        st.session_state.retriever = retriever

        st.success(f"{uploaded_file.name} loaded successfully")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

query = st.chat_input("Ask anything...")

if query:

    if st.session_state.retriever is None:
        st.warning("Upload file first")
    else:

        st.chat_message("user").write(query)

        result = st.session_state.graph.invoke({
            "query": query,
            "retriever": st.session_state.retriever
        })

        answer = result.get("answer", "No response")

        st.chat_message("assistant").write(answer)