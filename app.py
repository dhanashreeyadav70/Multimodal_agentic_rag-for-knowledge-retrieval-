# import streamlit as st
# import tempfile
# import uuid

# from file_loader import load_file
# from chunker import split_documents
# from vector_store import create_vector_store
# from retrieval import HybridRetriever
# from workflow import build_graph

# st.set_page_config(page_title="Agentic RAG", layout="wide")

# st.title("🧠 Agentic RAG PoC for Intelligent Knowledge Retrieval")

# # -------------------------
# # SESSION INIT
# # -------------------------
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())

# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# if "graph" not in st.session_state:
#     st.session_state.graph = build_graph()

# if "file_processed" not in st.session_state:
#     st.session_state.file_processed = False


# # -------------------------
# # FILE UPLOAD
# # -------------------------
# # uploaded_file = st.file_uploader(
# #     "Upload file",
# #     type=["pdf","docx","txt","csv","json","html","png","jpg","jpeg"]
# # )

# uploaded_file = st.file_uploader(
#     "Upload file",
#     type=[
#         "pdf","docx","txt","csv","json","html",
#         "png","jpg","jpeg",
#         "mp4","avi","mov",
#         "mp3","wav"   # ✅ ADD THIS
#     ]
# )


# if uploaded_file and not st.session_state.file_processed:

#     try:
#         with tempfile.NamedTemporaryFile(delete=False) as tmp:
#             tmp.write(uploaded_file.getbuffer())
#             file_path = tmp.name

#         st.info("📄 Processing file...")

#         docs = load_file(file_path, uploaded_file.name)

#         st.write(f"✅ Loaded {len(docs)} documents")

#         docs = split_documents(docs)

#         if not docs:
#             st.error("❌ No readable content found")
#             st.stop()

#         # 🔍 Show extracted text (important for debugging)
#         st.subheader("🔍 Extracted Content Preview")
#         st.write(docs[0].page_content[:500])

#         vector_db = create_vector_store(docs, uploaded_file.name)
#         retriever = HybridRetriever(docs, vector_db)

#         st.session_state.retriever = retriever
#         st.session_state.file_processed = True

#         st.success("✅ File processed successfully")

#     except Exception as e:
#         st.error(f"❌ Error: {str(e)}")


# # -------------------------
# # CHAT
# # -------------------------
# query = st.chat_input("Ask something...")

# if query:

#     if st.session_state.retriever is None:
#         st.warning("⚠️ Upload a file first")
#     else:
#         st.chat_message("user").write(query)

#         result = st.session_state.graph.invoke({
#             "query": query,
#             "retriever": st.session_state.retriever,
#             "session_id": st.session_state.session_id
#         })

#         answer = result.get("answer", "No response")

#         st.chat_message("assistant").write(answer)


import streamlit as st
import tempfile
import uuid

from file_loader import load_file
from chunker import split_documents
from vector_store import create_vector_store
from retrieval import HybridRetriever
from workflow import build_graph
from memory import chat_memory   # ✅ IMPORTANT

st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("🧠 Agentic RAG PoC for Intelligent Knowledge Retrieval")

# -------------------------
# SESSION INIT
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload file",
    type=[
        "pdf","docx","txt","csv","json","html",
        "png","jpg","jpeg",
        "mp4","avi","mov",
        "mp3","wav"
    ]
)

# 🔥 RESET LOGIC (MAIN FIX)
if uploaded_file:

    # Detect new file
    if st.session_state.current_file != uploaded_file.name:

        st.session_state.current_file = uploaded_file.name

        # 🔥 RESET EVERYTHING
        st.session_state.retriever = None
        st.session_state.file_processed = False

        # 🔥 CLEAR MEMORY
        chat_memory.clear()

        st.info("🔄 New file detected. Resetting knowledge base...")


# -------------------------
# FILE PROCESSING
# -------------------------
if uploaded_file and not st.session_state.file_processed:

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name

        st.info("📄 Processing file...")

        docs = load_file(file_path, uploaded_file.name)

        st.write(f"✅ Loaded {len(docs)} documents")

        docs = split_documents(docs)

        # ⚠️ SAFE CHECK (UPDATED)
        if not docs or docs[0].page_content.strip() == "":
            st.warning("⚠️ File has very little or no readable content")
            st.stop()

        # 🔍 DEBUG PREVIEW
        st.subheader("🔍 Extracted Content Preview")
        st.write(docs[0].page_content[:500])

        vector_db = create_vector_store(docs, uploaded_file.name)
        retriever = HybridRetriever(docs, vector_db)

        st.session_state.retriever = retriever
        st.session_state.file_processed = True

        st.success("✅ File processed successfully")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


# -------------------------
# CHAT
# -------------------------
query = st.chat_input("Ask something...")

if query:

    if st.session_state.retriever is None:
        st.warning("⚠️ Upload a file first")
    else:
        st.chat_message("user").write(query)

        result = st.session_state.graph.invoke({
            "query": query,
            "retriever": st.session_state.retriever,
            "session_id": st.session_state.session_id
        })

        answer = result.get("answer", "No response")

        st.chat_message("assistant").write(answer)
