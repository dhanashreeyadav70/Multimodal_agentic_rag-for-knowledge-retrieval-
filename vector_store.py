import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store(docs, filename):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ unique index per file
    safe_name = filename.replace(".", "_")
    FAISS_PATH = f"faiss_index_{safe_name}"

    if os.path.exists(FAISS_PATH):
        print(f"✅ Loading FAISS for {filename}")
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print(f"⚡ Creating FAISS for {filename}")

    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(FAISS_PATH)

    return vector_db
