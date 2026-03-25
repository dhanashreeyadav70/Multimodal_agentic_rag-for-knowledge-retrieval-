from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):

    # ✅ Do NOT chunk structured data
    if docs and "Employee record" in docs[0].page_content:
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(docs)