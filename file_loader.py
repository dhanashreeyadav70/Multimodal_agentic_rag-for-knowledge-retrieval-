import os
import pandas as pd
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from ingestion import load_json

# ✅ REMOVE HARDCODED PATH
# pytesseract will work automatically on Streamlit cloud

def load_file(file_path, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------- PDF ----------
    if ext == ".pdf":
        doc = fitz.open(file_path)
        documents = []

        for i, page in enumerate(doc):
            text = page.get_text()

            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i}
                ))

        return documents

    # ---------- TEXT ----------
    elif ext == ".txt":
        return TextLoader(file_path).load()

    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()

    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path).load()

    elif ext == ".json":
        return load_json(file_path)

    # ---------- CSV ----------
    elif ext == ".csv":
        df = pd.read_csv(file_path)

        documents = [Document(
            page_content=f"Columns: {', '.join(df.columns)}",
            metadata={"source": filename}
        )]

        for _, row in df.iterrows():
            row_text = ", ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
            )
            documents.append(Document(page_content=row_text))

        return documents

    # ---------- IMAGE ----------
    elif ext in [".png", ".jpg", ".jpeg"]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return [Document(page_content=text or "No text found")]

    # ❌ REMOVE AUDIO & VIDEO (NOT STREAMLIT FRIENDLY)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
