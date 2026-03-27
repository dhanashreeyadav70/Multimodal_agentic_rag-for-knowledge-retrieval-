import os
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
import easyocr
import numpy as np
from moviepy.editor import VideoFileClip
from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from ingestion import load_json

# ✅ Initialize EasyOCR once
reader = easyocr.Reader(['en'], gpu=False)


def load_file(file_path, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------- PDF ----------
    if ext == ".pdf":
        doc = fitz.open(file_path)
        documents = []

        for i, page in enumerate(doc):
            text = page.get_text()

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

    # ---------- IMAGE (EasyOCR) ----------
    elif ext in [".png", ".jpg", ".jpeg"]:

        result = reader.readtext(file_path)

        extracted_text = " ".join([text for (_, text, _) in result])

        if not extracted_text.strip():
            extracted_text = "No text found in image"

        return [Document(
            page_content=extracted_text,
            metadata={"source": filename}
        )]


    else:
        raise ValueError(f"Unsupported file type: {ext}")
