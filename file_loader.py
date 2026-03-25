import os
import pandas as pd
from PIL import Image
import pytesseract
import shutil
import fitz

from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from ingestion import load_json

# Auto detect tesseract
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def load_file(file_path, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------- PDF ----------
    if ext == ".pdf":
        doc = fitz.open(file_path)
        documents = []

        for i, page in enumerate(doc):
            text = page.get_text()

            if not text.strip():
                try:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                except:
                    text = ""

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
        documents = []

        documents.append(Document(
            page_content=f"Columns: {', '.join(df.columns)}",
            metadata={"source": filename}
        ))

        for _, row in df.iterrows():
            row_text = ", ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
            )
            documents.append(Document(page_content=row_text))

        return documents

    # ---------- IMAGE ----------
    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            if not text.strip():
                text = "No readable text detected in image."
        except:
            text = "Image uploaded. OCR not supported."

        return [Document(page_content=text, metadata={"source": filename})]

    # ---------- VIDEO ----------
    elif ext in [".mp4", ".avi"]:
        return [Document(
            page_content="Video uploaded",
            metadata={
                "source": filename,
                "type": "video"
            }
        )]

    # ---------- AUDIO ----------
    elif ext in [".mp3", ".wav"]:
        return [Document(
            page_content="AUDIO_FILE",
            metadata={"source": filename, "type": "audio"}
        )]

    # ---------- FALLBACK ----------
    else:
        return [Document(
            page_content=f"Unsupported file type: {ext}",
            metadata={"source": filename}
        )]
