import os
import pandas as pd
from PIL import Image
import pytesseract
import cv2
import fitz  # PyMuPDF

from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from ingestion import load_json

# -------------------------------
# SAFE OCR FUNCTION
# -------------------------------
def safe_ocr(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def load_file(file_path, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------- PDF ----------
    if ext == ".pdf":

        doc = fitz.open(file_path)
        documents = []

        for i, page in enumerate(doc):

            text = page.get_text()

            # OCR fallback (SAFE)
            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                ocr_text = safe_ocr(img)

                if ocr_text.strip():
                    text = ocr_text
                else:
                    text = "Scanned PDF page (no readable text detected)"

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

        image = Image.open(file_path)

        text = safe_ocr(image)

        if not text.strip():
            text = "Image uploaded. No readable text detected."

        return [Document(
            page_content=text,
            metadata={"source": filename}
        )]

    # ---------- VIDEO ----------
    elif ext in [".mp4", ".avi"]:

        cap = cv2.VideoCapture(file_path)
        texts = []
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % 30 == 0:
                img = Image.fromarray(frame)
                text = safe_ocr(img)

                if text.strip():
                    texts.append(text)

            count += 1

        cap.release()

        return [Document(
            page_content=" ".join(texts) if texts else "No readable content in video",
            metadata={"source": filename}
        )]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
