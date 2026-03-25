import os
import pandas as pd
from PIL import Image
import pytesseract

import cv2
import fitz  # PyMuPDF

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from ingestion import load_json

# ✅ SET PATH (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\16020\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def load_file(file_path, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------- PDF ----------
    if ext == ".pdf":

        doc = fitz.open(file_path)
        documents = []

        for i, page in enumerate(doc):

            text = page.get_text()

            # OCR fallback
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
        text = pytesseract.image_to_string(image)

        return [Document(page_content=text or "No text found")]
    elif ext in [".png", ".jpg", ".jpeg"]:
        return [Document(
    page_content="Image uploaded. OCR disabled in cloud deployment.",
    metadata={"source": filename}
    )]
    # elif ext in [".png", ".jpg", ".jpeg"]:

    #     processed = preprocess_image(file_path)
    #     text = pytesseract.image_to_string(processed)

    #     if not text.strip():
    #         text = "Image uploaded. No readable text detected. Likely a photo."

    #     return [Document(page_content=text)]

    # else:
    #     raise ValueError(f"Unsupported file type: {ext}")

    # ---------- AUDIO ----------
    # elif ext in [".mp3", ".wav"]:
    #     model = whisper.load_model("base")
    #     result = model.transcribe(file_path)

    #     return [Document(page_content=result["text"])]
    # elif ext in [".mp3", ".wav"]:
    #     return [Document(
    #     page_content="Audio file uploaded. Transcription not supported in this deployment.",
    #     metadata={"source": filename, "type": "audio"}
    # )]

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
                text = pytesseract.image_to_string(img)
                if text.strip():
                    texts.append(text)

            count += 1

        cap.release()

        return [Document(page_content=" ".join(texts))]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
