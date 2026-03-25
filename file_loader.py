import os
import pandas as pd
from PIL import Image
import pytesseract
import shutil

import fitz  # PyMuPDF

from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, UnstructuredHTMLLoader
)
from langchain_core.documents import Document
from ingestion import load_json


# ✅ Auto-detect tesseract (works locally + cloud)
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

            # OCR fallback (SAFE)
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
                text = "Image uploaded. No readable text detected."

        except:
            text = "Image uploaded. OCR not supported in this environment."

        return [Document(page_content=text, metadata={"source": filename})]

    # ---------- AUDIO ----------
    elif ext in [".mp3", ".wav"]:
        return [Document(
            page_content="Audio uploaded. Transcription disabled in cloud deployment.",
            metadata={"source": filename}
        )]

    # ---------- VIDEO ----------
    # elif ext in [".mp4", ".avi"]:
    #     return [Document(
    #         page_content="Video uploaded. Processing disabled in cloud deployment.",
    #         metadata={"source": filename}
    #     )]
    elif ext in [".mp4", ".avi"]:
        return [Document(
        page_content="""
Video file uploaded.

⚠️ Direct video processing is not supported in this deployment.

To analyze this video, you can:
1. Upload subtitles (.srt) or transcript
2. Provide a summary of the video
3. Upload extracted frames/images
4. Upload audio separately (.mp3/.wav)

If transcript is provided, I can summarize, analyze, and extract insights.
""",
        metadata={"source": filename, "type": "video"}
    )]

    # ---------- FALLBACK ----------
    else:
        return [Document(
            page_content=f"Unsupported file type: {ext}",
            metadata={"source": filename}
        )]
