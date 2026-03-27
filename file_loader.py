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
import speech_recognition as sr
from faster_whisper import WhisperModel

# ✅ Load once (global)
whisper_model = WhisperModel("base", compute_type="int8")


# ✅ Speech recognizer
recognizer = sr.Recognizer()

# ✅ Initialize EasyOCR once
reader = easyocr.Reader(['en'], gpu=False)





# -----------------------------
# 🎤 AUDIO TRANSCRIPTION (WHISPER)
# -----------------------------
def transcribe_audio(file_path):
    try:
        segments, _ = whisper_model.transcribe(file_path)

        text = " ".join([segment.text for segment in segments])

        return text if text.strip() else "No speech detected"

    except Exception as e:
        return f"Audio transcription failed: {str(e)}"


# -----------------------------
# 🎬 VIDEO → AUDIO → TEXT
# -----------------------------
def transcribe_video(file_path):
    try:
        video = VideoFileClip(file_path)

        audio_path = file_path + "_temp.wav"
        video.audio.write_audiofile(audio_path)

        text = transcribe_audio(audio_path)

        os.remove(audio_path)

        return text

    except Exception as e:
        return f"Video transcription failed: {str(e)}"

def transcribe_audio_chunks(file_path, chunk_length=60):
    import ffmpeg

    try:
        output_pattern = "chunk_%03d.wav"

        (
            ffmpeg
            .input(file_path)
            .output(output_pattern, f='segment', segment_time=chunk_length)
            .run(quiet=True, overwrite_output=True)
        )

        texts = []

        for file in sorted(os.listdir()):
            if file.startswith("chunk_") and file.endswith(".wav"):
                segments, _ = whisper_model.transcribe(file)
                texts.append(" ".join([s.text for s in segments]))
                os.remove(file)

        return " ".join(texts)

    except Exception as e:
        return f"Chunk transcription failed: {str(e)}"

# def load_file(file_path, filename):

#     ext = os.path.splitext(filename)[1].lower()

#     # ---------- PDF ----------
#     if ext == ".pdf":
#         doc = fitz.open(file_path)
#         documents = []

#         for i, page in enumerate(doc):
#             text = page.get_text()

#             if text.strip():
#                 documents.append(Document(
#                     page_content=text,
#                     metadata={"source": filename, "page": i}
#                 ))

#         return documents

#     # ---------- TEXT ----------
#     elif ext == ".txt":
#         return TextLoader(file_path).load()

#     elif ext == ".docx":
#         return Docx2txtLoader(file_path).load()

#     elif ext == ".html":
#         return UnstructuredHTMLLoader(file_path).load()

#     elif ext == ".json":
#         return load_json(file_path)

#     # ---------- CSV ----------
#     elif ext == ".csv":
#         df = pd.read_csv(file_path)

#         documents = [Document(
#             page_content=f"Columns: {', '.join(df.columns)}",
#             metadata={"source": filename}
#         )]

#         for _, row in df.iterrows():
#             row_text = ", ".join(
#                 [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
#             )
#             documents.append(Document(page_content=row_text))

#         return documents

#     # ---------- IMAGE (EasyOCR) ----------
#     elif ext in [".png", ".jpg", ".jpeg"]:

#         result = reader.readtext(file_path)

#         extracted_text = " ".join([text for (_, text, _) in result])

#         if not extracted_text.strip():
#             extracted_text = "No text found in image"

#         return [Document(
#             page_content=extracted_text,
#             metadata={"source": filename}
#         )]

#     else:
#         raise ValueError(f"Unsupported file type: {ext}")

# -----------------------------
# 📂 MAIN LOADER
# -----------------------------
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
                    metadata={"source": filename, "page": i, "type": "pdf"}
                ))

        return documents

    elif ext == ".pdf":
        doc = fitz.open(file_path)
        documents = []
    
        for i, page in enumerate(doc):
    
            # Try normal text extraction
            text = page.get_text()
    
            # ✅ If text exists → use it
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": i, "type": "pdf"}
                ))
    
            else:
                # 🔥 OCR fallback
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
                img_np = np.array(img)
    
                ocr_result = reader.readtext(img_np)
    
                extracted_text = " ".join([t for (_, t, _) in ocr_result])
    
                if extracted_text.strip():
                    documents.append(Document(
                        page_content=extracted_text,
                        metadata={"source": filename, "page": i, "type": "pdf_ocr"}
                    ))
    
        return documents

    # ---------- TEXT ----------
    elif ext == ".txt":
        docs = TextLoader(file_path).load()
        for d in docs:
            d.metadata["type"] = "text"
        return docs

    elif ext == ".docx":
        docs = Docx2txtLoader(file_path).load()
        for d in docs:
            d.metadata["type"] = "docx"
        return docs

    elif ext == ".html":
        docs = UnstructuredHTMLLoader(file_path).load()
        for d in docs:
            d.metadata["type"] = "html"
        return docs

    elif ext == ".json":
        docs = load_json(file_path)
        for d in docs:
            d.metadata["type"] = "json"
        return docs

    # ---------- CSV ----------
    elif ext == ".csv":
        df = pd.read_csv(file_path)

        documents = [Document(
            page_content=f"Columns: {', '.join(df.columns)}",
            metadata={"source": filename, "type": "csv"}
        )]

        for _, row in df.iterrows():
            row_text = ", ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
            )
            documents.append(Document(page_content=row_text))

        return documents

    # ---------- IMAGE ----------
    elif ext in [".png", ".jpg", ".jpeg"]:

        result = reader.readtext(file_path)
        extracted_text = " ".join([text for (_, text, _) in result])

        if not extracted_text.strip():
            extracted_text = "No text found in image"

        return [Document(
            page_content=extracted_text,
            metadata={"source": filename, "type": "image"}
        )]

    # ---------- AUDIO ----------
    elif ext in [".wav", ".mp3"]:

        text = transcribe_audio(file_path)

        return [Document(
            page_content=text,
            metadata={"source": filename, "type": "audio"}
        )]

    # ---------- VIDEO ----------
    elif ext in [".mp4", ".avi", ".mov"]:

        text = transcribe_video(file_path)

        return [Document(
            page_content=text,
            metadata={"source": filename, "type": "video"}
        )]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
