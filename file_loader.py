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

    # ---------- VIDEO ----------
    elif ext in [".mp4", ".avi", ".mov"]:
    
        texts = []
    
        try:
            # 🎥 Load video
            clip = VideoFileClip(file_path)
    
            # ---------- AUDIO TO TEXT ----------
            if clip.audio is not None:
                audio_path = file_path + "_audio.wav"
                clip.audio.write_audiofile(audio_path)
    
                import speech_recognition as sr
    
                r = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = r.record(source)
    
                    try:
                        text = r.recognize_google(audio_data)
                        texts.append("Audio Transcript: " + text)
                    except:
                        texts.append("Audio could not be transcribed")
    
            # ---------- FRAME OCR ----------
            duration = int(clip.duration)
    
            for t in range(0, duration, max(1, duration // 5)):  # sample ~5 frames
                frame = clip.get_frame(t)
    
                from PIL import Image
                img = Image.fromarray(frame)
    
                result = reader.readtext(np.array(img))
    
                frame_text = " ".join([text for (_, text, _) in result])
    
                if frame_text.strip():
                    texts.append(f"Frame {t}s: {frame_text}")
    
            clip.close()
    
        except Exception as e:
            texts.append(f"Video processing error: {str(e)}")
    
        final_text = "\n".join(texts)
    
        if not final_text.strip():
            final_text = "No useful content extracted from video"
    
        return [Document(
            page_content=final_text,
            metadata={"source": filename}
        )]

    else:
        raise ValueError(f"Unsupported file type: {ext}")
