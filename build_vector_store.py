# build_vector_store.py

import os
import pickle
import faiss
import re
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from typing import List
from sentence_transformers import SentenceTransformer
import pytesseract
import numpy as np
import pdfplumber

# from preprocess import get_chunks
# from extract import extract_text_from_pdf


# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 


def get_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    sentences = re.split(r'(?<=[।!?])\s+', text)  # split on Bengali sentence enders
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "

    if chunk:
        chunks.append(chunk.strip())

    # Add overlap
    final_chunks = []
    for i in range(0, len(chunks)):
        start = max(i - 1, 0)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined)

    return final_chunks

def clean_text(text: str) -> str:
    # Remove unwanted characters
    text = text.replace("\n", " ")  # Join broken lines
    text = re.sub(r"[^\u0980-\u09FF\s।!?]", "", text)  # Keep only Bengali characters and punctuations
    # text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = text.strip()
    return text



def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # Render page as image
        img = Image.open(BytesIO(pix.tobytes("png")))

        # OCR using Bengali language
        text = pytesseract.image_to_string(img, lang="ben")
        clean = clean_text(text)
        full_text += f"{clean.strip()} "
        # full_text += f"{text.strip()}"

    return full_text.strip()


def build_vector_store(pdf_path, save_dir="vector_store"):
    print(" Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print(" Splitting into chunks...")
    chunks = get_chunks(raw_text)

    print(" Loading embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks)

    print(" Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs(save_dir, exist_ok=True)
    
    print(f" Saving index to {save_dir}/faiss.index...")
    faiss.write_index(index, f"{save_dir}/faiss.index")
    
    print(f" Saving chunks to {save_dir}/chunks.pkl...")
    with open(f"{save_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(" Vector store generated successfully!")

if __name__ == "__main__":
    build_vector_store("HSC26-Bangla1st-Paper.pdf")
