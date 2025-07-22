# build_vector_store.py

import os
import pickle
import faiss
from extract import extract_text_from_pdf
from preprocess import get_chunks
from sentence_transformers import SentenceTransformer
import numpy as np

def build_vector_store(pdf_path, save_dir="vector_store"):
    print("🔍 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("✂️ Splitting into chunks...")
    chunks = get_chunks(raw_text)

    print("📐 Loading embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks)

    print("🧠 Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 Saving index to {save_dir}/faiss.index...")
    faiss.write_index(index, f"{save_dir}/faiss.index")
    
    print(f"💾 Saving chunks to {save_dir}/chunks.pkl...")
    with open(f"{save_dir}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector store generated successfully!")

if __name__ == "__main__":
    build_vector_store("HSC26-Bangla1st-Paper.pdf")
