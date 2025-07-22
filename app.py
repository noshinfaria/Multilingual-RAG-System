from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss, pickle, os
import numpy as np
from extract import extract_text_from_pdf
from memory import ChatMemory
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
app = FastAPI()
chat_memory = ChatMemory()

# Load KB
chunks = pickle.load(open("vector_store/chunks.pkl", "rb"))
index = faiss.read_index("vector_store/faiss.index")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

class QueryRequest(BaseModel):
    query: str


# # === Load and Preprocess the Document ===
# print("Loading PDF and splitting text...")
# raw_text = extract_text_from_pdf("HSC26-Bangla1st-Paper.pdf")

# # Text chunking
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
#     separators=["\n", "।", "."]
# )
# chunks = splitter.split_text(raw_text)

# # === Vector Embedding and Indexing ===
# print("Embedding and indexing...")
# vectors = model.encode(chunks)

# dimension = vectors.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(vectors)

# === Retrieval Function ===
def get_top_k_docs(query, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# === Generation Function ===
def generate_answer(query, long_term_context, short_term_context):
    context = "\n".join(short_term_context + long_term_context)
    prompt = f"""Answer the question based only on the context provided.
Context:
{context}

Question: {query}
Answer:"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# === API Endpoint ===
@app.post("/ask")
def ask_question(req: QueryRequest):
    long_context = get_top_k_docs(req.query)
    print(long_context)
    short_context = chat_memory.get()
    answer = generate_answer(req.query, long_context, short_context)
    chat_memory.add(req.query, answer)
    return {"query": req.query, "answer": answer}
