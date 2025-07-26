import faiss, pickle
from sentence_transformers import SentenceTransformer

# Load KB
chunks = pickle.load(open("vector_store/chunks.pkl", "rb"))
index = faiss.read_index("vector_store/faiss.index")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")