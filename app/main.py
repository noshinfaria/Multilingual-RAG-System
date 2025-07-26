from fastapi import FastAPI
from langdetect import detect
from dotenv import load_dotenv

from app.memory import ChatMemory
from app.models import QueryRequest, QueryResponse
from app.agent import get_top_k_docs, generate_answer
from app.evaluation import evaluate_rag
load_dotenv()

app = FastAPI(
    title="RAG Question Answering API",
    description="Ask questions and get answers from retrieved context using RAG pipeline.",
    version="1.0.0"
)
chat_memory = ChatMemory()



# === API Endpoint ===
@app.post("/ask", response_model=QueryResponse, summary="Ask a Question")
def ask_question(req: QueryRequest):
    # lang = detect(req.query)
    # print("User query language:", lang)
    long_context = get_top_k_docs(req.query)
    print(long_context)
    short_context = chat_memory.get()
    answer = generate_answer(req.query, long_context, short_context)
    chat_memory.add(req.query, answer)

    eval_result = evaluate_rag(
        query=req.query,
        answer=answer,
        retrieved_contexts=long_context
    )
    return {"query": req.query, "answer": answer, "evaluation": eval_result}
