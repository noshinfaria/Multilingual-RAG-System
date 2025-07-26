
# from openai import OpenAI
from dotenv import load_dotenv
from app.config import model, index, chunks
import openai

# load_dotenv()

client = openai.OpenAI()


# === Retrieval Function ===
def get_top_k_docs(query, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# === Generation Function ===
def generate_answer(query, long_term_context, short_term_context):
    context = "\n".join(short_term_context + long_term_context)
    prompt = f"""Answer the following questions based only on the given context.

Context:
সে তার মামার বাড়িতে বেড়াতে গিয়েছিল। তার মামা ছিলেন একজন বুদ্ধিমান মানুষ।

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
A: মামাকে
Context:
{context}

Question: {query}
Answer:"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives short, context-based answers in Bengali. Respond only using the context."},
                {"role": "user", "content": prompt}
            ]
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"