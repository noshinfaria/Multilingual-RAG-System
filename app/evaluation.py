from sklearn.metrics.pairwise import cosine_similarity
from app.config import model
import numpy as np

def evaluate_relevance(query, retrieved_contexts, top_k=3):
    query_vec = model.encode([query])
    ctx_vecs = model.encode(retrieved_contexts)
    similarities = cosine_similarity(query_vec, ctx_vecs)[0]
    top_similarities = sorted(similarities, reverse=True)[:top_k]
    avg_score  = float(np.mean(top_similarities))
    return avg_score 


def evaluate_groundedness(answer, retrieved_contexts):
    answer_vec = model.encode([answer])
    ctx_vecs = model.encode(retrieved_contexts)
    similarities = cosine_similarity(answer_vec, ctx_vecs)[0]
    max_score = float(np.max(similarities))
    avg_score = float(np.mean(similarities))
    return {"max_groundedness": max_score, "avg_groundedness": avg_score}



def evaluate_rag(query, answer, retrieved_contexts):
    relevance_score = evaluate_relevance(query, retrieved_contexts)
    groundedness_scores = evaluate_groundedness(answer, retrieved_contexts)
    # robust_metrics = evaluate_robust_metrics(gold_answer, generated_answer, lang=lang)

    return {
        "query": query,
        "answer": answer,
        "relevance_score": relevance_score,
        "groundedness": groundedness_scores
    }


# # --- Text similarity metrics ---

# def evaluate_robust_metrics(gold_answer, generated_answer, lang="en"):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     rouge_scores = scorer.score(gold_answer, generated_answer)
#     bleu_score = sacrebleu.corpus_bleu([generated_answer], [[gold_answer]])
#     P, R, F1 = bert_score.score([generated_answer], [gold_answer], lang=lang, verbose=False)

#     return {
#         "rouge1_f1": rouge_scores['rouge1'].fmeasure,
#         "rougeL_f1": rouge_scores['rougeL'].fmeasure,
#         "bleu": bleu_score.score,
#         "bertscore_f1": F1.mean().item()
#     }