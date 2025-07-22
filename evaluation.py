from sklearn.metrics.pairwise import cosine_similarity

def evaluate_similarity(query, answer, context):
    query_vec = model.encode([query])
    ctx_vecs = model.encode(context)
    avg_ctx_vec = np.mean(ctx_vecs, axis=0).reshape(1, -1)
    score = cosine_similarity(query_vec, avg_ctx_vec)[0][0]
    return score
