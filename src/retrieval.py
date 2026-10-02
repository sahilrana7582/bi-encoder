from src.embedding import embed_query
from src.similarity import cosine_similarity


def retrieve(query, model, documents, embeddings, top_k=5):

    query_embedding = embed_query(model, query)

    results = []

    for index, document_embedding in enumerate(embeddings):

        score = cosine_similarity(
            query_embedding,
            document_embedding
        )

        results.append({
            "index": index,
            "score": float(score),
            "document": documents[index]
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]