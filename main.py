from src.embedding import (
    load_model,
    load_documents,
    create_embeddings,
)

from src.similarity import cosine_similarity


DOCUMENT_PATH = "data/documents.txt"
QUERY = "How does a Kafka consumer keep track of its position?"


def main():
    model = load_model()

    documents = load_documents(DOCUMENT_PATH)

    embeddings = create_embeddings(
        model,
        documents,
    )

    query_embedding = model.encode(
        QUERY,
        convert_to_numpy=True,
    )

    results = []

    for idx, embedding in enumerate(embeddings):
        score = cosine_similarity(
            query_embedding,
            embedding,
        )

        results.append((score, documents[idx]))

    results.sort(reverse=True)

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(QUERY)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for rank, (score, document) in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Similarity: {score:.4f}")
        print("-" * 80)
        print(document)


if __name__ == "__main__":
    main()