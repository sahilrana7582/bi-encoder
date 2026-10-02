from src.embedding import (
    load_model,
    load_documents,
    create_embeddings,
)
from src.llm import (
    create_llm_model, get_settings
)
from src.retrieval import retrieve
from src.visualizer import visualize_embeddings


DOCUMENT_PATH = "data/documents.txt"
QUERY = "How does a Kafka consumer keep track of its position?"


def main():
    settings = get_settings()
    llm_model = create_llm_model(settings=settings)
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

    results = retrieve(
        query=QUERY,
        model=model,
        documents=documents,
        embeddings=embeddings,
        top_k=5,
    )

    print("=" * 80)
    print("QUERY")
    print("=" * 80)
    print(QUERY)

    print("\n" + "=" * 80)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank       : {rank}")
        print(f"Document   : {result['index']}")
        print(f"Similarity : {result['score']:.4f}")
        print("-" * 80)
        print(result["document"])

    # Visualize document embeddings + query embedding
    labels = [
        f"Doc {index}"
        for index in range(len(documents))
    ]

    visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        query_embedding=query_embedding,
        query_label="Query",
    )


if __name__ == "__main__":
    main()