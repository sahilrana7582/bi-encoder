from src.embedding import (
    load_model,
    load_documents,
    create_embeddings,
)
from src.llm import (
    create_llm_model, get_settings, format_context
)
from src.retrieval import retrieve
from src.visualizer import visualize_embeddings
from src.prompt import prompt


DOCUMENT_PATH = "data/documents.txt"
# QUERY = "How does Kubernetes maintain the used defined number of application instances while allowing changes to be introduced without replacing everything at once?"
QUERY = "What component is responsible for keeping application instances aligned with the intended state and handling changes to that state?"


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

    llm_context = format_context(results=results)
    formatted_prompt = prompt.invoke({
        "query": QUERY,
        "context": llm_context
    })

    response = llm_model.invoke(formatted_prompt)
    print(response)
    print("="*80)
    print("="*80)
    print(response.content)


    # Visualize document embeddings + query embedding
    # labels = [
    #     f"Doc {index}"
    #     for index in range(len(documents))
    # ]

    # visualize_embeddings(
    #     embeddings=embeddings,
    #     labels=labels,
    #     query_embedding=query_embedding,
    #     query_label="Query",
    # )


if __name__ == "__main__":
    main()