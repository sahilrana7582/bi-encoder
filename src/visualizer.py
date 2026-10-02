import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_embeddings(
    embeddings,
    labels,
    query_embedding=None,
    query_label="Query"
):

    vectors = embeddings

    if query_embedding is not None:
        vectors = list(embeddings) + [query_embedding]

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(vectors)

    document_points = reduced[:len(embeddings)]

    plt.figure(figsize=(12, 8))

    plt.scatter(
        document_points[:, 0],
        document_points[:, 1]
    )

    for i, label in enumerate(labels):

        plt.annotate(
            label,
            (
                document_points[i, 0],
                document_points[i, 1]
            )
        )

    if query_embedding is not None:

        query_point = reduced[-1]

        plt.scatter(
            query_point[0],
            query_point[1],
            marker="X",
            s=150
        )

        plt.annotate(
            query_label,
            (
                query_point[0],
                query_point[1]
            )
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Document Embeddings")

    plt.show()