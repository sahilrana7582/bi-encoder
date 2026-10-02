from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_documents(path):
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    documents = [
        doc.strip()
        for doc in content.split("\n\n")
        if doc.strip()
    ]

    return documents

def create_embeddings(model, documents):
    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    )

    return embeddings