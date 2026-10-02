import numpy as np
from numpy.typing import NDArray


def dot_product(a: NDArray[np.float32], b: NDArray[np.float32]) -> np.float32:
    return np.dot(a, b)


def vector_norm(vector: NDArray[np.float32]) -> np.float32:
    return np.linalg.norm(vector)


def cosine_similarity(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
) -> np.float32:
    dot = dot_product(a, b)

    norm_a = vector_norm(a)
    norm_b = vector_norm(b)

    return dot / (norm_a * norm_b)