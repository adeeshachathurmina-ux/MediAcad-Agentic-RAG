"""
Mathematical Mechanics for Vector Retrieval and Explainability.
MediAcad Agentic RAG - Member 2 (Hybrid Retrieval & Mathematical Explainability).

This module provides pure NumPy implementations of foundational linear algebra
and vector space mathematics used in information retrieval (Salton et al., 1975).
It eliminates 'black-box' abstractions by providing transparent mathematical operations:
  1. Vector Magnitude (Euclidean / L2 Norm) : ||v|| = sqrt(sum(v_i^2))
  2. Dot Product (Inner Product)           : q . d = sum(q_i * d_i)
  3. L2 Unit Normalization                 : v_hat = v / ||v||_2
  4. Cosine Similarity                     : cos(theta) = (q . d) / (||q|| * ||d||)
  5. Min-Max Normalization                 : s_norm = (s - min) / (max - min)
  6. Hybrid Score Convex Combination       : S_hybrid = alpha * S_dense + (1 - alpha) * S_sparse
"""

from typing import Sequence, Union
import numpy as np


def ensure_numpy_vector(v: Union[np.ndarray, Sequence[float]], name: str = "vector") -> np.ndarray:
    """
    Validates and converts input into a 1-D float64 NumPy array.

    Raises:
        ValueError: If input is empty or has dimensionality greater than 1.
    """
    arr = np.asarray(v, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if arr.ndim != 1:
        arr = arr.flatten()
    return arr


def vector_magnitude(v: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Computes the Euclidean (L2) magnitude of a vector.

    Formula:
        ||v||_2 = \\sqrt{\\sum_{i=1}^{n} v_i^2}

    Geometric Interpretation:
        Represents the length of the vector in n-dimensional Euclidean space.
        Used to measure document and query term-weight intensity.

    Args:
        v: 1-D array or sequence of real numbers.

    Returns:
        Euclidean length as a float >= 0.0.
    """
    arr = ensure_numpy_vector(v, name="v")
    # Using np.sqrt(np.dot(v, v)) for maximum numerical precision and speed
    return float(np.sqrt(np.dot(arr, arr)))


def dot_product(
    q: Union[np.ndarray, Sequence[float]],
    d: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Computes the algebraic dot product (inner product) between two vectors.

    Formula:
        q \\cdot d = \\sum_{i=1}^{n} (q_i \\times d_i)

    Geometric Interpretation:
        Measures the directional projection of query vector q onto document
        vector d scaled by their lengths.

    Args:
        q: Query vector of length n.
        d: Document vector of length n.

    Returns:
        Scalar real value of the inner product.

    Raises:
        ValueError: If vectors do not have identical dimensionality.
    """
    q_arr = ensure_numpy_vector(q, name="q")
    d_arr = ensure_numpy_vector(d, name="d")

    if q_arr.shape[0] != d_arr.shape[0]:
        raise ValueError(
            f"Dimensionality mismatch in dot product: q has dimension {q_arr.shape[0]} "
            f"while d has dimension {d_arr.shape[0]}."
        )

    return float(np.dot(q_arr, d_arr))


def l2_normalize(
    v: Union[np.ndarray, Sequence[float]],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Projects a vector onto the unit hypersphere (L2 normalization).

    Formula:
        \\hat{v} = \\frac{v}{||v||_2 + \\epsilon}

    Properties:
        ||\\hat{v}||_2 = 1.0 (for non-zero vectors).
        After L2 normalization, the dot product between two vectors equals
        their cosine similarity: \\hat{q} \\cdot \\hat{d} = \\cos(\\theta).

    Args:
        v: 1-D input vector.
        eps: Small stability constant to prevent division by zero.

    Returns:
        1-D NumPy array of unit Euclidean length.
    """
    arr = ensure_numpy_vector(v, name="v")
    norm = vector_magnitude(arr)
    if norm < eps:
        # Zero vector remains a zero vector (avoids NaN)
        return np.zeros_like(arr)
    return arr / norm


def cosine_similarity(
    q: Union[np.ndarray, Sequence[float]],
    d: Union[np.ndarray, Sequence[float]],
    eps: float = 1e-12,
) -> float:
    """
    Calculates the Cosine Similarity between query vector q and document vector d.

    Formula:
        \\cos(\\theta) = \\frac{q \\cdot d}{||q||_2 \\times ||d||_2}

    Properties:
        Range: [-1.0, 1.0] (or [0.0, 1.0] for non-negative TF-IDF spaces).
        Value of 1.0 indicates identical directional orientation (collinear).
        Value of 0.0 indicates orthogonal (no shared semantic terms/dimensions).
        Invariant to document length (length-normalization property).

    Args:
        q: Query vector.
        d: Document vector.
        eps: Precision constant.

    Returns:
        Cosine similarity bounded in [-1.0, 1.0].
    """
    q_arr = ensure_numpy_vector(q, name="q")
    d_arr = ensure_numpy_vector(d, name="d")

    if q_arr.shape[0] != d_arr.shape[0]:
        raise ValueError(
            f"Dimension mismatch for cosine similarity: {q_arr.shape[0]} != {d_arr.shape[0]}"
        )

    norm_q = vector_magnitude(q_arr)
    norm_d = vector_magnitude(d_arr)

    # If either vector is the zero vector, similarity is 0.0
    if norm_q < eps or norm_d < eps:
        return 0.0

    raw_dot = np.dot(q_arr, d_arr)
    sim = raw_dot / (norm_q * norm_d)

    # Clip to strictly adhere to the Cauchy-Schwarz inequality bounds [-1.0, 1.0]
    return float(np.clip(sim, -1.0, 1.0))


def batch_cosine_similarity(
    query_vec: Union[np.ndarray, Sequence[float]],
    doc_matrix: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes cosine similarity in vectorized batch mode between 1 query vector
    and M document vectors stored as rows in doc_matrix (shape M x D).

    Args:
        query_vec: 1-D array of shape (D,).
        doc_matrix: 2-D array of shape (M, D).
        eps: Small float to prevent division by zero.

    Returns:
        1-D array of shape (M,) containing cosine similarity for each document.
    """
    q = ensure_numpy_vector(query_vec, name="query_vec")
    matrix = np.asarray(doc_matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(f"doc_matrix must be 2-dimensional (M x D), got {matrix.ndim}D")
    if matrix.shape[1] != q.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: query has {q.shape[0]} dims, "
            f"doc_matrix has {matrix.shape[1]} dims."
        )

    # L2 normalize query
    q_norm = vector_magnitude(q)
    if q_norm < eps:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    q_unit = q / q_norm

    # L2 normalize each row in doc_matrix
    doc_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Mask zero norms
    zero_mask = (doc_norms < eps).flatten()
    doc_norms_safe = np.where(doc_norms < eps, 1.0, doc_norms)
    doc_unit = matrix / doc_norms_safe

    # Matrix-vector multiplication computes all dot products simultaneously
    sims = np.dot(doc_unit, q_unit)
    sims[zero_mask] = 0.0
    return np.clip(sims, -1.0, 1.0)


def min_max_normalize(
    scores: Union[np.ndarray, Sequence[float]],
    target_min: float = 0.0,
    target_max: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Scales a sequence of scores into the bounded range [target_min, target_max].
    Crucial for unifying disparate score distributions (e.g., BM25 unbounded scores
    with dense cosine similarity [-1, 1] or [0, 1]).

    Formula:
        s_norm = \\frac{s - s_{min}}{s_{max} - s_{min} + \\epsilon} \\times (target_{max} - target_{min}) + target_{min}

    Args:
        scores: Sequence or array of raw numerical scores.
        target_min: Lower bound of output (default: 0.0).
        target_max: Upper bound of output (default: 1.0).
        eps: Epsilon to prevent zero division when all scores are identical.

    Returns:
        1-D NumPy array of normalized scores.
    """
    arr = np.asarray(scores, dtype=np.float64).flatten()
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    if arr.size == 1:
        return np.array([target_max], dtype=np.float64)

    s_min = float(np.min(arr))
    s_max = float(np.max(arr))

    spread = s_max - s_min
    if spread < eps:
        # If all scores are equal, return target_max for all
        return np.full_like(arr, target_max)

    scaled = (arr - s_min) / spread
    rescaled = scaled * (target_max - target_min) + target_min
    return np.clip(rescaled, target_min, target_max)


def hybrid_fuse_score(
    dense_score: float,
    sparse_score: float,
    alpha: float = 0.60,
) -> float:
    """
    Convex linear combination combining dense semantic score and sparse lexical score.

    Formula:
        S_{hybrid} = \\alpha \\times S_{dense} + (1 - \\alpha) \\times S_{sparse}

    Parameters:
        dense_score: Normalized dense semantic similarity in [0, 1].
        sparse_score: Normalized BM25/TF-IDF sparse score in [0, 1].
        alpha: Weight parameter in [0.0, 1.0].
               alpha = 1.0 -> Dense only
               alpha = 0.0 -> Sparse only
               alpha = 0.60 -> Recommended MediAcad hybrid balance

    Returns:
        Fused hybrid score in [0.0, 1.0].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha parameter must be in [0.0, 1.0], got {alpha}")

    fused = (alpha * float(dense_score)) + ((1.0 - alpha) * float(sparse_score))
    return float(np.clip(fused, 0.0, 1.0))

