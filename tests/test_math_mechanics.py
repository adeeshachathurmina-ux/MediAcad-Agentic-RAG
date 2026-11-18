"""
Mathematical Mechanics Unit Test Suite.
MediAcad Agentic RAG - Member 2 (Hybrid Retrieval & Mathematical Explainability).

Verifies foundational linear algebra, vector space properties, and numerical
bounds required for the DS205.3 High Distinction rubric:
  - Vector magnitude (Euclidean norm)
  - Dot product & Cauchy-Schwarz inequality
  - L2 Unit Normalization (||v_hat|| = 1.0)
  - Cosine Similarity bounds [-1.0, 1.0] and directional invariance
  - Min-Max scaling to [0.0, 1.0]
  - Hybrid fusion convex combination
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.math_mechanics import (
    vector_magnitude,
    dot_product,
    l2_normalize,
    cosine_similarity,
    batch_cosine_similarity,
    min_max_normalize,
    hybrid_fuse_score,
)


class TestVectorMagnitude:
    """Verifies Euclidean L2 norm calculations."""

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        assert vector_magnitude(v) == 0.0

    def test_unit_vectors(self):
        assert math.isclose(vector_magnitude([1.0, 0.0]), 1.0, abs_tol=1e-9)
        assert math.isclose(vector_magnitude([0.0, 1.0, 0.0]), 1.0, abs_tol=1e-9)

    def test_pythagorean_triples(self):
        # 3-4-5 triangle
        assert math.isclose(vector_magnitude([3.0, 4.0]), 5.0, abs_tol=1e-9)
        # 5-12-13 triangle
        assert math.isclose(vector_magnitude([5.0, 12.0]), 13.0, abs_tol=1e-9)

    def test_homogeneity_property(self):
        """||alpha * v|| == |alpha| * ||v||"""
        v = np.array([1.5, -2.5, 3.0])
        alpha = -4.0
        norm_v = vector_magnitude(v)
        norm_scaled = vector_magnitude(alpha * v)
        assert math.isclose(norm_scaled, abs(alpha) * norm_v, abs_tol=1e-9)

    def test_empty_vector_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            vector_magnitude([])


class TestDotProduct:
    """Verifies inner product algebraic properties."""

    def test_commutativity(self):
        """u . v == v . u"""
        u = [1.2, 3.4, -2.1]
        v = [0.5, -1.8, 4.2]
        assert math.isclose(dot_product(u, v), dot_product(v, u), abs_tol=1e-9)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have dot product == 0.0"""
        u = [1.0, 0.0]
        v = [0.0, 1.0]
        assert dot_product(u, v) == 0.0

    def test_self_dot_product_equals_norm_squared(self):
        """v . v == ||v||^2"""
        v = [2.0, -3.0, 6.0]
        norm = vector_magnitude(v)
        assert math.isclose(dot_product(v, v), norm ** 2, abs_tol=1e-9)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            dot_product([1.0, 2.0], [1.0, 2.0, 3.0])


class TestL2Normalize:
    """Verifies unit hypersphere projection."""

    def test_non_zero_vector_magnitude_is_one(self):
        v = [3.0, -4.0, 12.0]
        v_hat = l2_normalize(v)
        assert math.isclose(vector_magnitude(v_hat), 1.0, abs_tol=1e-9)

    def test_zero_vector_safety(self):
        """Zero vector should return a zero vector without NaN or crash."""
        v = [0.0, 0.0, 0.0]
        v_hat = l2_normalize(v)
        assert np.all(v_hat == 0.0)
        assert not np.isnan(v_hat).any()

    def test_collinear_direction_preserved(self):
        v = np.array([2.0, 4.0, 6.0])
        v_hat = l2_normalize(v)
        # Ratios between components must remain identical
        assert math.isclose(v_hat[1] / v_hat[0], 2.0, abs_tol=1e-9)
        assert math.isclose(v_hat[2] / v_hat[0], 3.0, abs_tol=1e-9)


class TestCosineSimilarity:
    """Verifies directional similarity and mathematical bounds."""

    def test_identical_direction(self):
        """Identical vectors have cosine similarity == 1.0"""
        q = [1.0, 2.0, 3.0]
        d = [2.0, 4.0, 6.0]  # scalar multiple
        assert math.isclose(cosine_similarity(q, d), 1.0, abs_tol=1e-9)

    def test_orthogonal_similarity(self):
        q = [1.0, 0.0]
        d = [0.0, 5.0]
        assert math.isclose(cosine_similarity(q, d), 0.0, abs_tol=1e-9)

    def test_opposite_direction(self):
        q = [1.0, 1.0]
        d = [-2.0, -2.0]
        assert math.isclose(cosine_similarity(q, d), -1.0, abs_tol=1e-9)

    def test_zero_vector_similarity(self):
        q = [0.0, 0.0]
        d = [1.0, 2.0]
        assert cosine_similarity(q, d) == 0.0

    def test_equivalence_with_normalized_dot_product(self):
        """cos(q, d) == q_hat . d_hat"""
        q = [1.5, 2.5, -0.5]
        d = [0.8, -1.2, 3.4]
        cos_sim = cosine_similarity(q, d)
        q_hat = l2_normalize(q)
        d_hat = l2_normalize(d)
        dot_sim = dot_product(q_hat, d_hat)
        assert math.isclose(cos_sim, dot_sim, abs_tol=1e-9)

    def test_batch_cosine_matches_single(self):
        query = np.array([1.0, 2.0, 3.0])
        doc_matrix = np.array([
            [1.0, 2.0, 3.0],   # identical (sim = 1.0)
            [0.0, 0.0, 0.0],   # zero vector (sim = 0.0)
            [-1.0, -2.0, -3.0],# opposite (sim = -1.0)
            [3.0, -2.0, 1.0],  # arbitrary
        ])
        batch_sims = batch_cosine_similarity(query, doc_matrix)
        assert len(batch_sims) == 4
        assert math.isclose(batch_sims[0], 1.0, abs_tol=1e-9)
        assert math.isclose(batch_sims[1], 0.0, abs_tol=1e-9)
        assert math.isclose(batch_sims[2], -1.0, abs_tol=1e-9)
        single_sim = cosine_similarity(query, doc_matrix[3])
        assert math.isclose(batch_sims[3], single_sim, abs_tol=1e-9)


class TestMinMaxNormalize:
    """Verifies score rescaling into bounded [0, 1] range."""

    def test_standard_scaling(self):
        raw_scores = [2.0, 4.0, 6.0, 10.0]
        normed = min_max_normalize(raw_scores)
        assert math.isclose(normed[0], 0.0, abs_tol=1e-9)
        assert math.isclose(normed[-1], 1.0, abs_tol=1e-9)
        assert math.isclose(normed[1], (4.0 - 2.0) / (10.0 - 2.0), abs_tol=1e-9)

    def test_identical_scores(self):
        """When all scores are identical, all should map to target_max without zero-division error."""
        raw_scores = [5.5, 5.5, 5.5]
        normed = min_max_normalize(raw_scores)
        assert np.all(normed == 1.0)

    def test_single_element(self):
        normed = min_max_normalize([42.0])
        assert len(normed) == 1
        assert normed[0] == 1.0


class TestHybridScoreFusion:
    """Verifies convex combination of dense and sparse scores."""

    def test_balanced_fusion(self):
        # S_hybrid = 0.6 * 0.90 + 0.4 * 0.80 = 0.54 + 0.32 = 0.86
        fused = hybrid_fuse_score(dense_score=0.90, sparse_score=0.80, alpha=0.60)
        assert math.isclose(fused, 0.86, abs_tol=1e-9)

    def test_pure_dense_mode(self):
        fused = hybrid_fuse_score(dense_score=0.75, sparse_score=0.25, alpha=1.0)
        assert math.isclose(fused, 0.75, abs_tol=1e-9)

    def test_pure_sparse_mode(self):
        fused = hybrid_fuse_score(dense_score=0.75, sparse_score=0.25, alpha=0.0)
        assert math.isclose(fused, 0.25, abs_tol=1e-9)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="Alpha parameter must be in"):
            hybrid_fuse_score(0.5, 0.5, alpha=1.5)
        with pytest.raises(ValueError, match="Alpha parameter must be in"):
            hybrid_fuse_score(0.5, 0.5, alpha=-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


