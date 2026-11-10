"""
Interactive Mathematical Explainability Demo for MediAcad Hybrid Retriever.
Member 2: Hybrid Retrieval & Mathematical Explainability.

Designed for the 5-minute video demonstration (Minute 1:15 - 2:30) and Viva defense.
Exposes the white-box mathematical mechanics of every retrieval decision.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interfaces.contracts import DocumentChunk
from src.retrieval.hybrid_retriever import HybridExplainableRetriever


def run_demo():
    fixture_path = PROJECT_ROOT / "data" / "fixtures" / "sample_chunks.json"
    if not fixture_path.exists():
        print(f"Error: Fixture file not found at {fixture_path}")
        return

    raw_chunks = json.loads(fixture_path.read_text(encoding="utf-8"))
    chunks = [DocumentChunk(**item) for item in raw_chunks]

    print("=" * 80)
    print(" MediAcad Hybrid Retrieval & Mathematical Explainability Engine (Member 2)")
    print("=" * 80)
    print(f"Indexed Corpus Size: {len(chunks)} Policy Chunks")

    # Initialize retriever with recommended alpha = 0.60
    retriever = HybridExplainableRetriever(chunks=chunks, alpha=0.60)

    # Real-World Problem Scenario Query (Page 1 of Project Concept Guide)
    student_query = "Hospitalized for 10 days, attendance 72%, missed final exam: What procedure and forms are required?"
    print(f"\n[QUERY]: '{student_query}'\n")

    results = retriever.retrieve(student_query, top_k=3)

    print("-" * 80)
    print(f"{'Rank':<5} | {'Document ID':<16} | {'Clause/Section':<20} | {'BM25':<8} | {'Cosine':<8} | {'Hybrid (alpha=0.6)':<8}")
    print("-" * 80)

    for r in results:
        print(
            f"#{r.rank:<4} | {r.chunk.doc_id:<16} | {r.chunk.clause_id[:20]:<20} | "
            f"{r.sparse_score:<8.4f} | {r.dense_score:<8.4f} | {r.hybrid_score:<8.4f}"
        )

    print("\n" + "=" * 80)
    print(" WHITE-BOX MATHEMATICAL EXPLAINABILITY BREAKDOWN (Top Chunk #1)")
    print("=" * 80)

    top_trace = results[0].math_trace
    if top_trace:
        print(f"Target Document      : {results[0].chunk.doc_title}")
        print(f"Target Clause        : {results[0].chunk.clause_id} (Page {results[0].chunk.page_number})")
        print(f"Matched Query Terms  : {top_trace.matched_query_terms}")
        print(f"Term Frequencies (TF): {top_trace.term_frequencies}")
        print(f"IDF Weights          : {top_trace.idf_values}")
        print(f"Raw BM25 Score       : {top_trace.bm25_raw_score:.4f}")
        print(f"Normalized BM25      : {top_trace.bm25_normalized_score:.4f}  [Min-Max Scaled to 0.0 - 1.0]")
        print("-" * 50)
        print(f"Query Vector Norm    : ||q||_2  = {top_trace.query_vector_norm:.4f}")
        print(f"Doc Vector Norm      : ||d||_2  = {top_trace.doc_vector_norm:.4f}")
        print(f"Inner Dot Product    : (q . d)  = {top_trace.dot_product:.4f}")
        print(f"Cosine Similarity    : cos(theta) = (q . d) / (||q|| * ||d||) = {top_trace.cosine_similarity:.4f}")
        print("-" * 50)
        print(f"Fusion Equation      : S_hybrid = alpha * S_dense + (1 - alpha) * S_sparse")
        print(f"Calculated Score     : {top_trace.alpha_weight:.2f} * {top_trace.cosine_similarity:.4f} + "
              f"{(1 - top_trace.alpha_weight):.2f} * {top_trace.bm25_normalized_score:.4f} = {top_trace.hybrid_score:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()

