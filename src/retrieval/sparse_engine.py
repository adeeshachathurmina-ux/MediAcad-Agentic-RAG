"""
Custom Sparse Retrieval Engine (TF-IDF & BM25 Scoring with Explainability).
MediAcad Agentic RAG - Member 2 (Hybrid Retrieval & Mathematical Explainability).

Provides full white-box mathematical transparency for lexical information retrieval:
  1. Policy-Aware Tokenizer (preserves codes like EC-01, MED-04-CLR, 80%, 10-day)
  2. Vocabulary Building & Inverted Index
  3. Term Frequency (TF) & Smooth Inverse Document Frequency (IDF)
  4. Okapi BM25 Probabilistic Scoring (k1=1.5, b=0.75, length normalization)
  5. Min-Max Normalization to [0.0, 1.0]
  6. Explainability Breakdown (matched terms, IDF weights, per-term BM25 contributions)
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from src.interfaces.contracts import DocumentChunk
from src.retrieval.math_mechanics import min_max_normalize

# Standard English stop words to filter syntactic noise while keeping policy keywords
DEFAULT_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Policy-aware regex tokenizer.
    Preserves:
      - Alphanumeric codes: EC-01, MED-04-CLR, ATT-02A, EXT-03C, POL-ACAD-EX-01
      - Percentages: 80%, 72%, 65%
      - Durations / hyphens: 10-day, 5-working-day, 14-day
      - Standard words (lowercased)
    """
    if not text:
        return []

    # Pattern matches:
    # 1. Alphanumeric hyphenated codes/durations: e.g. POL-ACAD-EX-01, MED-04-CLR, 10-day
    # 2. Percentages: e.g. 80%, 72.5%
    # 3. Standard words: e.g. hospitalization, attendance
    pattern = r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|\d+(?:\.\d+)?%|[A-Za-z0-9]+"
    tokens = re.findall(pattern, text)
    processed_tokens: List[str] = []

    for t in tokens:
        t_lower = t.lower()
        if remove_stopwords and t_lower in DEFAULT_STOP_WORDS:
            continue
        # Avoid single-letter punctuation artifacts
        if len(t_lower) > 1 or t_lower.isdigit():
            processed_tokens.append(t_lower)

    return processed_tokens


def extract_chunk_search_text(chunk: Union[DocumentChunk, str]) -> str:
    """
    Extracts comprehensive searchable text from a DocumentChunk.
    Performs context header injection (doc_title, section_title, clause_id)
    so queries matching clause identifiers or headers retrieve accurately.
    """
    if isinstance(chunk, str):
        return chunk
    parts = []
    if getattr(chunk, "doc_title", None):
        parts.append(chunk.doc_title)
    if getattr(chunk, "section_title", None):
        parts.append(chunk.section_title)
    if getattr(chunk, "clause_id", None):
        parts.append(chunk.clause_id)
    if getattr(chunk, "text", None):
        parts.append(chunk.text)
    return " ".join(parts) if parts else getattr(chunk, "text", "")


class TFIDFAndBM25Engine:
    """
    Custom Sparse Scoring Engine combining TF-IDF and Okapi BM25 with
    mathematical trace extraction for explainable academic retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
    ):
        """
        Args:
            k1: Term frequency saturation parameter (standard: 1.2 to 2.0).
            b: Document length normalization parameter (standard: 0.75).
            remove_stopwords: Whether to remove common English stopwords.
        """
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise ValueError(f"b must be between 0.0 and 1.0, got {b}")

        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords

        # Corpus indexing state
        self.chunks: List[DocumentChunk] = []
        self.corpus_size: int = 0
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: np.ndarray = np.array([], dtype=np.float64)
        self.avg_doc_length: float = 0.0

        # Lexicon and statistics
        self.vocabulary: Dict[str, int] = {}
        self.doc_frequencies: Dict[str, int] = {}
        self.bm25_idf: Dict[str, float] = {}
        self.tfidf_idf: Dict[str, float] = {}

    def fit(self, chunks: List[DocumentChunk]) -> "TFIDFAndBM25Engine":
        """
        Builds inverted index, vocabulary, and IDF tables from DocumentChunk corpus.
        """
        if not chunks:
            raise ValueError("Cannot fit TFIDFAndBM25Engine with empty chunks list.")

        self.chunks = list(chunks)
        self.corpus_size = len(chunks)

        # 1. Tokenize each document with context header text
        self.doc_tokens = [
            tokenize(extract_chunk_search_text(chunk), remove_stopwords=self.remove_stopwords)
            for chunk in self.chunks
        ]

        # 2. Document lengths and average document length (avgdl)
        lengths = [max(len(toks), 1) for toks in self.doc_tokens]
        self.doc_lengths = np.array(lengths, dtype=np.float64)
        self.avg_doc_length = float(np.mean(self.doc_lengths))

        # 3. Document frequencies df(t)
        self.doc_frequencies.clear()
        self.vocabulary.clear()

        for d_idx, toks in enumerate(self.doc_tokens):
            unique_terms = set(toks)
            for term in unique_terms:
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)

        # 4. Compute IDFs
        N = self.corpus_size
        self.bm25_idf.clear()
        self.tfidf_idf.clear()

        for term, df in self.doc_frequencies.items():
            # Standard Robertson-Spärck Jones BM25 IDF (non-negative formulation)
            # Formula: ln(1 + (N - df + 0.5) / (df + 0.5))
            bm25_val = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            self.bm25_idf[term] = max(bm25_val, 1e-6)

            # Smooth Scikit-learn style TF-IDF IDF
            # Formula: ln((1 + N) / (1 + df)) + 1.0
            tfidf_val = math.log((1.0 + N) / (1.0 + df)) + 1.0
            self.tfidf_idf[term] = tfidf_val

        return self

    def get_term_frequency(self, term: str, doc_index: int) -> int:
        """Returns raw frequency f(t, d) of a term in a document."""
        if not (0 <= doc_index < self.corpus_size):
            raise IndexError(f"Document index {doc_index} out of bounds.")
        term_lower = term.lower()
        return self.doc_tokens[doc_index].count(term_lower)

    def calculate_bm25_score_for_doc(
        self,
        query_terms: List[str],
        doc_index: int,
    ) -> Tuple[float, Dict[str, float], Dict[str, float], List[str]]:
        """
        Computes BM25 score for a single document with full mathematical trace.

        Formula:
            BM25(q, d) = sum_{t in q} IDF(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * (|d| / avgdl)))

        Returns:
            Tuple of:
              - raw_bm25_score (float)
              - per_term_bm25_contribution (Dict[str, float])
              - term_frequencies (Dict[str, float])
              - matched_terms (List[str])
        """
        L_d = self.doc_lengths[doc_index]
        avgdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
        norm_factor = 1.0 - self.b + self.b * (L_d / avgdl)

        total_score = 0.0
        contributions: Dict[str, float] = {}
        tf_dict: Dict[str, float] = {}
        matched: List[str] = []

        doc_toks = self.doc_tokens[doc_index]

        for term in query_terms:
            if term not in self.vocabulary:
                continue

            f_td = doc_toks.count(term)
            if f_td > 0:
                matched.append(term)
                tf_dict[term] = float(f_td)
                idf = self.bm25_idf.get(term, 0.0)

                # Term saturation numerator and denominator
                numerator = f_td * (self.k1 + 1.0)
                denominator = f_td + self.k1 * norm_factor
                term_score = idf * (numerator / denominator)

                contributions[term] = float(term_score)
                total_score += term_score

        return float(total_score), contributions, tf_dict, matched

    def score_corpus(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Scores the entire indexed corpus against the query.
        Returns detailed scoring dicts sorted descending by BM25 score.

        Dict fields:
          - doc_index: int
          - chunk: DocumentChunk
          - raw_score: float
          - normalized_score: float (Min-Max scaled to [0.0, 1.0])
          - matched_terms: List[str]
          - term_frequencies: Dict[str, float]
          - idf_values: Dict[str, float]
          - term_contributions: Dict[str, float]
          - top_keywords: List[Tuple[str, float]]
        """
        if self.corpus_size == 0:
            raise RuntimeError("Engine has not been fitted. Call fit() first.")

        query_tokens = tokenize(query, remove_stopwords=self.remove_stopwords)
        if not query_tokens:
            # Query has no content words (e.g., all stopwords or empty)
            return [
                {
                    "doc_index": i,
                    "chunk": self.chunks[i],
                    "raw_score": 0.0,
                    "normalized_score": 0.0,
                    "matched_terms": [],
                    "term_frequencies": {},
                    "idf_values": {},
                    "term_contributions": {},
                    "top_keywords": [],
                }
                for i in range(self.corpus_size)
            ]

        raw_scores: List[float] = []
        details: List[Dict[str, Any]] = []

        for i in range(self.corpus_size):
            score, contrib, tfs, matched = self.calculate_bm25_score_for_doc(
                query_tokens, i
            )
            raw_scores.append(score)

            # IDF lookup for matched terms
            idfs = {t: self.bm25_idf.get(t, 0.0) for t in matched}

            # Top keywords sorted by BM25 contribution
            sorted_kw = sorted(
                contrib.items(), key=lambda item: item[1], reverse=True
            )

            details.append({
                "doc_index": i,
                "chunk": self.chunks[i],
                "raw_score": score,
                "matched_terms": matched,
                "term_frequencies": tfs,
                "idf_values": idfs,
                "term_contributions": contrib,
                "top_keywords": sorted_kw,
            })

        # Min-Max normalize scores into [0.0, 1.0]
        norm_scores = min_max_normalize(raw_scores, target_min=0.0, target_max=1.0)
        for i, norm_val in enumerate(norm_scores):
            # If raw score is 0.0 (no matches), ensure normalized score is 0.0
            if raw_scores[i] == 0.0:
                details[i]["normalized_score"] = 0.0
            else:
                details[i]["normalized_score"] = float(norm_val)

        # Sort descending by raw BM25 score
        details.sort(key=lambda d: d["raw_score"], reverse=True)
        return details

    def retrieve_top_k(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k scored chunks with full explainability traces.
        """
        scored_docs = self.score_corpus(query)
        return scored_docs[:top_k]
