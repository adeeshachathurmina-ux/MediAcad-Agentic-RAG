"""
Persistent ChromaDB Vector Store Builder and Chunks Exporter for MediAcad.
Member 1: Document Engineering & Data Pipeline.
Features:
- Encodes structured chunks using Sentence-Transformers (all-MiniLM-L6-v2)
- Builds persistent vector collection in data/chroma_db/
- Exports clean JSON chunks to data/processed_chunks.json for Member 2 (Sparse BM25/TF-IDF)
- Provides query interface for sanity checking dense vector retrieval
"""

import os
import json
from dataclasses import asdict
from typing import List, Dict, Any, Optional

# Configure offline model cache path
CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "model_cache")
)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer
from src.interfaces.contracts import DocumentChunk
from src.ingestion.clause_chunker import ClauseAwareChunker


class LocalSentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """ChromaDB-compatible embedding function wrapping SentenceTransformer with local cache."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None):
        self.model_name = model_name
        self.cache_folder = cache_folder or CACHE_DIR
        os.makedirs(self.cache_folder, exist_ok=True)
        self.model = SentenceTransformer(
            model_name,
            cache_folder=self.cache_folder,
            local_files_only=True
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def name(self) -> str:
        return f"SentenceTransformer_{self.model_name}"


class PersistentChromaStore:
    """Manages persistent ChromaDB indexing, embedding generation, and chunks JSON export."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "academic_policies",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = LocalSentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 50) -> int:
        """
        Embeds and indexes DocumentChunk objects into persistent ChromaDB in batches.
        """
        if not chunks:
            print("⚠️ No chunks provided for indexing.")
            return 0

        total = len(chunks)
        print(f"📦 Indexing {total} chunks into ChromaDB at '{self.persist_dir}'...")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]

            # Flatten metadata values for Chroma compatibility (strings, ints, floats, bools)
            metadatas = []
            for c in batch:
                meta_dict = {
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "policy_version": c.policy_version,
                    "effective_year": c.effective_year,
                    "page_number": c.page_number,
                    "section_title": c.section_title[:100],
                    "clause_id": c.clause_id,
                    "chunk_type": c.chunk_type,
                    "is_legacy": bool(c.metadata.get("is_legacy", False)),
                    "module": str(c.metadata.get("module", "General"))
                }
                metadatas.append(meta_dict)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        count = self.collection.count()
        print(f"✅ Successfully indexed {total} chunks. Collection total count: {count}")
        return count

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Queries the vector database using cosine dense similarity."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return results

    @staticmethod
    def export_chunks_to_json(chunks: List[DocumentChunk], export_path: str = "data/processed_chunks.json") -> str:
        """
        Exports strongly-typed chunks to JSON format for consumption by Member 2 (BM25/TF-IDF)
        and Member 4 (Evaluations & Dashboards).
        """
        export_abs = os.path.abspath(export_path)
        os.makedirs(os.path.dirname(export_abs), exist_ok=True)

        chunk_dicts = [asdict(c) for c in chunks]

        with open(export_abs, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)

        print(f"📄 Exported {len(chunks)} chunks to '{export_abs}'")
        return export_abs


def run_pipeline(pdf_dir: str = "data/raw_pdfs", chroma_dir: str = "data/chroma_db", json_export: str = "data/processed_chunks.json"):
    """Executes the complete Member 1 end-to-end ingestion pipeline."""
    print("🚀 Starting Member 1 Ingestion Pipeline...")

    chunker = ClauseAwareChunker()
    chunks = chunker.parse_and_chunk(pdf_dir)
    print(f"🧩 Structure-aware parsing complete: {len(chunks)} chunks generated.")

    # Export to JSON
    PersistentChromaStore.export_chunks_to_json(chunks, json_export)

    # Index into ChromaDB
    store = PersistentChromaStore(persist_dir=chroma_dir)
    store.index_chunks(chunks)

    # Sanity verification query
    sample_query = "10 days hospitalized medical certificate attendance condonation"
    print(f"\n🔍 Running verification query: '{sample_query}'")
    query_res = store.query(sample_query, top_k=3)
    if query_res and "ids" in query_res and query_res["ids"]:
        for rank, (chunk_id, dist) in enumerate(zip(query_res["ids"][0], query_res["distances"][0])):
            doc = query_res["metadatas"][0][rank]["doc_id"]
            clause = query_res["metadatas"][0][rank]["clause_id"]
            print(f"  [{rank+1}] {chunk_id} ({doc} | {clause}) - Cosine Distance: {dist:.4f}")

    print("\n🎉 Member 1 Ingestion Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
