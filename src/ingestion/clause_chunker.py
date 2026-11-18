"""
Structure-Aware Clause and Table Chunker for MediAcad.
Member 1: Document Engineering & Data Pipeline.
Preserves legal clause boundaries, governance matrices, and hierarchical sections.
Enriches chunks with Context Header Injection and Domain-Specific Metadata.
Inherits from BaseChunker interface.
"""

import os
import re
from typing import List, Dict, Any, Optional
from src.interfaces.contracts import BaseChunker, DocumentChunk
from src.ingestion.pdf_loader import PDFLoader


class ClauseAwareChunker(BaseChunker):
    """
    Intelligent Chunker that splits policy documents along structural boundaries
    (Sections, Clauses, Tables, and Legacy Callouts) with Context Header Injection.
    """

    # Matches Section, Clause, Table, or Callout headers
    HEADER_PATTERN = re.compile(
        r'(?:^|\n)('
        r'SECTION\s+\d+:[^\n]+|'
        r'Clause\s+\d+(\.\d+)*[^\n:]*|'
        r'Sub-clause\s+\d+(\.\d+)*[^\n:]*|'
        r'Table\s+\d+(\.\d+)*:[^\n]+|'
        r'CRITICAL\s+GOVERNANCE\s+NOTICE[^\n]*|'
        r'Conflict\s+Alert\s*#?\d*:[^\n]*'
        r')',
        re.IGNORECASE
    )

    # Forms & Key Governance Thresholds regex
    FORM_PATTERN = re.compile(r'\b(Form\s+[A-Z0-9\-]+)\b', re.IGNORECASE)
    TIMELINE_PATTERN = re.compile(r'\b(\d+\s*(?:working\s+days|calendar\s+days|hours|weeks))\b', re.IGNORECASE)
    PERCENT_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?%)\b')

    def __init__(self, min_chunk_char_length: int = 40):
        self.min_chunk_char_length = min_chunk_char_length

    def parse_and_chunk(self, pdf_directory: str) -> List[DocumentChunk]:
        """
        Parses all policy PDFs in the specified directory and produces strongly-typed DocumentChunk objects.
        Satisfies BaseChunker interface.
        """
        if not os.path.isdir(pdf_directory):
            raise FileNotFoundError(f"PDF directory does not exist: {pdf_directory}")

        pdf_files = sorted([f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")])
        all_chunks: List[DocumentChunk] = []

        for pdf_name in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_name)
            loader = PDFLoader(pdf_path)
            pages = loader.load_pages()

            current_section = "GENERAL POLICY PROVISIONS"

            for page in pages:
                page_chunks = self.chunk_page(page, current_section=current_section)
                for chunk in page_chunks:
                    if chunk.section_title and chunk.section_title != current_section:
                        current_section = chunk.section_title
                    all_chunks.append(chunk)

        return all_chunks

    def _build_context_header(self, doc_id: str, version: str, year: int, section: str, clause: str, page: int) -> str:
        """Injects authoritative academic governance header into chunk text for explainable RAG retrieval."""
        clean_section = re.sub(r'\s+', ' ', section).strip()[:60]
        return f"[Doc: {doc_id} | Version: {version} ({year}) | Section: {clean_section} | Clause: {clause} | Page: {page}]"

    def _enrich_metadata(self, text: str, base_meta: Dict[str, Any], clause_id: str, page_num: int) -> Dict[str, Any]:
        """Extracts mentioned administrative forms, deadlines, percentages, and legacy audit tags."""
        forms_found = sorted(list(set(self.FORM_PATTERN.findall(text))))
        timelines_found = sorted(list(set(self.TIMELINE_PATTERN.findall(text))))
        percentages_found = sorted(list(set(self.PERCENT_PATTERN.findall(text))))

        is_legacy = bool(base_meta.get("effective_year") == 2023 or "Section 5" in text or "Legacy" in clause_id)

        meta = dict(base_meta)
        meta.update({
            "page": page_num,
            "clause": clause_id,
            "mentioned_forms": forms_found,
            "mentioned_timelines": timelines_found,
            "mentioned_percentages": percentages_found,
            "is_legacy": is_legacy,
            "authority_status": "LEGACY_SUPERSEDED" if is_legacy else "AUTHORITATIVE_2025"
        })
        return meta

    def chunk_page(self, page_data: Dict[str, Any], current_section: str = "GENERAL PROVISIONS") -> List[DocumentChunk]:
        """Splits a single page into structured clause and table chunks with Context Header Injection."""
        page_num = page_data["page_number"]
        meta = page_data["metadata"]
        text = page_data["text"]
        tables = page_data.get("tables", [])

        chunks: List[DocumentChunk] = []
        doc_id = meta["doc_id"]
        doc_title = meta["doc_title"]
        version = meta["policy_version"]
        effective_year = meta["effective_year"]

        # Track and build table chunks
        table_chunks = self._build_table_chunks(tables, page_num, meta, current_section)
        chunks.extend(table_chunks)

        # Segment page text by Clause / Section boundaries
        matches = list(self.HEADER_PATTERN.finditer(text))

        if not matches:
            if len(text.strip()) >= self.min_chunk_char_length:
                chunk_id = f"{doc_id}_p{page_num}_main"
                header = self._build_context_header(doc_id, version, effective_year, current_section, "General", page_num)
                injected_text = f"{header}\n{text.strip()}"
                enriched_meta = self._enrich_metadata(text, meta, "General", page_num)

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    policy_version=version,
                    effective_year=effective_year,
                    page_number=page_num,
                    section_title=current_section,
                    clause_id="General",
                    chunk_type="clause",
                    text=injected_text,
                    metadata=enriched_meta
                ))
            return chunks

        # Extract segments
        active_section = current_section
        for i, match in enumerate(matches):
            header_str = match.group(0).strip()
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segment_text = text[start_pos:end_pos].strip()

            if len(segment_text) < self.min_chunk_char_length:
                continue

            # Check if this header introduces a new SECTION
            if header_str.upper().startswith("SECTION"):
                active_section = header_str
                body_after_header = segment_text[len(header_str):].strip()
                if not body_after_header:
                    continue

            # Determine chunk type and clause_id
            chunk_type, clause_id = self._classify_header(header_str)

            # Context Header Injection
            header = self._build_context_header(doc_id, version, effective_year, active_section, clause_id, page_num)
            full_text = f"{header}\n{segment_text}"

            # Enriched metadata
            enriched_meta = self._enrich_metadata(segment_text, meta, clause_id, page_num)

            # Generate stable chunk_id
            safe_clause = re.sub(r'[^a-zA-Z0-9_\.]', '', clause_id.lower().replace(" ", "_"))
            chunk_id = f"{doc_id}_p{page_num}_{safe_clause or 'c' + str(i)}"

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                policy_version=version,
                effective_year=effective_year,
                page_number=page_num,
                section_title=active_section,
                clause_id=clause_id,
                chunk_type=chunk_type,
                text=full_text,
                metadata=enriched_meta
            ))

        return chunks

    def _classify_header(self, header_str: str) -> tuple[str, str]:
        """Identifies whether header is a clause, table, section, or callout."""
        clean_header = header_str.strip()
        clause_match = re.search(r'(Clause\s+\d+(\.\d+)*|Sub-clause\s+\d+(\.\d+)*)', clean_header, re.IGNORECASE)
        table_match = re.search(r'(Table\s+\d+(\.\d+)*)', clean_header, re.IGNORECASE)

        if "CONFLICT ALERT" in clean_header.upper() or "CRITICAL GOVERNANCE NOTICE" in clean_header.upper():
            return "callout", "Conflict Notice"
        elif table_match:
            return "table", table_match.group(0)
        elif clause_match:
            return "clause", clause_match.group(0)
        elif clean_header.upper().startswith("SECTION"):
            sec_num = re.search(r'SECTION\s+\d+', clean_header, re.IGNORECASE)
            return "clause", sec_num.group(0) if sec_num else "Section"
        return "clause", clean_header[:25]

    def _build_table_chunks(self, tables: List[Dict[str, Any]], page_num: int, meta: Dict[str, Any], section_title: str) -> List[DocumentChunk]:
        """Constructs dedicated table chunks with injected context headers."""
        chunks: List[DocumentChunk] = []
        doc_id = meta["doc_id"]
        version = meta["policy_version"]
        effective_year = meta["effective_year"]

        for t_info in tables:
            t_idx = t_info["table_index"]
            md_table = t_info["markdown"]
            raw_rows = t_info.get("raw_rows", [])
            caption = f"Table {page_num}.{t_idx}"
            if raw_rows and len(raw_rows) > 0 and raw_rows[0]:
                for cell in raw_rows[0]:
                    if cell and "Table" in str(cell):
                        m = re.search(r'Table\s+\d+(\.\d+)*', str(cell))
                        if m:
                            caption = m.group(0)
                            break

            chunk_id = f"{doc_id}_p{page_num}_t{t_idx}"
            header = self._build_context_header(doc_id, version, effective_year, section_title, caption, page_num)
            full_table_text = f"{header}\n{md_table}"
            enriched_meta = self._enrich_metadata(md_table, meta, caption, page_num)
            enriched_meta["is_table"] = True
            enriched_meta["table_index"] = t_idx

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=meta["doc_title"],
                policy_version=version,
                effective_year=effective_year,
                page_number=page_num,
                section_title=section_title,
                clause_id=caption,
                chunk_type="table",
                text=full_table_text,
                metadata=enriched_meta
            ))

        return chunks
