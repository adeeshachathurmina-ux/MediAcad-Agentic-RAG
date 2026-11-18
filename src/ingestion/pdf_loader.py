"""
PDF Ingestion and Structure-Aware Extraction Module for MediAcad.
Member 1: Document Engineering & Data Pipeline.
Features:
- Dual-engine extraction using PyMuPDF (fast text) and pdfplumber (table matrices)
- Clean Markdown formatting for university governance tables
- Automated document-level and page-level metadata extraction
"""

import os
import re
from typing import List, Dict, Any, Optional
import pymupdf
import pdfplumber


class PDFLoader:
    """Extracts clean text, structured tables, and metadata page-by-page from policy PDFs."""

    KNOWN_DOCUMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
        "01": {
            "doc_id": "POL-ACAD-EX-01",
            "doc_title": "Examination and Medical Consideration Regulations",
            "policy_version": "v4.0 (2025 Mandatory Senate Revision)",
            "effective_year": 2025,
            "module": "Examinations"
        },
        "02": {
            "doc_id": "POL-ACAD-AT-02",
            "doc_title": "Attendance and Medical Absence Administration Policy",
            "policy_version": "v3.2 (2025 Mandatory Senate Revision)",
            "effective_year": 2025,
            "module": "Attendance"
        },
        "03": {
            "doc_id": "POL-ACAD-CW-03",
            "doc_title": "Coursework, Continuous Assessment, and Extension Regulations",
            "policy_version": "v3.0 (2025 Mandatory Senate Revision)",
            "effective_year": 2025,
            "module": "Coursework"
        },
        "04": {
            "doc_id": "POL-MED-WF-04",
            "doc_title": "Medical Documentation Standards and Student Welfare Guidelines",
            "policy_version": "v3.5 (2025 Mandatory Senate Revision)",
            "effective_year": 2025,
            "module": "Welfare"
        },
        "05": {
            "doc_id": "HBK-ACAD-PR-05",
            "doc_title": "Academic Progression Regulations and Student Handbook",
            "policy_version": "v1.8 (2023 Legacy / 2025 Overrides)",
            "effective_year": 2023,
            "module": "Progression"
        }
    }

    def __init__(self, pdf_path: str):
        self.pdf_path = os.path.abspath(pdf_path)
        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        self.filename = os.path.basename(self.pdf_path)
        self.metadata = self._resolve_metadata()

    def _resolve_metadata(self) -> Dict[str, Any]:
        """Resolves authoritative document metadata by prefix or content analysis."""
        for prefix, meta in self.KNOWN_DOCUMENT_REGISTRY.items():
            if self.filename.startswith(prefix):
                return dict(meta)

        # Fallback to regex analysis from first page text
        doc = pymupdf.open(self.pdf_path)
        first_page_text = doc[0].get_text() if len(doc) > 0 else ""
        doc.close()

        doc_id_match = re.search(r'(POL-[A-Z]+-[A-Z0-9]+|HBK-[A-Z]+-[A-Z0-9]+)', first_page_text)
        version_match = re.search(r'(v\d+\.\d+[^)\n]*)', first_page_text)
        year_match = re.search(r'\b(202[0-9])\b', first_page_text)

        doc_id = doc_id_match.group(0) if doc_id_match else "UNKNOWN-DOC"
        version = version_match.group(0) if version_match else "v1.0 (2025)"
        year = int(year_match.group(0)) if year_match else 2025

        return {
            "doc_id": doc_id,
            "doc_title": self.filename.replace(".pdf", "").replace("_", " "),
            "policy_version": version,
            "effective_year": year,
            "module": "General"
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Cleans bullet point glyphs, font ligatures, and extra trailing whitespace."""
        if not text:
            return ""
        # Replace cid bullets or strange glyphs
        cleaned = re.sub(r'\(cid:\d+\)', '• ', text)
        cleaned = cleaned.replace('\xa0', ' ')
        # Normalize double blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    @staticmethod
    def _table_to_markdown(table_rows: List[List[Optional[str]]]) -> str:
        """Converts raw table cell matrix into clean Markdown format."""
        if not table_rows or len(table_rows) < 1:
            return ""

        # Normalize cells: replace newlines inside cell with space, None with empty string
        normalized = []
        for row in table_rows:
            if not row:
                continue
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    c_str = str(cell).replace("\n", " ").strip()
                    cleaned_row.append(c_str)
            if any(c for c in cleaned_row):  # skip completely empty rows
                normalized.append(cleaned_row)

        if not normalized:
            return ""

        header = normalized[0]
        col_count = len(header)
        md_lines = []
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * col_count) + " |")

        for row in normalized[1:]:
            # Pad row if columns mismatch
            padded = row + [""] * (col_count - len(row)) if len(row) < col_count else row[:col_count]
            md_lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(md_lines)

    def load_pages(self) -> List[Dict[str, Any]]:
        """
        Loads document pages using PyMuPDF for text and pdfplumber for tables.
        Returns list of structured page dictionaries.
        """
        pages_data = []

        # 1. Extract tables page-by-page with pdfplumber
        tables_per_page: Dict[int, List[Dict[str, Any]]] = {}
        with pdfplumber.open(self.pdf_path) as pdf:
            for p_idx, page in enumerate(pdf.pages):
                page_tables = []
                extracted_tables = page.extract_tables()
                if extracted_tables:
                    for t_idx, raw_table in enumerate(extracted_tables):
                        md_table = self._table_to_markdown(raw_table)
                        if md_table:
                            page_tables.append({
                                "table_index": t_idx + 1,
                                "markdown": md_table,
                                "raw_rows": raw_table
                            })
                tables_per_page[p_idx + 1] = page_tables

        # 2. Extract text page-by-page with PyMuPDF
        doc = pymupdf.open(self.pdf_path)
        for page_idx, page in enumerate(doc):
            p_num = page_idx + 1
            raw_text = page.get_text()
            clean_text = self._clean_text(raw_text)
            tables = tables_per_page.get(p_num, [])

            pages_data.append({
                "page_number": p_num,
                "text": clean_text,
                "tables": tables,
                "metadata": dict(self.metadata)
            })

        doc.close()
        return pages_data
