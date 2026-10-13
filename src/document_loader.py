from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import fitz


@dataclass
class PaperDocument:
    document_id: str
    title: str
    filename: str
    file_path: str
    page_count: int
    pages: List[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def stable_document_id(file_path: str) -> str:
    return hashlib.sha1(file_path.encode('utf-8')).hexdigest()[:12]


def extract_title_from_pdf(doc: fitz.Document, fallback: str) -> str:
    metadata_title = (doc.metadata or {}).get('title')
    if metadata_title and metadata_title.strip():
        return metadata_title.strip()
    first_page = doc[0]
    text = first_page.get_text('text', sort=True)
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if len(cleaned) > 0:
        return cleaned[:160]
    return fallback


def load_pdf_text(file_path: str) -> tuple[PaperDocument | None, list[str]]:
    path = Path(file_path)
    if not path.exists():
        return None, ['File not found.']
    try:
        pdf = fitz.open(str(path))
    except Exception as exc:  # pragma: no cover
        return None, [f'Failed to open PDF: {exc}']

    page_count = len(pdf)
    title = extract_title_from_pdf(pdf, path.stem)
    document_id = stable_document_id(str(path))
    pages: List[dict[str, Any]] = []
    errors: List[str] = []
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        text = page.get_text('text', sort=True)
        cleaned = re.sub(r'\s+', ' ', text).strip()
        if cleaned:
            pages.append({
                'document_id': document_id,
                'paper_title': title,
                'source_file': path.name,
                'page_number': page_index + 1,
                'text': cleaned,
            })
        else:
            errors.append(f'Page {page_index + 1}: no selectable text extracted (OCR may be needed).')
    if not pages:
        pdf.close()
        return PaperDocument(
            document_id=document_id,
            title=title,
            filename=path.name,
            file_path=str(path),
            page_count=page_count,
            pages=[],
            error='No selectable text found. This PDF may be scanned, password-protected, or image-based. OCR may be required.',
        ), errors
    pdf.close()
    return PaperDocument(
        document_id=document_id,
        title=title,
        filename=path.name,
        file_path=str(path),
        page_count=page_count,
        pages=pages,
        error=None,
    ), errors


def validate_pdf(file_path: str, max_size_mb: int = 30) -> tuple[bool, str]:
    path = Path(file_path)
    if not path.exists():
        return False, 'File does not exist.'
    if path.suffix.lower() != '.pdf':
        return False, 'Only PDF files are accepted.'
    if path.stat().st_size > max_size_mb * 1024 * 1024:
        return False, f'File exceeds {max_size_mb} MB limit.'
    return True, 'OK'
