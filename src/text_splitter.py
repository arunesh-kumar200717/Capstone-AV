from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    RecursiveCharacterTextSplitter = None


def split_chunks_for_pages(document_id: str, title: str, source_file: str, pages: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_text = page.get('text', '')
        if not page_text.strip():
            continue
        if RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=['\n\n', '\n', '. ', ' ', ''],
                length_function=len,
            )
            texts = splitter.split_text(page_text)
        else:
            texts = re.split(r'(?<=\.)\s+|\n\n+', page_text)
            texts = [segment for segment in texts if segment.strip()]
            if not texts:
                texts = [page_text]
            texts = _manual_chunking(texts, chunk_size, chunk_overlap)

        for idx, chunk_text in enumerate(texts):
            cleaned = re.sub(r'\s+', ' ', chunk_text).strip()
            if not cleaned:
                continue
            chunk_id = f'{document_id}-p{page["page_number"]}-c{len(chunks) + 1}'
            chunks.append({
                'document_id': document_id,
                'chunk_id': chunk_id,
                'paper_title': title,
                'source_file': source_file,
                'page_number': page['page_number'],
                'chunk_index': len(chunks) + 1,
                'text': cleaned,
            })
    return chunks


def _manual_chunking(texts: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    output: List[str] = []
    current = ''
    for text in texts:
        if len(current) + len(text) <= chunk_size:
            current = (current + ' ' + text).strip()
            continue
        if current.strip():
            output.append(current)
        if chunk_overlap > 0 and output:
            previous = output[-1]
            current = (previous[-chunk_overlap:] + ' ' + text).strip()
        else:
            current = text
    if current.strip():
        output.append(current)
    return output
