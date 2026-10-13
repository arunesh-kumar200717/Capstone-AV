from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL, PAPERS_DIR
from src.document_loader import load_pdf_text, validate_pdf
from src.embeddings import EmbeddingManager
from src.text_splitter import split_chunks_for_pages
from src.vector_store import ResearchVectorStore


def ingest_directory(directory: str, model_name: str = DEFAULT_EMBEDDING_MODEL):
    store = ResearchVectorStore('research_chunks')
    manager = EmbeddingManager(model_name)
    paper_dir = Path(directory)
    successes = []
    failures = []
    for paper in sorted(paper_dir.glob('*.pdf')):
        ok, message = validate_pdf(str(paper))
        if not ok:
            failures.append((paper.name, message))
            continue
        doc, errors = load_pdf_text(str(paper))
        if doc is None or doc.error:
            failures.append((paper.name, doc.error if doc else 'Unable to load PDF'))
            continue
        chunks = split_chunks_for_pages(doc.document_id, doc.title, doc.filename, doc.pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not chunks:
            failures.append((paper.name, 'No extractable text found for indexing.'))
            continue
        embeddings = manager.embed_documents([chunk['text'] for chunk in chunks], model_name=model_name)
        store.add_documents(chunks, embeddings, model_name=model_name)
        successes.append((paper.name, len(chunks)))
    store.persist()
    return successes, failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest PDFs into the vector store.')
    parser.add_argument('--directory', default=str(PAPERS_DIR), help='Directory containing PDFs.')
    parser.add_argument('--model', default=DEFAULT_EMBEDDING_MODEL, help='Sentence-transformers embedding model to use.')
    args = parser.parse_args()
    successes, failures = ingest_directory(args.directory, args.model)
    print(f'Indexed successfully: {len(successes)} files')
    for name, count in successes:
        print(f'- {name}: {count} chunks')
    if failures:
        print(f'Failed files: {len(failures)}')
        for name, reason in failures:
            print(f'- {name}: {reason}')
