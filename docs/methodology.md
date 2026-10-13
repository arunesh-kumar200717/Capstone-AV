# Methodology

## Dataset selection
The project uses a curated set of research PDFs in the local document library. The system is designed around realistic paper ingestion and retrieval rather than synthetic static sample data.

## PDF parsing
Each PDF is loaded with PyMuPDF. The loader preserves the PDF metadata title, filename, and per-page numbers. When a PDF contains no selectable text, it raises a warning that OCR may be required.

## Chunking and overlap
Text is split into overlapping chunks with configurable chunk size and overlap. Chunk metadata includes document ID, chunk ID, paper title, file name, page number, and chunk index.

## Embedding comparison
The system compares two embedding models:
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

These models are configurable and can be selected in the retrieval lab and app settings.

## Vector database
ChromaDB stores persistent vectors keyed by chunk metadata and chunk identifiers. This allows the system to persist results between restarts.

## Retrieval strategies
The implementation supports dense retrieval, MMR, BM25, and hybrid retrieval with rank-based fusion. Results are deduplicated and ranked before answer generation.

## LLM integration
The app can connect to OpenAI-compatible providers (including OpenRouter) or a local Ollama model. If no provider is configured, it continues in retrieval-only mode and clearly indicates that no AI-generated answer is available.

## Citation generation
The system retrieves evidence passages and requires the model to return JSON with an answer and valid chunk references. Only referenced chunk IDs from retrieved passages are accepted.

## Evaluation methodology
The project includes example evaluation questions and metrics such as precision@3, recall@3, MRR, hit rate, and nDCG. Results are exportable as JSON or CSV.

## Limitations and future improvements
- Real benchmark datasets are not bundled with the project.
- Inference quality depends on the configured LLM and document collection.
- OCR is required for scanned papers without selectable text.
- Performance can be improved with reranking and query rewriting.
