# Architecture

```mermaid
flowchart LR
    User[User] --> UI[Streamlit UI]
    UI --> Library[Document Library]
    UI --> Chat[Research Chat]
    UI --> Lab[Retrieval Lab]
    UI --> Eval[Evaluation Dashboard]
    Library --> Loader[PDF Loader]
    Loader --> Splitter[Chunking]
    Splitter --> Embeddings[Sentence-Transformers embeddings]
    Embeddings --> Chroma[ChromaDB vector store]
    Chat --> RAG[Retrieval + RAG pipeline]
    Lab --> Retrieval[Dense / BM25 / hybrid]
    Retrieval --> Chroma
    RAG --> LLM[OpenAI-compatible / Ollama]
    RAG --> Citations[Citation validation]
    Eval --> Metrics[Evaluation metrics and reports]
```

## Component notes
- The PDF loader extracts page-level text and preserves page numbers.
- The chunker creates overlapping text units with stable chunk IDs.
- The embedding manager stores vectors in persistent ChromaDB collections.
- Retrieval combines dense semantic matching and BM25 keyword evidence.
- The RAG layer prompts the LLM with retrieved passages and validates citations.
