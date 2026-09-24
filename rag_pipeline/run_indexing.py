import logging

from rag_pipeline.chunking import chunk_documents
from rag_pipeline.config import SampleDocsConfig
from rag_pipeline.parse_documents import parse_document
from rag_pipeline.vector_store import (
    create_embedding_model,
    build_vector_store,
    load_vector_store,
    vector_store_exists,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

docs_config = SampleDocsConfig()


def run_indexing():
    # Parse
    pages = []
    for path in [docs_config.doc1_path, docs_config.doc2_path]:
        doc_pages = parse_document(path)
        pages.extend(doc_pages)
        log.info("Parsed %s: %d pages", path.name, len(doc_pages))

    # Chunk
    chunks = chunk_documents(pages)
    log.info("Produced %d chunks from %d pages", len(chunks), len(pages))

    # Embed & index
    embedding_model = create_embedding_model()

    if vector_store_exists():
        log.info("Vector store already exists — loading from disk.")
        store = load_vector_store(embedding_model)
    else:
        log.info("Building vector store from %d chunks...", len(chunks))
        store = build_vector_store(chunks, embedding_model)
        log.info("Vector store built and persisted.")

    return store


if __name__ == "__main__":
    run_indexing()
