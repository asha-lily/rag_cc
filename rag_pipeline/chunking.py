from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.config import ChunkingConfig

config = ChunkingConfig()


def chunk_documents(
    documents: list[Document],
    chunk_size: int = config.chunk_size,
    chunk_overlap: int = config.chunk_overlap,
) -> list[Document]:
    """Split page-level Documents into chunks.

    Each chunk inherits its source page's metadata (`source`, `page`, ...)
    plus a `chunk_index` marking its position within that page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for document in documents:
        for chunk_index, chunk in enumerate(splitter.split_documents([document])):
            chunk.metadata["chunk_index"] = chunk_index
            chunks.append(chunk)

    return chunks
