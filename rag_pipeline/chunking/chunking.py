from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
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
