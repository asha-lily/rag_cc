from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline.config import EmbeddingModelConfig

config = EmbeddingModelConfig()


def create_embedding_model() -> HuggingFaceEmbeddings:
    """Configure and return a HuggingFace embedding model.

    normalize_embeddings=True makes cosine similarity equivalent to dot product
    on the resulting vectors — slightly faster at search time and the recommended
    setting for BGE models.

    The model is downloaded from HuggingFace on first use and cached locally;
    subsequent calls load from cache.
    """
    return HuggingFaceEmbeddings(
        model_name=config.model_name,
        encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
    )


def embed_chunks(
    chunks: list[Document],
    model: HuggingFaceEmbeddings,
) -> list[list[float]]:
    """Return one embedding vector per chunk, in input order."""
    return model.embed_documents([chunk.page_content for chunk in chunks])
