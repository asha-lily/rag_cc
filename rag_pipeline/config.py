from pathlib import Path
from dataclasses import dataclass


@dataclass
class SampleDocsConfig:
    doc1_path: Path = Path(
        "/Users/ashapatel/Documents/projects/rag_cc/documents/insurance-product-information-document.pdf"
    )
    doc2_path: Path = Path(
        "/Users/ashapatel/Documents/projects/rag_cc/documents/policy-wording.pdf"
    )


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150


@dataclass
class EmbeddingModelConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    normalize_embeddings: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "
