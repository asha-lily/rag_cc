from pathlib import Path
from dataclasses import dataclass


@dataclass
class SampleDocsConfig:
    doc1_path: Path = Path("data/documents/Document_1.pdf")
    doc2_path: Path = Path("data/documents/Document_2.pdf")


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
