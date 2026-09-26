import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def parse_document(document_path: Path) -> list[Document]:
    """Load a PDF and return one cleaned `Document` (+ metadata) per page."""
    loader = PyPDFLoader(document_path)
    raw_pages = loader.load()

    medicine_name = (
        Path(document_path).stem.replace("_PIL", "").replace("_", " ").capitalize()
    )

    return [
        Document(
            page_content=_clean_text(raw_page.page_content),
            metadata={
                "medicine_name": medicine_name,
                "source": Path(raw_page.metadata["source"]).name,
                "page": raw_page.metadata["page"] + 1,
                "total_pages": raw_page.metadata["total_pages"],
            },
        )
        for raw_page in raw_pages
    ]


def _clean_text(text: str) -> str:
    """Collapse whitespace and limit back-to-back newlines to two."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
