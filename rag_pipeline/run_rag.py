import argparse
import logging

from rag_pipeline.generation import generate
from rag_pipeline.retrieval import create_retriever
from rag_pipeline.vector_store import create_embedding_model, load_vector_store, vector_store_exists

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def run_rag(question: str) -> str:
    if not vector_store_exists():
        raise RuntimeError("Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first.")

    embedding_model = create_embedding_model()
    store = load_vector_store(embedding_model)
    retriever = create_retriever(store, embedding_model)

    return generate(question, retriever)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question about the travel insurance documents.")
    parser.add_argument("question", type=str, help="The question to answer.")
    args = parser.parse_args()

    answer = run_rag(args.question)
