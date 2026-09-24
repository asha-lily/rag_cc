import logging

from rag_pipeline.evaluation import evaluate_rag_samples
from rag_pipeline.generation import create_rag_chain
from rag_pipeline.retrieval import create_retriever
from rag_pipeline.vector_store import (
    create_embedding_model,
    load_vector_store,
    vector_store_exists,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# A small test set of questions about the travel insurance documents.
# Fill in ground_truth strings once you know the correct answers from the PDFs;
# they are only required for context_precision and context_recall metrics.
TEST_SET = [
    {"question": "What is the maximum cancellation cover?", "ground_truth": ""},
    {"question": "What is the emergency medical expenses limit?", "ground_truth": ""},
    {
        "question": "Is there cover for pre-existing medical conditions?",
        "ground_truth": "",
    },
    {"question": "What is the excess amount on the policy?", "ground_truth": ""},
    {
        "question": "Is baggage cover included, and what is the limit?",
        "ground_truth": "",
    },
    {"question": "How do I make a claim?", "ground_truth": ""},
    {
        "question": "Are winter sports or adventure activities covered?",
        "ground_truth": "",
    },
    {
        "question": "What is the maximum trip duration that is covered?",
        "ground_truth": "",
    },
]


def _collect_sample(question: str, retriever, chain) -> dict:
    """Run retrieval and generation for one question, capturing both contexts and answer."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    answer = chain.invoke(question)
    return {"question": question, "answer": answer, "contexts": contexts}


def run_evaluation() -> None:
    if not vector_store_exists():
        raise RuntimeError(
            "Vector store not found. Run `uv run python -m rag_pipeline.run_indexing` first."
        )

    embedding_model = create_embedding_model()
    store = load_vector_store(embedding_model)
    retriever = create_retriever(store, embedding_model)
    chain = create_rag_chain(retriever)

    log.info("Collecting answers for %d test questions...", len(TEST_SET))
    samples = []
    for entry in TEST_SET:
        q = entry["question"]
        log.info("  Q: %s", q)
        sample = _collect_sample(q, retriever, chain)
        sample["ground_truth"] = entry.get("ground_truth", "")
        samples.append(sample)

    log.info("Running RAGAS evaluation...")
    result = evaluate_rag_samples(samples)

    print("\n=== RAGAS Results ===")
    print(result)
    print()
    df = result.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))


if __name__ == "__main__":
    run_evaluation()
