import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

from rag_pipeline.config import EmbeddingModelConfig, GenerationConfig, RAGASConfig

log = logging.getLogger(__name__)


def _make_ragas_llm(config: RAGASConfig) -> LangchainLLMWrapper:
    """Wrap a Langchain chat Ollama model in a RAGAS LLM interface."""
    return LangchainLLMWrapper(ChatOllama(model=config.model_name))


def _make_ragas_embeddings(config: EmbeddingModelConfig) -> LangchainEmbeddingsWrapper:
    """Wrap a Langchain embedding model in a RAGAS embeddings interface."""
    emb = HuggingFaceEmbeddings(
        model_name=config.model_name,
        encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
    )
    return LangchainEmbeddingsWrapper(emb)


def build_metrics(
    ragas_config: RAGASConfig | None = None,
    emb_config: EmbeddingModelConfig | None = None,
) -> list:
    """Return RAGAS metrics configured to use the local Ollama model & HuggingFace embeddings."""
    if ragas_config is None:
        ragas_config = RAGASConfig()
    if emb_config is None:
        emb_config = EmbeddingModelConfig()
    llm = _make_ragas_llm(ragas_config)
    embeddings = _make_ragas_embeddings(emb_config)
    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]


def evaluate_rag_samples(
    samples: list[dict],
    ragas_config: RAGASConfig | None = None,
    emb_config: EmbeddingModelConfig | None = None,
):
    """Run RAGAS evaluation on a list of collected RAG outputs.

    Each sample dict must have:
        question  (str)
        answer    (str)
        contexts  (list[str])   — the retrieved chunk texts

    Optionally:
        ground_truth (str) — needed for context_precision / context_recall metrics
    """
    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("ground_truth", ""),
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)
    metrics = build_metrics(ragas_config, emb_config)
    log.info("Running RAGAS on %d samples with %d metrics.", len(samples), len(metrics))
    return evaluate(dataset, metrics=metrics)
