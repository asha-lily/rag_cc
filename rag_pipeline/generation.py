import logging

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_pipeline.config import GenerationConfig
from rag_pipeline.retrieval import DocumentRetriever

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about travel insurance.
Answer the question using only the context provided below. If the context does not contain enough information to answer the question confidently, say "I don't have enough information to answer that question based on the available documents."

Do not make up information or draw on knowledge outside the provided context.
Cite the source document and page number when you use specific information."""

_HUMAN_TEMPLATE = """Context:
{context}

Question: {question}"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_TEMPLATE),
    ]
)


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a context string with source citations."""
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] Source: {source}, page {page}\n{doc.page_content}")
    return "\n\n".join(parts)


def create_rag_chain(retriever: DocumentRetriever, config: GenerationConfig | None = None):
    """Return an LCEL chain: retriever → prompt → LLM → string.
    This can be invoked with a question string:
    """
    if config is None:
        config = GenerationConfig()

    llm = ChatOllama(model=config.model_name, num_predict=config.max_tokens)

    chain = (
        {"context": retriever | _format_context, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def generate(question: str, retriever: DocumentRetriever, config: GenerationConfig | None = None) -> str:
    """Run a single RAG query and return the answer string."""
    chain = create_rag_chain(retriever, config)
    answer = chain.invoke(question)
    log.info("Q: %s", question)
    log.info("A: %s", answer)
    return answer
