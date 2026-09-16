Building an end-to-end RAG pipeline.

# Setup

1. Install dependencies (creates the virtual environment):

       uv sync

2. Install the pre-commit hooks (one-time, per clone):

       uv tool install pre-commit   # if you don't already have it
       pre-commit install

Ruff now runs automatically on staged files at every commit.

<br>

# RAG Pipeline Components

### 1. Ingestion & Parsing
- Convert raw data (e.g PDFs) into structured text.
- Preserve metadata, e.g page numbers, section titles, source, date etc

    *Choose a parsing method & what metadata to preserve.*

### 2. Chunking
- Split documents into chunks to be embedded. 
- Chunks should be large enough to capture context, but small enough that they contain a distinct idea which could be retrieved to answer a question.

    *Choose a chunking method based on the types of documents.*

### 3. Embedding
- Convert document chunks & the query into vectors.

    *Choose an embedding model.*

### 4. Vector Store
- Where embeddings are stored and retrieved from.

    *Choose a vector store, indexing method and metadata filtering.*

### 5. Query processing
- Query re-writing
- HyDE: hypothetical document embeddings
- multi-query generation
- decomposition of complex questions into sub-questions

### 6. Retrieval
- Retrieve the k most relevant chunks

    *Choose k and a search technique.*

### 7. Re-ranking
- Cross-encoder or LLM that assigns a relevance score to each of the initial retrieved chunks
- Retain the n-most relevant chunks

    *Choose a re-ranking model and value of n*

### 8. Context assembly & prompting
- Construct the final prompt
- Include instructions for when the model doesn't know the answer

    *Choose chunk order and how metadata is injected.*

### 9. Generation
- Pass the query + context to the LLM

    *Choose which LLM is most suitable, temperature, response schema / structured output format*

### 10. Evaluation
- retrieval quality: precision, recall@k, MRR
- generation quality: faithfulness/groundedness, answer relevance, hallucination rate
- frameworks: RAGAS, LLM-as-a-judge

    *Choose metrics & frameworks.*

### 11. Serving & Orchestration
- caching
- async / streaming
- latency budgets
- updates to the vectore store as new documents become available

### 12. Observability & Monitoring
- logging queries, retrieved chunks & generations
    - LangSmith?
- record latency at each stage
- capture user feedback to help find failure modes

<br>
<br>

# Breaking down the problem

First we want to load, parse, chunk and index the documents into the vector store. This can be considered an 'offline' phase.

Once we have a vector store, we can perform retrieval and generation, i.e get an answer to a query. 

So, we actually need to build 2 pipelines, one for each of the above stages.

In this project we'll use LangChain.

Once we have a basic RAG system working, we want to evaluate its performance. From there we can consider adding re-ranking, query processing, context assembly etc and see whether each change improves performance.


# Sample Documents

I am using my travel insurance documents - specifically the ones that don't contain any of my personal information. This is a good use case for RAG since for a single travel insurance policy there are multiple documents, some of which are dense with information.

# Running Indexing

`uv run python -m rag_pipeline.run_indexing`


# Future Work

Parsing
- How are tables extracted using `PDFPlumberLoader`?
- How can we extract images / diagrams? `UnstructuredPDFLoader`
- How do the in-built langchain loaders compare to docling



# Questions

- where are classes suitable?
- how does logging work? E.g in run_rag.py