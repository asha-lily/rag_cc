Building a RAG pipeline using Claude Code.

The purpose of this project is to refresh my knowledge of RAG and to gain experience using Claude Code (previously I have only used github copilot).

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
- record latency at each stage
- capture user feedback to help find failure modes

<br>
<br>

# Breaking down the problem

First we want to load, parse, chunk and index the documents into the vector store. This can be considered an 'offline' phase.

Once we have a vector store, we can perform RAG, i.e get an answer to a query. 

So, we actually need to build 2 pipelines, one for each of the above stages.

In this project we'll use LangChain.