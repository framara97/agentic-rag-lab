# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

AWS credentials are required. The code defaults to the `personal` AWS profile in region `eu-west-1`. Set `AWS_PROFILE` if using a different profile.

## Running the system

There are no top-level entrypoint scripts — run via Python's `-c` flag or a REPL:

```python
# Simple RAG agent (keyword search only)
from src.agents import RAGAgent
agent = RAGAgent()
print(agent.answer("What is Agentic RAG?"))

# Router agent (selects best retrieval strategy per query)
from src.agents import RetrievalRouterAgent
agent = RetrievalRouterAgent()
print(agent.retrieve("How does Amazon Bedrock support RAG?"))

# Classic RAG pipeline (no Strands, direct Bedrock call)
from src.rag import RAGPipeline
pipeline = RAGPipeline()
print(pipeline.answer_query("What are vector databases?"))

# Evaluate all three retrievers
from src.evaluation.retrieval_evaluator import evaluate_retrievers
evaluate_retrievers(k=3, verbose=True)
```

## Architecture

The system has two parallel query paths:

**1. Classic RAG pipeline** (`src/rag/rag_pipeline.py`): synchronous retrieve → prompt → generate. Uses `BedrockLLM` (`src/llm/bedrock_client.py`) which calls `bedrock-runtime` directly via boto3.

**2. Agentic RAG** (`src/agents/`): the LLM decides when and how to retrieve. Built on [Strands Agents](https://github.com/strands-agents/sdk-python). The agent receives retrieval tools wrapped with the `@tool` decorator and reasons about which to call.

### Retrieval layer (`src/rag/`)

Three retrieval strategies share the same in-memory `SAMPLE_DOCUMENTS` corpus (defined in `keyword_retriever.py`):

- **Keyword** (`keyword_retriever.py`): token intersection scoring with title bonus weights.
- **Semantic** (`semantic_retriever.py`): cosine similarity over sentence-transformer embeddings (`all-MiniLM-L6-v2`). A global `SEMANTIC_RETRIEVER` singleton is loaded at import time — importing this module downloads/loads the model.
- **Hybrid** (`hybrid_retriever.py`): combines keyword + semantic results via Reciprocal Rank Fusion (RRF, k=60).

### Agent layer (`src/agents/`)

- **`RAGAgent`**: single tool (`search_documents`) backed by keyword retrieval. The agent decides whether to call it.
- **`RetrievalRouterAgent`**: three tools (`keyword_search`, `semantic_search`, `hybrid_search`). The agent routes each query to the most appropriate strategy based on the system prompt's routing rules. Returns retrieved documents only — no final answer generation.

### Tool bridge (`src/tools/search_tool.py`)

Thin wrapper used by `RAGAgent` to adapt the keyword retriever into a Strands-compatible tool format.

### Evaluation (`src/evaluation/retrieval_evaluator.py`)

`evaluate_retrievers()` runs an 8-query dataset against all three retrievers and reports `Recall@k` for each, then prints a comparison.

## Key design notes

- The `Document` dataclass and `SAMPLE_DOCUMENTS` corpus live in `keyword_retriever.py` and are imported by the other retrievers — this file is the single source of truth for the corpus.
- Bedrock model default: `eu.anthropic.claude-3-7-sonnet-20250219-v1:0` (EU cross-region inference profile).
- Strands agent debug logging is enabled by default in `rag_agent.py`; set `logging.getLogger("strands").setLevel(logging.WARNING)` to silence it.
