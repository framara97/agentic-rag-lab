# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

AWS credentials required. Defaults to `personal` AWS profile in `eu-west-1`.
Override with `AWS_PROFILE=your-profile`.

## Commands

```bash
# Run tests
pytest tests/ -v

# Run retrieval comparison + evaluation
python main.py

# Quick REPL usage — see examples below
python -c "from src.agents import RAGAgent; print(RAGAgent().answer('What is RAG?'))"
```

**⚠️ First import of `semantic_retriever` downloads ~90MB model** (`all-MiniLM-L6-v2`).
This happens automatically via the `SEMANTIC_RETRIEVER` singleton at module load time.
Do not move or lazy-load this singleton without understanding the downstream impact on hybrid retrieval.

## Architecture

Two parallel query paths:

**1. Classic RAG** (`src/rag/rag_pipeline.py`): retrieve → prompt → generate.
Uses `BedrockLLM` (`src/llm/bedrock_client.py`) via boto3 directly.

**2. Agentic RAG** (`src/agents/`): LLM decides when/how to retrieve.
Built on [Strands Agents](https://github.com/strands-agents/sdk-python).

<important if="modifying retrieval layer or corpus documents">
`keyword_retriever.py` is the single source of truth for the corpus.
Do NOT define or modify documents elsewhere.
</important>

<important if="refactoring imports or moving modules">
`SEMANTIC_RETRIEVER` singleton downloads ~90MB model at import time.
Do not move or lazy-load without understanding downstream impact.
</important>

### Retrieval layer (`src/rag/`)

All three retrievers share `SAMPLE_DOCUMENTS` from `keyword_retriever.py`.

> **⚠️ Critical:** `keyword_retriever.py` is the single source of truth for the corpus.
> Do NOT define or modify documents elsewhere.

| Retriever | File | Strategy |
|---|---|---|
| Keyword | `keyword_retriever.py` | Token intersection + title bonus |
| Semantic | `semantic_retriever.py` | Cosine similarity over sentence-transformer embeddings |
| Hybrid | `hybrid_retriever.py` | RRF fusion of keyword + semantic (k=60) |

### Agent layer (`src/agents/`)

- **`RAGAgent`**: single tool (`search_documents`), keyword-backed.
- **`RetrievalRouterAgent`**: three tools, routes per query type. Returns documents only — no answer generation.

### Evaluation (`src/evaluation/retrieval_evaluator.py`)

`evaluate_retrievers()` runs 8 queries, reports `Recall@k` per retriever.

## Known issues / migration state

- `rag_pipeline.py` and `semantic_retriever.py` still use `print()` — structured logging not yet implemented (roadmap: Fase 1)
- `hybrid_retriever.py` has a latent bug: `List[Document]` annotation without import — use built-in `list[Document]` (Python 3.10+)
- Strands agent debug logging enabled by default in `rag_agent.py` — set `logging.getLogger("strands").setLevel(logging.WARNING)` to silence

## Bedrock config

Default model: `eu.anthropic.claude-3-7-sonnet-20250219-v1:0` (EU cross-region inference profile).

## Lessons
@.claude/lessons.md