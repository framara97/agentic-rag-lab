# Agentic RAG Lab

Experimental implementation of an Agentic Retrieval-Augmented Generation (RAG) system using Amazon Bedrock and Strands Agents.

## Features

- Bedrock integration with Claude
- Tool-based document retrieval
- Agentic reasoning with Strands Agents
- Keyword retrieval pipeline
- Semantic retrieval (embeddings) – work in progress

## Architecture

```
User Query
    ↓
Agent (Strands)
    ↓
Tool Call → Retriever
    ↓
Documents
    ↓
LLM (Claude via Bedrock)
    ↓
Final Answer
```

## Tech Stack

- Python
- Amazon Bedrock
- Claude
- Strands Agents
- RAG architecture

## Example

**Query:** What is Agentic RAG?

**Agent:** Calls retrieval tool → retrieves documents → generates grounded answer.

## Future Improvements

- Semantic search with embeddings
- Vector database integration
- Evaluation metrics for hallucinations
