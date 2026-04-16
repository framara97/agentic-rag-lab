import os
from strands import Agent, tool
from strands.models import BedrockModel
from ..rag.keyword_retriever import search_documents_keyword
from ..rag.semantic_retriever import search_documents_semantic
from ..rag.hybrid_retriever import search_documents_hybrid


@tool
def keyword_search(query: str) -> str:
    """
    Search documents using keyword matching.
    
    Best for:
    - Queries with specific terms or proper nouns
    - Exact terminology searches
    - Examples: "Amazon Bedrock", "Vector Databases", "Claude"
    
    Args:
        query: The search query
        
    Returns:
        Formatted string with retrieved documents
    """
    documents = search_documents_keyword(query, top_k=3)
    
    if not documents:
        return "No relevant documents found for the query."
    
    formatted_docs = []
    for doc in documents:
        formatted_docs.append(f"Title: {doc.title}\nText: {doc.text}")
    
    return "Retrieved documents:\n\n" + "\n\n".join(formatted_docs)


@tool
def semantic_search(query: str) -> str:
    """
    Search documents using semantic similarity.
    
    Best for:
    - Conceptual queries
    - Questions about general topics
    - Examples: "How do LLMs retrieve knowledge?", "Explain agent collaboration"
    
    Args:
        query: The search query
        
    Returns:
        Formatted string with retrieved documents
    """
    documents = search_documents_semantic(query, top_k=3)
    
    formatted_docs = []
    for doc in documents:
        formatted_docs.append(f"Title: {doc.title}\nText: {doc.text}")
    
    return "Retrieved documents:\n\n" + "\n\n".join(formatted_docs)


@tool
def hybrid_search(query: str) -> str:
    """
    Search documents using hybrid retrieval (keyword + semantic).
    
    Best for:
    - Queries combining specific terms and concepts
    - Complex questions requiring both exact matches and semantic understanding
    - Examples: "How does Amazon Bedrock support RAG systems?"
    
    Args:
        query: The search query
        
    Returns:
        Formatted string with retrieved documents
    """
    documents = search_documents_hybrid(query, top_k=3)
    
    formatted_docs = []
    for doc in documents:
        formatted_docs.append(f"Title: {doc.title}\nText: {doc.text}")
    
    return "Retrieved documents:\n\n" + "\n\n".join(formatted_docs)


class RetrievalRouterAgent:
    """Agent that routes queries to the most appropriate retrieval strategy."""
    
    def __init__(
        self,
        profile_name: str = "personal",
        region_name: str = "eu-west-1",
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
    ):
        """
        Initialize the retrieval router agent.
        
        Args:
            profile_name: AWS profile name
            region_name: AWS region
            model_id: Bedrock model ID
        """
        # Imposta il profilo AWS
        os.environ["AWS_PROFILE"] = profile_name
        
        # Crea il modello Bedrock
        self.model = BedrockModel(
            model_id=model_id,
            region_name=region_name
        )
        
        # Crea l'agente con i tre retrieval tools
        self.agent = Agent(
            model=self.model,
            tools=[keyword_search, semantic_search, hybrid_search],
            system_prompt="""You are a retrieval routing agent. Your job is to select the best retrieval strategy for each query.

Available retrieval strategies:

1. KEYWORD SEARCH (keyword_search)
   - Use for queries with specific terms, proper nouns, or exact terminology
   - Examples: "Amazon Bedrock", "Vector Databases", "Claude model"
   - Best when the user asks about specific named entities

2. SEMANTIC SEARCH (semantic_search)
   - Use for conceptual queries and general topic questions
   - Examples: "How do LLMs retrieve knowledge?", "Explain agent collaboration"
   - Best when the query is about concepts, not specific terms

3. HYBRID SEARCH (hybrid_search)
   - Use for queries combining specific terms AND concepts
   - Examples: "How does Amazon Bedrock support RAG systems?"
   - Best when the query needs both exact matches and semantic understanding

Your task:
1. Analyze the query
2. Choose the most appropriate retrieval strategy
3. Call the corresponding tool
4. Return the retrieved documents

Do NOT generate an answer to the query. Only retrieve and return the documents."""
        )
    
    def retrieve(self, query: str) -> str:
        """
        Route the query to the appropriate retriever and return documents.
        
        Args:
            query: The user's query
            
        Returns:
            Retrieved documents as a formatted string
        """
        result = self.agent(query)
        
        # Estrae il testo dalla risposta
        if isinstance(result.message, dict):
            content = result.message.get("content", [])
            if content and isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                return "\n".join(text_parts) if text_parts else str(result.message)
        
        return str(result.message)
