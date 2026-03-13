from ..rag.retriever import search_documents_keyword


def search_documents_tool(query: str, top_k: int = 3) -> str:
    """
    Tool per cercare documenti rilevanti.
    
    Questo tool può essere utilizzato da un agente LLM per recuperare
    informazioni da una knowledge base.
    
    Args:
        query: La query di ricerca
        top_k: Numero massimo di documenti da recuperare
        
    Returns:
        Stringa formattata con i documenti trovati
    """
    documents = search_documents_keyword(query, top_k=top_k)
    
    if not documents:
        return "No relevant documents found for the query."
    
    # Formatta i documenti
    formatted_docs = []
    for doc in documents:
        formatted_docs.append(f"Title: {doc.title}\nText: {doc.text}")
    
    return "Retrieved documents:\n\n" + "\n\n".join(formatted_docs)
