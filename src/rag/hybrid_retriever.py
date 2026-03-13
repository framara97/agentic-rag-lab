from typing import List, Dict
from collections import defaultdict
from .retriever import search_documents, Document
from .embedding_retriever import search_documents_semantic


def reciprocal_rank_fusion(
    keyword_docs: List[Document],
    semantic_docs: List[Document],
    k: int = 60
) -> List[Document]:
    """
    Combina i risultati di due retriever usando Reciprocal Rank Fusion (RRF).
    
    Formula RRF: score = 1 / (k + rank)
    dove rank è la posizione del documento (1, 2, 3, ...)
    
    Args:
        keyword_docs: Documenti dal keyword retriever
        semantic_docs: Documenti dal semantic retriever
        k: Costante per la formula RRF (default: 60)
        
    Returns:
        Lista di documenti ordinati per score RRF
    """
    # Dizionario per aggregare gli score: {document_title: score}
    doc_scores: Dict[str, float] = defaultdict(float)
    
    # Dizionario per mantenere i documenti: {document_title: Document}
    doc_map: Dict[str, Document] = {}
    
    # Calcola score RRF per keyword retriever
    for rank, doc in enumerate(keyword_docs, start=1):
        rrf_score = 1.0 / (k + rank)
        doc_scores[doc.title] += rrf_score
        doc_map[doc.title] = doc
    
    # Calcola score RRF per semantic retriever
    for rank, doc in enumerate(semantic_docs, start=1):
        rrf_score = 1.0 / (k + rank)
        doc_scores[doc.title] += rrf_score
        doc_map[doc.title] = doc
    
    # Ordina documenti per score decrescente
    sorted_titles = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Restituisci i documenti ordinati
    return [doc_map[title] for title, _ in sorted_titles]


def search_documents_hybrid(query: str, top_k: int = 3) -> List[Document]:
    """
    Cerca documenti usando hybrid retrieval (keyword + semantic).
    
    Combina i risultati di keyword retrieval e semantic retrieval
    usando Reciprocal Rank Fusion per ottenere un ranking finale.
    
    Args:
        query: La query di ricerca
        top_k: Numero massimo di documenti da restituire
        
    Returns:
        Lista di documenti ordinati per score RRF
    """
    # Recupera documenti da entrambi i retriever
    # Recuperiamo più documenti (top_k * 2) per avere più candidati da fondere
    retrieval_k = max(top_k * 2, 5)
    
    keyword_docs = search_documents(query, top_k=retrieval_k)
    semantic_docs = search_documents_semantic(query, top_k=retrieval_k)
    
    # Combina usando RRF
    fused_docs = reciprocal_rank_fusion(keyword_docs, semantic_docs)
    
    # Restituisci top_k documenti
    return fused_docs[:top_k]
