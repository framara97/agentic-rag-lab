from typing import List, Tuple
from ..rag import search_documents, search_documents_semantic, search_documents_hybrid, Document


# Evaluation dataset: (query, expected_document_title)
EVAL_DATASET = [
    ("What is RAG?", "Introduzione al RAG"),
    ("How do agents work together?", "Sistemi Multi-Agente"),
    ("What is Agentic RAG?", "Agentic RAG"),
    ("What are vector databases?", "Vector Databases"),
    ("What is Amazon Bedrock?", "Amazon Bedrock"),
    ("Explain retrieval augmented generation", "Introduzione al RAG"),
    ("How do multiple agents collaborate?", "Sistemi Multi-Agente"),
    ("What are embedding databases?", "Vector Databases"),
]


def calculate_recall_at_k(
    queries: List[str],
    expected_titles: List[str],
    retriever_func,
    k: int = 3,
    verbose: bool = False
) -> float:
    """
    Calcola Recall@k per un retriever.
    
    Args:
        queries: Lista di query
        expected_titles: Lista di titoli attesi (uno per query)
        retriever_func: Funzione di retrieval da testare
        k: Numero di documenti da recuperare
        verbose: Se True, stampa i risultati per ogni query
        
    Returns:
        Recall@k (percentuale di query per cui il documento atteso è nei top-k)
    """
    hits = 0
    
    for query, expected_title in zip(queries, expected_titles):
        # Recupera documenti
        retrieved_docs = retriever_func(query, top_k=k)
        
        # Verifica se il documento atteso è nei risultati
        retrieved_titles = [doc.title for doc in retrieved_docs]
        hit = expected_title in retrieved_titles
        
        if hit:
            hits += 1
        
        # Stampa dettagli se verbose
        if verbose:
            status = "✓" if hit else "✗"
            print(f"\n{status} Query: {query}")
            print(f"  Expected: {expected_title}")
            print(f"  Retrieved: {retrieved_titles}")
    
    recall = hits / len(queries) if queries else 0.0
    return recall


def evaluate_retrievers(k: int = 3, verbose: bool = True) -> None:
    """
    Valuta e confronta keyword, semantic e hybrid retriever.
    
    Args:
        k: Numero di documenti da recuperare per il calcolo di Recall@k
        verbose: Se True, mostra i risultati per ogni query
    """
    # Prepara dataset
    queries = [item[0] for item in EVAL_DATASET]
    expected_titles = [item[1] for item in EVAL_DATASET]
    
    print("="*70)
    print("Retrieval Evaluation")
    print("="*70)
    print(f"\nDataset size: {len(queries)} queries")
    print(f"Metric: Recall@{k}\n")
    
    # Valuta keyword retriever
    print("="*70)
    print("[KEYWORD RETRIEVER]")
    print("="*70)
    keyword_recall = calculate_recall_at_k(
        queries, expected_titles, search_documents, k=k, verbose=verbose
    )
    print(f"\nRecall@{k}: {keyword_recall:.2f}")
    
    # Valuta semantic retriever
    print("\n" + "="*70)
    print("[SEMANTIC RETRIEVER]")
    print("="*70)
    semantic_recall = calculate_recall_at_k(
        queries, expected_titles, search_documents_semantic, k=k, verbose=verbose
    )
    print(f"\nRecall@{k}: {semantic_recall:.2f}")
    
    # Valuta hybrid retriever
    print("\n" + "="*70)
    print("[HYBRID RETRIEVER (RRF)]")
    print("="*70)
    hybrid_recall = calculate_recall_at_k(
        queries, expected_titles, search_documents_hybrid, k=k, verbose=verbose
    )
    print(f"\nRecall@{k}: {hybrid_recall:.2f}")
    
    # Confronto
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Keyword:  {keyword_recall:.2f}")
    print(f"Semantic: {semantic_recall:.2f}")
    print(f"Hybrid:   {hybrid_recall:.2f}")
    
    # Determina il migliore
    best_recall = max(keyword_recall, semantic_recall, hybrid_recall)
    if hybrid_recall == best_recall:
        print("\n✓ Hybrid retriever performs best")
    elif semantic_recall == best_recall:
        print("\n✓ Semantic retriever performs best")
    else:
        print("\n✓ Keyword retriever performs best")
