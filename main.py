from src.rag import search_documents_keyword, search_documents_semantic, search_documents_hybrid
from src.evaluation import evaluate_retrievers


def main():
    print("="*70)
    print("Confronto Keyword vs Semantic vs Hybrid Retrieval")
    print("="*70)
    
    # Query di test
    queries = [
        "What is an embedding database?",
        "Explain vector storage",
        "How do LLMs retrieve knowledge?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)
        
        # Keyword retrieval
        print("\n[KEYWORD RETRIEVAL]")
        keyword_docs = search_documents_keyword(query, top_k=3)
        print(f"  Found: {len(keyword_docs)} documents")
        if keyword_docs:
            for j, doc in enumerate(keyword_docs, 1):
                print(f"  {j}. {doc.title}")
        else:
            print("  No documents found")
        
        # Semantic retrieval
        print("\n[SEMANTIC RETRIEVAL]")
        semantic_docs = search_documents_semantic(query, top_k=3)
        print(f"  Found: {len(semantic_docs)} documents")
        for j, doc in enumerate(semantic_docs, 1):
            print(f"  {j}. {doc.title}")
        
        # Hybrid retrieval
        print("\n[HYBRID RETRIEVAL (RRF)]")
        hybrid_docs = search_documents_hybrid(query, top_k=3)
        print(f"  Found: {len(hybrid_docs)} documents")
        for j, doc in enumerate(hybrid_docs, 1):
            print(f"  {j}. {doc.title}")
        
        print()
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")
    evaluate_retrievers(k=3)


if __name__ == "__main__":
    main()
