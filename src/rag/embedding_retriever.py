import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .retriever import Document, SAMPLE_DOCUMENTS


class EmbeddingRetriever:
    """Retriever basato su embeddings semantici usando sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inizializza il retriever con embeddings.
        
        Args:
            model_name: Nome del modello sentence-transformers da usare
        """
        print(f"Caricamento modello {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Pre-calcola gli embeddings dei documenti
        self.documents = SAMPLE_DOCUMENTS
        self.doc_embeddings = self._compute_document_embeddings()
        print(f"Embeddings calcolati per {len(self.documents)} documenti")
    
    def _compute_document_embeddings(self) -> np.ndarray:
        """
        Calcola gli embeddings per tutti i documenti.
        
        Returns:
            Array numpy con gli embeddings dei documenti
        """
        # Combina titolo e testo per ogni documento
        doc_texts = [f"{doc.title}. {doc.text}" for doc in self.documents]
        
        # Calcola embeddings
        embeddings = self.model.encode(doc_texts, convert_to_numpy=True)
        
        return embeddings
    
    def _cosine_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcola la cosine similarity tra query e documenti.
        
        Args:
            query_embedding: Embedding della query
            doc_embeddings: Embeddings dei documenti
            
        Returns:
            Array con i punteggi di similarità
        """
        # Normalizza i vettori
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # Calcola cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Cerca documenti rilevanti usando semantic similarity.
        
        Args:
            query: La query di ricerca
            top_k: Numero massimo di documenti da restituire
            
        Returns:
            Lista di documenti ordinati per similarità semantica
        """
        # Calcola embedding della query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Calcola similarità con tutti i documenti
        similarities = self._cosine_similarity(query_embedding, self.doc_embeddings)
        
        # Ottieni gli indici dei top_k documenti più simili
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Restituisci i documenti corrispondenti
        return [self.documents[idx] for idx in top_indices]


# Istanza globale del retriever (caricata una sola volta)
SEMANTIC_RETRIEVER = EmbeddingRetriever()


def search_documents_semantic(query: str, top_k: int = 3) -> List[Document]:
    """
    Funzione helper per cercare documenti usando semantic retrieval.
    
    Usa un'istanza globale del retriever per evitare di ricaricare
    il modello ad ogni chiamata.
    
    Args:
        query: La query di ricerca
        top_k: Numero massimo di documenti da restituire
        
    Returns:
        Lista di documenti ordinati per similarità semantica
    """
    return SEMANTIC_RETRIEVER.search(query, top_k=top_k)
