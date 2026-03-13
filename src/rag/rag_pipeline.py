from typing import List
from .keyword_retriever import search_documents_keyword, Document
from ..llm import BedrockLLM


class RAGPipeline:
    """Pipeline RAG che combina retrieval e generazione."""
    
    def __init__(
        self,
        profile_name: str = "personal",
        region_name: str = "eu-west-1",
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        top_k: int = 3
    ):
        """
        Inizializza la pipeline RAG.
        
        Args:
            profile_name: Nome del profilo AWS
            region_name: Regione AWS
            model_id: ID del modello Bedrock
            top_k: Numero di documenti da recuperare
        """
        self.llm = BedrockLLM(
            profile_name=profile_name,
            region_name=region_name,
            model_id=model_id
        )
        self.top_k = top_k
    
    def _build_context(self, documents: List[Document]) -> str:
        """
        Costruisce il contesto dai documenti recuperati.
        
        Args:
            documents: Lista di documenti
            
        Returns:
            Stringa formattata con il contesto
        """
        context_parts = []
        for doc in documents:
            context_parts.append(f"Title: {doc.title}\nText: {doc.text}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, context: str, query: str) -> str:
        """
        Costruisce il prompt per il modello.
        
        Args:
            context: Il contesto dai documenti
            query: La query dell'utente
            
        Returns:
            Il prompt completo
        """
        return f"""You are an AI assistant answering questions using the provided context.

Only use the information contained in the context.
If the answer cannot be found in the context, say: "I don't have enough information in the provided documents."

Context:
{context}

Question:
{query}

Answer:"""
    
    def answer_query(self, query: str) -> str:
        """
        Risponde a una query usando la pipeline RAG.
        
        Args:
            query: La query dell'utente
            
        Returns:
            La risposta generata dal modello
        """
        # 1. Recupera documenti rilevanti
        documents = search_documents_keyword(query, top_k=self.top_k)
        
        if not documents:
            return "No relevant documents found."
        
        print("Retrieved documents:", [d.title for d in documents])
        
        # 2. Costruisce il contesto
        context = self._build_context(documents)
        
        # 3. Costruisce il prompt
        prompt = self._build_prompt(context, query)
        
        # 4. Genera la risposta
        response = self.llm.generate(prompt)
        
        return response
