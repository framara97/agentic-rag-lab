import os
import logging
import boto3
from strands import Agent, tool
from strands.models import BedrockModel
from strands.handlers.callback_handler import PrintingCallbackHandler
from ..tools import search_documents_tool


# Configura logging per vedere il reasoning dell'agente
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)


# Wrapper del tool per Strands Agents
@tool
def search_documents(query: str) -> str:
    """
    Search for relevant documents in the knowledge base.
    
    Use this tool when you need to find information about:
    - RAG (Retrieval-Augmented Generation)
    - Agentic RAG systems
    - Multi-agent systems
    - Vector databases
    - Amazon Bedrock
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        A formatted string containing the retrieved documents with their titles and content
    """
    return search_documents_tool(query)


class RAGAgent:
    """Agente che usa Strands Agents con tool di ricerca documenti."""
    
    def __init__(
        self,
        profile_name: str = "personal",
        region_name: str = "eu-west-1",
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
    ):
        """
        Inizializza l'agente RAG.
        
        Args:
            profile_name: Nome del profilo AWS
            region_name: Regione AWS
            model_id: ID del modello Bedrock
        """
        # Imposta il profilo AWS come variabile d'ambiente per boto3
        os.environ["AWS_PROFILE"] = profile_name
        
        # Crea il modello Bedrock usando Strands
        self.model = BedrockModel(
            model_id=model_id,
            region_name=region_name
        )
        
        # Crea l'agente con il tool di ricerca e callback handler verbose
        self.agent = Agent(
            model=self.model,
            tools=[search_documents],
            callback_handler=PrintingCallbackHandler(verbose_tool_use=True),
            system_prompt="""You are a helpful AI assistant with access to a knowledge base.

When answering questions:
1. Use the search_documents tool ONLY when the question requires knowledge from the knowledge base
2. If the question can be answered without external information, answer directly
3. Base your answers on the information found in the documents when using the tool
4. If the information is not in the documents, clearly state that you don't have that information
5. Be concise and accurate in your responses"""
        )
    
    def answer(self, query: str) -> str:
        """
        Risponde a una query usando l'agente RAG.
        
        Args:
            query: La domanda dell'utente
            
        Returns:
            La risposta dell'agente come stringa di testo
        """
        result = self.agent(query)
        
        # Estrae il testo dalla struttura della risposta di Strands
        # La struttura è: {'role': 'assistant', 'content': [{'text': '...'}]}
        if isinstance(result.message, dict):
            content = result.message.get("content", [])
            if content and isinstance(content, list):
                # Concatena tutti i blocchi di testo nella risposta
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                return "\n".join(text_parts) if text_parts else str(result.message)
        
        # Fallback: restituisce la rappresentazione stringa del messaggio
        return str(result.message)
