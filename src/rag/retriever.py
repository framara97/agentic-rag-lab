import re
from dataclasses import dataclass
from typing import List, Set


@dataclass
class Document:
    """Rappresenta un documento con titolo e testo."""
    title: str
    text: str


# Documenti di esempio sul tema Agentic RAG e AI
SAMPLE_DOCUMENTS = [
    Document(
        title="Introduzione al RAG",
        text="Il Retrieval-Augmented Generation (RAG) è una tecnica che combina il recupero di informazioni "
             "con la generazione di testo. Permette ai modelli linguistici di accedere a conoscenze esterne "
             "per produrre risposte più accurate e aggiornate."
    ),
    Document(
        title="Sistemi Multi-Agente",
        text="I sistemi multi-agente sono architetture dove più agenti autonomi collaborano per risolvere "
             "problemi complessi. Ogni agente ha competenze specifiche e può comunicare con gli altri "
             "per coordinare le attività e raggiungere obiettivi comuni."
    ),
    Document(
        title="Agentic RAG",
        text="L'Agentic RAG estende il RAG tradizionale introducendo agenti intelligenti che possono "
             "pianificare, ragionare e prendere decisioni su come recuperare e utilizzare le informazioni. "
             "Gli agenti possono scomporre query complesse, orchestrare ricerche multiple e sintetizzare "
             "risultati da diverse fonti."
    ),
    Document(
        title="Vector Databases",
        text="I vector database sono sistemi ottimizzati per memorizzare e ricercare embedding vettoriali. "
             "Sono fondamentali nei sistemi RAG moderni perché permettono di trovare documenti semanticamente "
             "simili attraverso ricerche di similarità vettoriale."
    ),
    Document(
        title="Amazon Bedrock",
        text="Amazon Bedrock è un servizio AWS che fornisce accesso a modelli linguistici di grandi dimensioni "
             "tramite API. Include modelli come Claude di Anthropic, e offre funzionalità per costruire "
             "applicazioni AI generative scalabili e sicure."
    ),
]


def tokenize(text: str) -> Set[str]:
    """
    Tokenizza il testo estraendo le parole e rimuovendo la punteggiatura.
    
    Args:
        text: Il testo da tokenizzare
        
    Returns:
        Set di token in lowercase
    """
    return set(re.findall(r"\w+", text.lower()))


def search_documents_keyword(query: str, top_k: int = 3) -> List[Document]:
    """
    Cerca documenti rilevanti usando keyword matching.
    
    Args:
        query: La query di ricerca
        top_k: Numero massimo di documenti da restituire
        
    Returns:
        Lista di documenti ordinati per rilevanza
    """
    query_terms = tokenize(query)
    
    # Calcola score per ogni documento
    scored_docs = []
    for doc in SAMPLE_DOCUMENTS:
        # Combina titolo e testo per la ricerca
        doc_text = f"{doc.title} {doc.text}"
        doc_terms = tokenize(doc_text)
        
        # Score basato su:
        # 1. Numero di termini della query presenti nel documento
        # 2. Presenza nel titolo (peso maggiore)
        matching_terms = query_terms.intersection(doc_terms)
        score = len(matching_terms)
        
        # Bonus se i termini appaiono nel titolo
        title_terms = tokenize(doc.title)
        for term in query_terms:
            if term in title_terms:
                score += 2
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Ordina per score decrescente e restituisci top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]
