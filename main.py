from src.agents import RAGAgent


def main():
    print("="*70)
    print("Test Agentic RAG con Strands Agents")
    print("="*70)
    
    # Inizializza l'agente RAG
    agent = RAGAgent(
        profile_name="personal",
        region_name="eu-west-1",
        model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )
    
    # Query di test
    queries = [
        "Cos'è l'Agentic RAG?",
        "Cosa sono i vector database?",
        "Traduci hello world in italiano"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)
        
        response = agent.answer(query)
        
        print(f"\nRisposta:\n{response}")


if __name__ == "__main__":
    main()
