"""
Simple Usage Example - UPB RAG System
Demonstrates basic usage of the conversational RAG chain.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from setup_retrieval import setup_retrieval_system
from rag.chain import UPBRAGChain


def main():
    """Simple example of using the RAG chain."""
    
    print("Initializing UPB RAG system...")
    
    # Setup retrieval system
    retriever, _, _ = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    
    # Create RAG chain
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    
    print("\nRAG system ready! Type your questions (or 'salir' to exit)\n")
    print("=" * 70)
    
    # Interactive loop
    while True:
        # Get user question
        question = input("\nTu pregunta: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['salir', 'exit', 'quit']:
            print("\nGracias por usar el asistente UPB. Hasta pronto!")
            break
        
        if question.lower() == 'limpiar':
            rag_chain.clear_history()
            print("Historial de conversación limpiado.")
            continue
        
        # Get response
        print("\nAsistente: ", end="", flush=True)
        response = rag_chain.invoke(question, include_sources=False)
        print(response['answer'])

        if 'sources' in response:
            print("\n[Fragmentos recuperados]")
            for i, source in enumerate(response['sources'], 1):
                print(f"\n{i}. Categoria: {source['category']}")
                print(f"   Archivo: {source['source']}")
                print(f"   Contenido: {source['content']}")

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
