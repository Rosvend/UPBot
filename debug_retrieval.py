#!/usr/bin/env python3
"""Debug script to see what chunks are retrieved for different queries."""

from src.setup_retrieval import setup_retrieval_system

def debug_retrieval(query: str):
    """Show what chunks are being retrieved for a query."""
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    print()
    
    retriever, _, _ = setup_retrieval_system()
    
    # Retrieve chunks
    results = retriever.retrieve(query, method="hybrid", k=6)
    
    print(f"Retrieved {len(results)} chunks:")
    print()
    
    for i, doc in enumerate(results, 1):
        content = doc.page_content
        metadata = doc.metadata
        
        # Extract context prefix if exists
        lines = content.split('\n')
        prefix_lines = [l for l in lines if l.startswith('[')]
        content_lines = [l for l in lines if not l.startswith('[')]
        
        print(f"[CHUNK {i}]")
        if prefix_lines:
            print("Context Tags:")
            for line in prefix_lines:
                print(f"  {line}")
        
        print("\nMetadata:")
        for key, value in metadata.items():
            if key not in ['source', 'original_content']:
                print(f"  {key}: {value}")
        
        print("\nContent Preview:")
        preview = '\n'.join(content_lines[:5])
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"  {preview}")
        
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    # The failing query
    debug_retrieval("¿Cuántos semestres tiene la carrera de ingeniería de sistemas?")
    
    # A working query for comparison
    print("\n\n")
    debug_retrieval("¿Cuántos semestres dura la carrera de ingeniería en ciencia de datos?")
