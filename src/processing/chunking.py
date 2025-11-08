"""
Document Chunking Module
Splits documents into smaller chunks optimized for embedding and retrieval.
Supports YAML frontmatter extraction and header-based chunking.
"""

import re
import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def extract_frontmatter(content: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from markdown content.
    
    Args:
        content: Markdown content with optional YAML frontmatter
    
    Returns:
        tuple: (frontmatter_dict, content_without_frontmatter)
    """
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(frontmatter_pattern, content, re.DOTALL)
    
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1))
            content_without_fm = content[match.end():]
            return frontmatter or {}, content_without_fm
        except yaml.YAMLError:
            return {}, content
    
    return {}, content


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200, use_headers=True, add_context_prefix=True):
    """
    Split documents into smaller chunks with YAML frontmatter and header metadata.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk in characters (default: 1000)
        chunk_overlap: Number of characters to overlap between chunks (default: 200)
        use_headers: Use markdown headers for splitting (default: True)
        add_context_prefix: Add contextual prefix to each chunk (default: True)
    
    Returns:
        list: List of chunked Document objects with enriched metadata including:
            - Original document metadata (source, category)
            - YAML frontmatter (title, institution, program_code, etc.)
            - Header hierarchy (Header 1, Header 2, Header 3)
            - Chunk position (start_index)
            - Contextual prefix in content
    """
    if use_headers:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    all_chunks = []
    
    for doc in documents:
        frontmatter, content = extract_frontmatter(doc.page_content)
        
        if use_headers and content.strip():
            try:
                header_splits = markdown_splitter.split_text(content)
            except Exception:
                header_splits = [Document(page_content=content, metadata={})]
        else:
            header_splits = [Document(page_content=content, metadata={})]
        
        for split in header_splits:
            combined_metadata = {
                **doc.metadata,
                **frontmatter,
                **split.metadata
            }
            split.metadata = combined_metadata
        
        sized_chunks = text_splitter.split_documents(header_splits)
        
        if add_context_prefix:
            for chunk in sized_chunks:
                prefix = _build_context_prefix(chunk.metadata)
                chunk.page_content = f"{prefix}\n\n{chunk.page_content}"
        
        all_chunks.extend(sized_chunks)
    
    return all_chunks


def _build_context_prefix(metadata: dict) -> str:
    """
    Build a contextual prefix for each chunk to prevent hallucinations.
    This prefix helps both embeddings and LLM identify the source program.
    
    Args:
        metadata: Chunk metadata dictionary
    
    Returns:
        Formatted context prefix string
    """
    parts = []
    
    if metadata.get('title'):
        parts.append(f"[PROGRAMA: {metadata['title']}]")
    
    if metadata.get('program_code'):
        parts.append(f"[CODIGO: {metadata['program_code']}]")
    
    if metadata.get('program_duration'):
        parts.append(f"[DURACION: {metadata['program_duration']}]")
    
    if metadata.get('category'):
        category_map = {
            'engineering': 'Ingeniería',
            'enrollment': 'Inscripciones',
            'scholarships': 'Becas',
            'contact': 'Contacto',
            'general': 'Información General',
            'metadata': 'Catálogo de Programas'
        }
        cat_label = category_map.get(metadata['category'], metadata['category'])
        parts.append(f"[CATEGORIA: {cat_label}]")
    
    if metadata.get('Header 2'):
        parts.append(f"[SECCION: {metadata['Header 2']}]")
    
    return " ".join(parts) if parts else "[DOCUMENTO UPB]"


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from loader.ingest import load_upb_documents
    
    print("Loading documents...\n")
    documents = load_upb_documents()
    
    print(f"Loaded {len(documents)} documents")
    print(f"Total characters: {sum(len(doc.page_content) for doc in documents):,}\n")
    
    print("Chunking documents with headers and frontmatter...\n")
    chunks = chunk_documents(documents, use_headers=True)
    
    print(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks):,} characters")

    chunk_categories = {}
    for chunk in chunks:
        cat = chunk.metadata.get('category', 'unknown')
        chunk_categories[cat] = chunk_categories.get(cat, 0) + 1

    print("\nChunks by category:")
    for cat, count in sorted(chunk_categories.items()):
        print(f"  - {cat}: {count} chunks")
    
    if chunks:
        print("\nExample chunk metadata:")
        example = chunks[0].metadata
        for key, value in list(example.items())[:10]:
            print(f"  {key}: {value}")
        
        print(f"\nExample chunk content preview:")
        print(chunks[0].page_content[:200] + "...")

    print("\nChunks ready for embedding!")
