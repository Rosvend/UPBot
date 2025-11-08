"""
Quick Evaluation Runner
Run this script to perform a complete evaluation of the RAG system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.eval import RAGEvaluator
from setup_retrieval import setup_retrieval_system
from rag.chain import UPBRAGChain


def run_quick_eval():
    """Run a quick evaluation with a subset of questions."""
    print("Starting Quick Evaluation (10 questions)...")
    print("This will test retrieval metrics and RAGAS metrics.\n")
    
    # Setup
    print("Setting up RAG system...")
    retriever, _, _ = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    evaluator = RAGEvaluator(retriever, rag_chain)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        k_values=[3, 5],
        retrieval_methods=["hybrid"],
        ragas_sample_size=10
    )
    
    print("\nQuick evaluation complete!")
    return results


def run_full_eval():
    """Run a complete evaluation with all questions."""
    print("Starting Full Evaluation (all questions)...")
    print("This may take 10-15 minutes depending on API rate limits.\n")
    
    # Setup
    print("Setting up RAG system...")
    retriever, _, _ = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    evaluator = RAGEvaluator(retriever, rag_chain)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        k_values=[3, 5, 10],
        retrieval_methods=["bm25", "similarity", "hybrid"],
        ragas_sample_size=None  # All questions
    )
    
    print("\nFull evaluation complete!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Evaluation mode: 'quick' (10 questions) or 'full' (all questions)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_eval()
    else:
        run_full_eval()
