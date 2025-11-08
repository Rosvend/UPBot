"""
Comprehensive RAG Evaluation System
Implements retrieval metrics (recall@k, precision@k, context length) and RAGAS evaluation.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    LLMContextPrecisionWithoutReference,
    Faithfulness,
    LLMContextRecall,
    FactualCorrectness
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import AzureChatOpenAI

from setup_retrieval import setup_retrieval_system
from rag.chain import UPBRAGChain


class RAGEvaluator:
    """
    Comprehensive RAG evaluation with retrieval metrics and RAGAS.
    """
    
    def __init__(
        self, 
        retriever, 
        rag_chain: UPBRAGChain,
        eval_questions_path: str = "src/evaluation/eval_questions.json"
    ):
        """
        Initialize evaluator.
        
        Args:
            retriever: UPBRetriever instance
            rag_chain: UPBRAGChain instance for answer generation
            eval_questions_path: Path to evaluation questions JSON
        """
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.eval_questions = self._load_eval_questions(eval_questions_path)
        
        # Initialize Azure OpenAI LLM for RAGAS evaluation
        # Use gpt-4.1-nano instead of o4-mini to avoid temperature restrictions
        # o4-mini only allows temperature=1 and RAGAS tries to override it
        self.evaluator_llm = LangchainLLMWrapper(
            AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
                openai_api_version="2024-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=0.1,  # RAGAS will use this for evaluation
            )
        )
        
    def _load_eval_questions(self, path: str) -> List[Dict]:
        """Load evaluation questions from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_recall_at_k(
        self, 
        query: str, 
        ground_truth_keywords: List[str], 
        k: int = 5,
        retrieval_method: str = "hybrid"
    ) -> float:
        """
        Calculate Recall@k: proportion of relevant keywords found in top-k results.
        
        Args:
            query: Search query
            ground_truth_keywords: List of keywords that should be retrieved
            k: Number of documents to retrieve
            retrieval_method: Retrieval strategy to use
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        # Get retriever based on method
        if retrieval_method == "bm25":
            retriever = self.retriever.get_bm25_retriever(k=k)
        elif retrieval_method == "similarity":
            retriever = self.retriever.get_dense_retriever(k=k, search_type="similarity")
        elif retrieval_method == "mmr":
            retriever = self.retriever.get_dense_retriever(k=k, search_type="mmr")
        else:  # hybrid
            retriever = self.retriever.get_hybrid_retriever(k=k)
        
        # Retrieve documents
        docs = retriever.invoke(query)
        
        # Combine all retrieved content
        retrieved_text = " ".join([doc.page_content.lower() for doc in docs])
        
        # Count how many ground truth keywords are found
        found_keywords = sum(
            1 for keyword in ground_truth_keywords 
            if keyword.lower() in retrieved_text
        )
        
        # Recall = found / total expected
        recall = found_keywords / len(ground_truth_keywords) if ground_truth_keywords else 0.0
        
        return recall
    
    def calculate_precision_at_k(
        self,
        query: str,
        relevant_categories: List[str],
        k: int = 5,
        retrieval_method: str = "hybrid"
    ) -> float:
        """
        Calculate Precision@k: proportion of retrieved docs that are relevant.
        
        Args:
            query: Search query
            relevant_categories: List of categories considered relevant for this query
            k: Number of documents to retrieve
            retrieval_method: Retrieval strategy to use
            
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        # Get retriever based on method
        if retrieval_method == "bm25":
            retriever = self.retriever.get_bm25_retriever(k=k)
        elif retrieval_method == "similarity":
            retriever = self.retriever.get_dense_retriever(k=k, search_type="similarity")
        elif retrieval_method == "mmr":
            retriever = self.retriever.get_dense_retriever(k=k, search_type="mmr")
        else:  # hybrid
            retriever = self.retriever.get_hybrid_retriever(k=k)
        
        # Retrieve documents
        docs = retriever.invoke(query)
        
        # Count relevant documents (based on metadata category)
        relevant_count = sum(
            1 for doc in docs 
            if doc.metadata.get("category", "").lower() in 
            [cat.lower() for cat in relevant_categories]
        )
        
        # Precision = relevant retrieved / total retrieved
        precision = relevant_count / k if k > 0 else 0.0
        
        return precision
    
    def calculate_average_context_length(
        self,
        queries: List[str],
        k: int = 5,
        retrieval_method: str = "hybrid"
    ) -> Dict[str, float]:
        """
        Calculate average context length metrics for retrieved documents.
        
        Args:
            queries: List of queries to test
            k: Number of documents to retrieve per query
            retrieval_method: Retrieval strategy to use
            
        Returns:
            Dict with avg_chars, avg_tokens (estimated), avg_docs metrics
        """
        total_chars = 0
        total_docs = 0
        
        for query in queries:
            # Get retriever
            if retrieval_method == "bm25":
                retriever = self.retriever.get_bm25_retriever(k=k)
            elif retrieval_method == "similarity":
                retriever = self.retriever.get_dense_retriever(k=k, search_type="similarity")
            elif retrieval_method == "mmr":
                retriever = self.retriever.get_dense_retriever(k=k, search_type="mmr")
            else:  # hybrid
                retriever = self.retriever.get_hybrid_retriever(k=k)
            
            # Retrieve documents
            docs = retriever.invoke(query)
            
            # Accumulate stats
            total_docs += len(docs)
            total_chars += sum(len(doc.page_content) for doc in docs)
        
        num_queries = len(queries)
        avg_chars_per_query = total_chars / num_queries if num_queries > 0 else 0
        avg_docs_per_query = total_docs / num_queries if num_queries > 0 else 0
        
        # Rough token estimation (1 token ~= 4 chars for Spanish)
        avg_tokens_per_query = avg_chars_per_query / 4
        
        return {
            "avg_chars_per_query": avg_chars_per_query,
            "avg_tokens_per_query": avg_tokens_per_query,
            "avg_docs_per_query": avg_docs_per_query,
            "total_queries": num_queries
        }
    
    def run_retrieval_evaluation(
        self,
        k_values: List[int] = [3, 5, 10],
        retrieval_methods: List[str] = ["bm25", "similarity", "hybrid"]
    ) -> Dict:
        """
        Run comprehensive retrieval evaluation across different k and methods.
        
        Args:
            k_values: List of k values to test
            retrieval_methods: List of retrieval methods to test
            
        Returns:
            Dict with evaluation results
        """
        print("\n" + "=" * 70)
        print("RETRIEVAL EVALUATION - Recall@k, Precision@k, Context Length")
        print("=" * 70)
        
        results = {
            "by_method": {},
            "by_k": {},
            "summary": {}
        }
        
        # Extract keywords from ground truths for recall calculation
        queries_with_keywords = []
        for item in self.eval_questions:
            # Extract key terms from ground truth
            keywords = self._extract_keywords(item["ground_truth"])
            queries_with_keywords.append({
                "question": item["question"],
                "keywords": keywords,
                "category": item.get("category", "general")
            })
        
        # Test each method and k combination
        for method in retrieval_methods:
            results["by_method"][method] = {}
            
            for k in k_values:
                print(f"\nEvaluating {method} with k={k}...")
                
                recalls = []
                precisions = []
                
                for item in queries_with_keywords:
                    # Calculate recall@k
                    recall = self.calculate_recall_at_k(
                        item["question"],
                        item["keywords"],
                        k=k,
                        retrieval_method=method
                    )
                    recalls.append(recall)
                    
                    # Calculate precision@k (relevant = matches query category)
                    relevant_cats = self._get_relevant_categories(item["category"])
                    precision = self.calculate_precision_at_k(
                        item["question"],
                        relevant_cats,
                        k=k,
                        retrieval_method=method
                    )
                    precisions.append(precision)
                
                avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
                avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
                
                results["by_method"][method][k] = {
                    "recall@k": avg_recall,
                    "precision@k": avg_precision
                }
                
                print(f"  Recall@{k}: {avg_recall:.3f}")
                print(f"  Precision@{k}: {avg_precision:.3f}")
        
        # Calculate average context length
        print("\nCalculating average context length...")
        queries = [item["question"] for item in self.eval_questions]
        
        for method in retrieval_methods:
            context_stats = self.calculate_average_context_length(
                queries, 
                k=5,  # Use k=5 as standard
                retrieval_method=method
            )
            results["by_method"][method]["context_length"] = context_stats
            
            print(f"\n{method.upper()} - Context Statistics:")
            print(f"  Avg chars/query: {context_stats['avg_chars_per_query']:.0f}")
            print(f"  Avg tokens/query: {context_stats['avg_tokens_per_query']:.0f}")
            print(f"  Avg docs/query: {context_stats['avg_docs_per_query']:.1f}")
        
        return results
    
    def run_ragas_evaluation(
        self,
        sample_size: int = None,
        retrieval_method: str = "hybrid"
    ) -> Dict:
        """
        Run RAGAS evaluation with o4-mini for context precision, faithfulness,
        answer correctness, and context recall.
        
        Args:
            sample_size: Number of questions to evaluate (None = all)
            retrieval_method: Retrieval method to use
            
        Returns:
            RAGAS evaluation results
        """
        print("\n" + "=" * 70)
        print("RAGAS EVALUATION - o4-mini Metrics")
        print("=" * 70)
        
        # Prepare dataset
        questions = self.eval_questions[:sample_size] if sample_size else self.eval_questions
        
        print(f"\nGenerating answers for {len(questions)} questions...")
        
        ragas_data = {
            "user_input": [],
            "reference": [],
            "response": [],
            "retrieved_contexts": []
        }
        
        for i, item in enumerate(questions, 1):
            print(f"Processing {i}/{len(questions)}: {item['question'][:60]}...")
            
            # Get answer from RAG chain
            result = self.rag_chain.invoke(
                item["question"], 
                include_sources=True
            )
            
            # Collect data for RAGAS
            ragas_data["user_input"].append(item["question"])
            ragas_data["reference"].append(item["ground_truth"])
            ragas_data["response"].append(result["answer"])
            
            # Extract context strings from sources
            contexts = [src["content"] for src in result.get("sources", [])]
            ragas_data["retrieved_contexts"].append(contexts)
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        # Create RAGAS dataset
        dataset = Dataset.from_dict(ragas_data)
        
        # Initialize metrics with o4-mini
        print("\nInitializing RAGAS metrics with o4-mini...")
        metrics = [
            LLMContextPrecisionWithoutReference(llm=self.evaluator_llm),
            Faithfulness(llm=self.evaluator_llm),
            FactualCorrectness(llm=self.evaluator_llm),
            LLMContextRecall(llm=self.evaluator_llm)
        ]
        
        # Run evaluation
        print("\nRunning RAGAS evaluation (this may take a few minutes)...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        # Display results
        print("\n" + "=" * 70)
        print("RAGAS RESULTS")
        print("=" * 70)
        
        scores_df = result.to_pandas()
        
        print("\nAggregate Metrics:")
        for metric in ["context_precision", "faithfulness", "factual_correctness", "context_recall"]:
            if metric in scores_df.columns:
                avg_score = scores_df[metric].mean()
                print(f"  {metric}: {avg_score:.3f}")
        
        return {
            "ragas_result": result,
            "scores_df": scores_df,
            "aggregate_scores": {
                metric: scores_df[metric].mean() 
                for metric in scores_df.columns 
                if metric in ["context_precision", "faithfulness", "factual_correctness", "context_recall"]
            }
        }
    
    def run_full_evaluation(
        self,
        k_values: List[int] = [3, 5, 10],
        retrieval_methods: List[str] = ["bm25", "similarity", "hybrid"],
        ragas_sample_size: int = None
    ) -> Dict:
        """
        Run complete evaluation suite: retrieval metrics + RAGAS.
        
        Args:
            k_values: k values for retrieval evaluation
            retrieval_methods: Retrieval methods to test
            ragas_sample_size: Number of questions for RAGAS (None = all)
            
        Returns:
            Complete evaluation results
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE RAG EVALUATION SUITE")
        print("=" * 70)
        print(f"Total evaluation questions: {len(self.eval_questions)}")
        print(f"Retrieval methods: {', '.join(retrieval_methods)}")
        print(f"K values: {', '.join(map(str, k_values))}")
        print(f"RAGAS sample size: {ragas_sample_size or 'all'}")
        
        # Run retrieval evaluation
        retrieval_results = self.run_retrieval_evaluation(
            k_values=k_values,
            retrieval_methods=retrieval_methods
        )
        
        # Run RAGAS evaluation
        ragas_results = self.run_ragas_evaluation(
            sample_size=ragas_sample_size,
            retrieval_method="hybrid"  # Use best method for RAGAS
        )
        
        # Combine results
        full_results = {
            "retrieval_metrics": retrieval_results,
            "ragas_metrics": ragas_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "k_values": k_values,
                "retrieval_methods": retrieval_methods,
                "total_questions": len(self.eval_questions),
                "ragas_sample_size": ragas_sample_size
            }
        }
        
        # Save results
        self._save_results(full_results)
        
        # Print summary
        self._print_summary(full_results)
        
        return full_results
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from ground truth text."""
        # Simple keyword extraction - can be enhanced
        import re
        
        # Remove common words and split
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        stopwords = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'es', 'para', 'con', 'por', 'que',
            'su', 'sus', 'son', 'tiene', 'tienen', 'y', 'o', 'a'
        }
        
        words = [w for w in text.split() if len(w) > 3 and w not in stopwords]
        
        # Return unique keywords
        return list(set(words))[:15]  # Limit to 15 keywords
    
    def _get_relevant_categories(self, query_category: str) -> List[str]:
        """Map query category to relevant document categories."""
        # Map to actual categories used in the data (English)
        category_mapping = {
            "Recuperación factual directa": [
                "engineering", "Apoyo financiero", "enrollment", "general", "contact"
            ],
            "Recomendación basada en preferencias": [
                "engineering", "general"
            ],
            "Detección de ausencia": [
                "engineering", "general"
            ],
            "Verificación de plan de estudios": [
                "engineering"
            ]
        }
        
        return category_mapping.get(query_category, ["engineering"])
    
    def _save_results(self, results: Dict):
        """Save evaluation results to JSON file."""
        output_path = Path("src/evaluation/results")
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"eval_results_{timestamp}.json"
        
        # Convert DataFrame to dict for JSON serialization
        if "ragas_metrics" in results:
            results["ragas_metrics"]["scores_list"] = \
                results["ragas_metrics"]["scores_df"].to_dict('records')
            del results["ragas_metrics"]["scores_df"]
            del results["ragas_metrics"]["ragas_result"]  # Can't serialize
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {filename}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        # Best retrieval method
        if "retrieval_metrics" in results:
            print("\nBest Retrieval Performance:")
            best_method = None
            best_score = 0
            
            for method, metrics in results["retrieval_metrics"]["by_method"].items():
                if 5 in metrics:  # Check k=5
                    score = (metrics[5]["recall@k"] + metrics[5]["precision@k"]) / 2
                    if score > best_score:
                        best_score = score
                        best_method = method
            
            if best_method:
                print(f"  Method: {best_method}")
                print(f"  Avg Score: {best_score:.3f}")
        
        # RAGAS summary
        if "ragas_metrics" in results and "aggregate_scores" in results["ragas_metrics"]:
            print("\nRAGAS Metrics Summary:")
            for metric, score in results["ragas_metrics"]["aggregate_scores"].items():
                print(f"  {metric}: {score:.3f}")
        
        print("\n" + "=" * 70)


def main():
    """Run complete evaluation."""
    print("Initializing RAG system for evaluation...")
    
    # Setup retrieval system
    retriever, _, _ = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    
    # Create RAG chain
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    
    # Create evaluator
    evaluator = RAGEvaluator(retriever, rag_chain)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(
        k_values=[3, 5, 10],
        retrieval_methods=["bm25", "similarity", "hybrid"],
        ragas_sample_size=10  # Start with 10 for testing, increase for full eval
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
