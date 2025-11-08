# RAG Evaluation System

This evaluation system provides comprehensive metrics for assessing the RAG system's performance including retrieval quality and answer generation quality.

## Features

1. Retrieval Metrics
   - Recall@k: Measures how many relevant keywords are retrieved in top-k documents
   - Precision@k: Measures proportion of retrieved documents that are relevant
   - Average Context Length: Tracks context size (chars, tokens, doc count)

2. RAGAS Metrics (using o4-mini)
   - Context Precision: Relevance of retrieved contexts to the query
   - Faithfulness: How grounded the answer is in the retrieved context
   - Factual Correctness: Accuracy of the answer compared to ground truth
   - Context Recall: How much of the ground truth is supported by retrieved context

## Evaluation Questions

The evaluation uses 15 carefully crafted questions in `src/evaluation/eval_questions.json` covering:
- Factual retrieval (admission requirements, costs, duration, accreditations)
- Recommendations (program suggestions based on interests)
- Absence detection (programs that don't exist)
- Plan verification (specific courses in programs)

## Usage

### Quick Evaluation (10 questions)

```bash
uv run python run_evaluation.py --mode quick
```

Evaluates with 10 questions, tests hybrid retrieval only. Takes ~5 minutes.

### Full Evaluation (all questions)

```bash
uv run python run_evaluation.py --mode full
```

Evaluates all 15 questions, tests all retrieval methods (BM25, similarity, hybrid) with multiple k values. Takes ~15 minutes.


## Metrics Explained

### Recall@k
Measures what proportion of expected keywords/information is found in the top-k retrieved documents.
- Score: 0.0 to 1.0 (higher is better)
- Example: If 8 out of 10 expected keywords are found, recall@k = 0.8

### Precision@k
Measures what proportion of retrieved documents are actually relevant to the query.
- Score: 0.0 to 1.0 (higher is better)
- Example: If 4 out of 5 retrieved docs are relevant, precision@k = 0.8

### Average Context Length
Tracks the amount of context passed to the LLM:
- `avg_chars_per_query`: Average character count
- `avg_tokens_per_query`: Estimated token count (chars/4 for Spanish)
- `avg_docs_per_query`: Average number of documents retrieved

### Context Precision (RAGAS)
Evaluates whether the retrieved contexts are relevant to answering the query. Uses LLM to judge relevance.
- Score: 0.0 to 1.0 (higher is better)
- Higher scores mean less irrelevant context

### Faithfulness (RAGAS)
Measures if the generated answer is grounded in the retrieved context (no hallucinations).
- Score: 0.0 to 1.0 (higher is better)
- Checks if claims in the answer are supported by the context

### Factual Correctness (RAGAS)
Compares the generated answer to the ground truth reference answer.
- Score: 0.0 to 1.0 (higher is better)
- Measures semantic similarity and factual accuracy

### Context Recall (RAGAS)
Measures how much of the ground truth answer is supported by the retrieved context.
- Score: 0.0 to 1.0 (higher is better)
- High scores mean the system retrieved relevant information

## Output

Results are saved to `src/evaluation/results/eval_results_TIMESTAMP.json` with:
- Retrieval metrics for each method and k value
- RAGAS scores for each question
- Aggregate statistics
- Configuration used

Example output structure:
```json
{
  "retrieval_metrics": {
    "by_method": {
      "hybrid": {
        "5": {
          "recall@k": 0.75,
          "precision@k": 0.82
        },
        "context_length": {
          "avg_chars_per_query": 4200,
          "avg_tokens_per_query": 1050,
          "avg_docs_per_query": 5.0
        }
      }
    }
  },
  "ragas_metrics": {
    "aggregate_scores": {
      "context_precision": 0.78,
      "faithfulness": 0.85,
      "factual_correctness": 0.81,
      "context_recall": 0.73
    }
  }
}
```


## Requirements

Install dependencies:
```bash
uv sync
```

Required environment variables:
```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4.1-nano  # Recommended for RAGAS (flexible temperature)
```

Note: RAGAS evaluation uses your Azure OpenAI deployment. The system uses gpt-4.1-nano by default because o4-mini only supports temperature=1, but RAGAS metrics need temperature flexibility. No separate OpenAI API key needed.