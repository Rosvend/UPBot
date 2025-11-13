# UPBot: Your AI Career Exploration Assistant

Unsure what to study? Need quick, reliable information about engineering programs at Universidad Pontificia Bolivariana (UPB)?
Meet UPBot — an intelligent, conversational assistant built to help prospective students explore UPB’s academic offerings in natural Spanish.

UPBot is powered by a Retrieval-Augmented Generation (RAG) system designed for accurate, context-aware, and hallucination-resistant responses.

## Features

- **Conversational AI** with GPT-4o-mini for natural interactions in Spanish
- **Anti-Hallucination System**: Strict context prefixes and rules prevent mixing information between programs
- **Multi-strategy retrieval**: BM25, Vector Similarity, MMR, and Hybrid RRF
- **Header-based chunking** with semantic structure preservation (H1/H2/H3)
- **Rich metadata extraction** from YAML frontmatter (program codes, duration, accreditation, etc.)
- **Context-Enriched Chunks**: Each chunk tagged with [PROGRAMA:], [CODIGO:], [DURACION:], [CATEGORIA:], [SECCION:]
- **Conversation memory** for multi-turn dialogues
- **Source citations** for transparency
- **Comprehensive Evaluation Suite**: Retrieval metrics (recall@k, precision@k, context length) and RAGAS metrics (context precision, faithfulness, factual correctness, context recall) using o4-mini
- 17 curated documents covering 12 engineering programs plus metadata catalog
- 370 optimized chunks with hierarchical metadata for efficient retrieval
- 15 evaluation questions covering factual retrieval, recommendations, and edge cases

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Azure ChatOpenAI / OpenAI GPT-4o-mini |
| **Embeddings** | AzureOpenAIEmbeddings (text-embedding-3-small) |
| **Vector Store** | FAISS (CPU version) |
| **Framework** | LangChain 1.0.2 |
| **Retrieval** | BM25 (rank-bm25) + MMR + RRF Ensemble |
| **Evaluation** | RAGAS 0.2.15+ with o4-mini |
| **UI** | Gradio *(planned)* |
| **Deployment** | Hugging Face Spaces *(planned)* |
| **Package Manager** | UV |

## Project Structure

```
.
├── data/                      # Curated markdown content
│   ├── about_upb.md          # University information
│   ├── contact/              # Contact information
│   ├── engineerings/         # Engineering program details (12 programs)
│   ├── enroll/               # Enrollment information
│   ├── metadata/             # Program metadata catalog
│   │   └── metadata.json    # Structured program information
│   └── scholarships/         # Financial aid & scholarships
│
├── src/
│   ├── embeddings/
│   │   └── embeddings.py    # Azure/OpenAI embeddings initialization
│   ├── evaluation/
│   │   ├── eval_questions.json  # 15 curated evaluation questions
│   │   ├── eval.py          # Comprehensive evaluation system
│   │   ├── README.md        # Evaluation documentation
│   │   └── results/         # Evaluation results (JSON)
│   ├── loader/
│   │   └── ingest.py        # Document loader with metadata.json support
│   ├── processing/
│   │   └── chunking.py      # Header-based chunking with YAML frontmatter
│   ├── rag/
│   │   └── chain.py         # RAG chain with conversation memory
│   ├── retrieval/
│   │   └── retriever.py     # Multi-strategy retriever (BM25/MMR/Hybrid)
│   ├── vectorstore/
│   │   └── store.py         # FAISS vector store manager
│   ├── pipeline.py          # Document preparation pipeline
│   └── setup_retrieval.py   # Complete retrieval system setup
│
├── run_evaluation.py        # Evaluation runner script
├── vectorstore/             # FAISS index files (gitignored)
└── pyproject.toml           # Dependencies (UV)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- [UV](https://docs.astral.sh/uv/) package manager


### Installation

```bash
# Clone the repository
git clone https://github.com/Rosvend/UPBot.git
cd UPBot

# Install dependencies with UV
uv sync
```

### Configuration

Create a `.env` file with your Azure OpenAI credentials (for embeddings & LLM):

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_LLM_DEPLOYMENT=gpt-4o-mini  # or gpt-4.1-nano (recommended for evaluation)
```

Note: For evaluation, gpt-4.1-nano is recommended because RAGAS metrics need temperature flexibility. o4-mini only supports temperature=1 which causes compatibility issues with RAGAS. The RAG system uses the same Azure OpenAI deployment for both generation and evaluation.

### Running the Pipeline

#### 1. **RAG Chain with Conversation** (Full System)
```bash
# Test complete RAG chain with GPT-4o-mini, memory, and source citations
uv run python src/rag/chain.py
```
**What it does**:
- Sets up complete retrieval system
- Initializes GPT-4o-mini LLM
- Tests multi-turn conversation
- Shows source citations
- Demonstrates conversation memory

**Output**: Complete conversational RAG system test

#### 2. **Complete Retrieval Setup**
```bash
# Set up embeddings, vector store, and all retrieval methods
uv run python src/setup_retrieval.py
```
**What it does**:
- Loads 17 documents (16 MD + 1 metadata catalog)
- Creates 370 context-enriched chunks with anti-hallucination prefixes
- Initializes Azure OpenAI embeddings
- Creates/loads FAISS vector store
- Tests all retrieval methods (BM25, Similarity, MMR, Hybrid)

**Output**: Fully initialized retrieval system ready for RAG

#### 3. **Individual Modules**

**Load Documents**
```bash
uv run python src/loader/ingest.py
```
**Output**: 17 documents (16 MD + metadata.json) with category metadata

**Chunk Documents**
```bash
uv run python src/processing/chunking.py
```
**Output**: 370 chunks with context prefixes (avg ~800 chars)

**Test Embeddings**
```bash
uv run python src/embeddings/embeddings.py
```
**Output**: Embedding model initialization test

**Test Vector Store**
```bash
uv run python src/vectorstore/store.py
```
**Output**: FAISS index creation, save, and load test

**Test Retrieval**
```bash
uv run python src/retrieval/retriever.py
```
**Output**: BM25 retrieval test (no embeddings needed)


## Quick Start Examples

### Interactive Chat
```bash
# Run interactive chat interface
uv run python src/main.py
```

Type your questions in Spanish and the assistant will respond using the RAG system. Commands:
- `salir` - Exit the chat
- `limpiar` - Clear conversation history

### Programmatic Usage

**Basic RAG Query**:
```python
from setup_retrieval import setup_retrieval_system
from rag.chain import UPBRAGChain

# Initialize
retriever, _, _ = setup_retrieval_system()
rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")

# Ask question
response = rag_chain.invoke("¿Qué es la ingeniería de sistemas?")
print(response['answer'])
```

**With Source Citations**:
```python
response = rag_chain.invoke(
    "¿Qué becas están disponibles?",
    include_sources=True
)

print(response['answer'])
print("\nFuentes:")
for source in response['sources']:
    print(f"- {source['category']}: {source['source']}")
```

**Multi-turn Conversation**:
```python
# First question
r1 = rag_chain.invoke("¿Qué ingenierías tienen?")

# Follow-up (uses conversation memory)
r2 = rag_chain.invoke("¿Cuál me recomiendas si me gusta programar?")

# Another follow-up
r3 = rag_chain.invoke("¿Cuánto dura ese programa?")

# Clear history when done
rag_chain.clear_history()
```

## Evaluation

The project includes a comprehensive evaluation system to measure RAG performance.

### Quick Evaluation

```bash
# Run quick evaluation with 10 questions
uv run python run_evaluation.py --mode quick
```

This evaluates:
- Recall@k and Precision@k (retrieval quality)
- Average context length (efficiency)
- RAGAS metrics: context precision, faithfulness, factual correctness, context recall

### Full Evaluation

```bash
# Run full evaluation with all questions and methods
uv run python run_evaluation.py --mode full
```

Tests all retrieval methods (BM25, similarity, hybrid) with multiple k values across all 15 evaluation questions.

### Evaluation Metrics

**Retrieval Metrics**:
- Recall@k: Proportion of relevant information retrieved (0-1, higher better)
- Precision@k: Proportion of retrieved docs that are relevant (0-1, higher better)
- Context Length: Average chars/tokens/docs per query

**RAGAS Metrics** (using o4-mini):
- Context Precision: Relevance of retrieved contexts (0-1, higher better)
- Faithfulness: Answer grounded in context, no hallucinations (0-1, higher better)
- Factual Correctness: Accuracy vs ground truth (0-1, higher better)
- Context Recall: Ground truth coverage by retrieved context (0-1, higher better)

Results are saved to `src/evaluation/results/` with timestamps.

For detailed evaluation documentation, see [src/evaluation/README.md](src/evaluation/README.md).


