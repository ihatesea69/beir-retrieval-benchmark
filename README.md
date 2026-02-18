# BM25 vs RAG Retrieval Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BeIR Benchmark](https://img.shields.io/badge/benchmark-BeIR-orange.svg)](https://github.com/beir-cellar/beir)
[![LlamaIndex](https://img.shields.io/badge/framework-LlamaIndex-purple.svg)](https://www.llamaindex.ai/)
[![PostgreSQL + pgvector](https://img.shields.io/badge/database-PostgreSQL%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)

A research project benchmarking **BM25 (sparse retrieval)** against **RAG (dense retrieval)** and **Hybrid (BM25 + Dense with Reciprocal Rank Fusion)** on the [BeIR benchmark](https://github.com/beir-cellar/beir), powered by LlamaIndex and PostgreSQL with pgvector.

## Overview

This project provides a reproducible evaluation pipeline for comparing three information retrieval strategies:

| Strategy | Type | Description |
|----------|------|-------------|
| BM25 | Sparse / Lexical | Classic term-frequency based retrieval using inverted index |
| Dense (RAG) | Semantic / Vector | Embedding-based retrieval with sentence-transformers + pgvector |
| Hybrid | Combined | Reciprocal Rank Fusion (RRF) merging BM25 and Dense results |

Evaluation metrics include NDCG@k, Recall@k, Precision@k, MAP, and MRR, following standard IR evaluation methodology.

## Project Structure

```
.
├── src/                        # Core retrieval implementations
│   ├── data_loader.py          # BeIR dataset loader and preprocessor
│   ├── llamaindex_bm25.py      # BM25 sparse retrieval (LlamaIndex)
│   ├── llamaindex_rag.py       # Dense retrieval with pgvector
│   └── llamaindex_hybrid.py    # Hybrid retrieval with RRF
├── evaluation/
│   └── metrics.py              # NDCG, Recall, Precision, MAP, MRR
├── notebooks/
│   └── experiment_llamaindex.py # Experiment runner and comparison
├── scripts/
│   ├── analyze_dataset.py      # Dataset statistics and analysis
│   └── md_to_latex.py          # Markdown to LaTeX converter
├── tests/                      # Unit and property-based tests
├── docs/                       # Documentation (LaTeX format)
├── data/                       # BeIR datasets (not tracked)
├── models/                     # Cached embedding models (not tracked)
├── docker-compose.yml          # PostgreSQL + pgvector setup
├── init.sql                    # Database schema and functions
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for PostgreSQL + pgvector)
- An OpenAI API key (for RAG generation tasks)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/nghidanh2005/beir-retrieval-benchmark.git
cd beir-retrieval-benchmark
```

### 2. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### 4. Start PostgreSQL with pgvector

```bash
docker-compose up -d
```

### 5. Run the experiment

```bash
python notebooks/experiment_llamaindex.py
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| NDCG@k | Normalized Discounted Cumulative Gain at rank k |
| Recall@k | Fraction of relevant documents retrieved in top-k |
| Precision@k | Fraction of top-k results that are relevant |
| MAP | Mean Average Precision across all queries |
| MRR | Mean Reciprocal Rank of the first relevant result |

## Datasets

This project uses the [BeIR benchmark](https://github.com/beir-cellar/beir) datasets:

| Dataset | Domain | Documents | Description |
|---------|--------|-----------|-------------|
| NFCorpus | Medical / Nutrition | 3,633 | Biomedical information retrieval |
| MS MARCO | General | 8.8M (subset) | Web search queries |
| FiQA | Finance | 57,638 | Financial question answering |
| SciFact | Science | 5,183 | Scientific claim verification |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run property-based tests only
python -m pytest tests/test_md_to_latex_properties.py -v

# Run unit tests only
python -m pytest tests/test_md_to_latex_unit.py -v
```

## Documentation

Detailed documentation is available in LaTeX format under the `docs/` directory:

- `SYSTEM_ARCHITECTURE.tex` -- System design and component overview
- `PIPELINE_DETAILED_FLOW.tex` -- Step-by-step pipeline execution flow
- `BM25_SPARSE_INVERTED_INDEX.tex` -- BM25 algorithm and inverted index explanation
- `DATASET_DOCUMENTATION.tex` -- Dataset details and preprocessing
- `POSTGRESQL_GUIDE.tex` -- PostgreSQL + pgvector setup and usage
- `USAGE.tex` -- Usage guide and configuration reference

## Tech Stack

- **Framework**: [LlamaIndex](https://www.llamaindex.ai/) for retrieval orchestration
- **Database**: PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) for vector storage
- **Embeddings**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Benchmark**: [BeIR](https://github.com/beir-cellar/beir) for standardized IR evaluation
- **Testing**: pytest + [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
