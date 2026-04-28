# RAGAS Benchmarking for Reverie Core

This document outlines the RAGAS benchmarking suite for Reverie Core, specifically focusing on the Amnesty International QA dataset.

## Dataset: Amnesty QA
The benchmark uses the [vibrantlabsai/amnesty_qa](https://huggingface.co/datasets/vibrantlabsai/amnesty_qa) dataset (specifically the `english_v3` subset). 

### Source and Purpose
This dataset is a curated set of questions and answers based on Amnesty International reports. It was specifically designed for testing Retrieval-Augmented Generation (RAG) systems. It provides "ground truth" answers and the original contexts from which those answers should be derived.

### Data Format
The dataset is provided in Parquet format. Each sample contains:
- `user_input`: The question to be answered.
- `reference`: The ground truth answer for evaluation.
- `response`: A baseline generated answer (not used in our evaluation, as we generate our own).
- `retrieved_contexts`: A list of relevant context passages retrieved for answering the question.

In our benchmark, we treat the `retrieved_contexts` as the "external knowledge" that Reverie Core needs to "learn" and then retrieve correctly to answer the `user_input`.

## What is RAGAS?
**RAGAS** (Retrieval Augmented Generation Assessment) is a framework that helps evaluate RAG pipelines without requiring human-annotated labels for every single query. It uses a "LLM-as-a-judge" approach to calculate metrics that measure the quality of both the retrieval and generation components.

### Key Metrics Used
1.  **Faithfulness**: Measures if the answer is derived solely from the retrieved context. This ensures the model isn't "hallucinating" facts outside of its memory.
2.  **Context Precision**: Measures the signal-to-noise ratio of the retrieved contexts. It checks if the truly relevant documents are ranked higher than irrelevant ones.

## Why RAGAS for Reverie Core?
Reverie Core acts as a sophisticated memory and knowledge graph system for agents. To ensure its effectiveness, we need to validate:
- **Retrieval Quality**: Is the graph traversal finding the right "memories" (nodes) for a given query?
- **Generation Quality**: Is the agent using those memories correctly to provide accurate, grounded responses?

RAGAS allows us to programmatically track how changes to our indexing logic, graph structure, or reranking algorithms impact these core performance metrics.

## Setup and Configuration

The benchmark scripts are configurable via environment variables. If not set, they default to the Amnesty International dataset and a local Gemma instance.

### LLM Configuration
| Variable | Description | Default |
| :--- | :--- | :--- |
| `REVERIE_LLM_BASE_URL` | The API endpoint for the LLM judge/generator. | `http://172.22.0.1:8080/v1` |
| `REVERIE_LLM_MODEL` | The model name to use. | `gemma4-e4b` |
| `REVERIE_LLM_API_KEY` | API key for the LLM service. | `sk-reverie-internal` |

### Dataset Configuration
| Variable | Description | Default |
| :--- | :--- | :--- |
| `REVERIE_DATASET_PATH` | Hugging Face dataset path. | `vibrantlabsai/amnesty_qa` |
| `REVERIE_DATASET_NAME` | Dataset subset/name. | `english_v3` |
| `REVERIE_DATASET_SPLIT` | Dataset split to use. | `eval` |
| `REVERIE_SESSION_ID` | Session ID for memory ingestion. | `amnesty_benchmark` |

## Running the Benchmark

The benchmark is split into two phases: **Ingestion** and **Evaluation**.

### 1. Ingest the Dataset
First, you must ingest the Amnesty dataset into the Reverie Core database. This populates the memory with the necessary context.

```bash
PYTHONPATH=. python tests/ragas_ingest_dataset.py
```

### 2. Run the Evaluation
Once ingested, run the evaluation script. This will query the system for each question in the dataset, retrieve memories, generate answers, and then use RAGAS to score the results.

```bash
# Optional: Set your own LLM endpoint
export REVERIE_LLM_BASE_URL="http://your-server:8080/v1"
export REVERIE_LLM_MODEL="your-model-name"

PYTHONPATH=. python tests/ragas_evaluate_dataset.py
```

The results will be printed to the console, showing the final scores for Faithfulness and Context Precision.
