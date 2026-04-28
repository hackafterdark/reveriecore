# ReverieCore Configuration Guide

This document explains the configuration options available in `.reveriecore.yaml`. The configuration is organized into three primary functional layers: **System**, **Retrieval**, and **Enrichment**.

---

## 1. System Settings (`system`)
Global settings for the ReverieCore plugin.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `user_identity` | `string` | `"USER"` | The primary user's name/ID, used for provenance tracking. |
| `memory_char_limit` | `int` | `32768` | The absolute maximum character limit for memory injection into the prompt. |

---

## 2. Retrieval Configuration (`retrieval`)
Controls how memories are discovered, ranked, and budgeted during a search.

### Discovery (`retrieval.discovery`)
Options for the "Discovery" phase where potential memory candidates are found.

- **`default_limit`**: Default number of memories to return if not specified.
- **`anchoring`**:
  - `clean_slate_keywords`: List of keywords that trigger a "Clean Slate" retrieval (clearing recent context).
- **`vector`**:
  - `precision_gate`: The cosine similarity threshold (0.0 to 1.0) for vector search.
  - `candidate_multiplier`: Multiplier for the initial candidate pool (e.g., if limit is 5 and multiplier is 3, fetch 15 candidates).
  - `fallback_threshold`: Minimum number of candidates required before attempting higher-recall fallback strategies.
- **`graph_expansion`**:
  - `seed_limit`: Number of top vector matches used as "seeds" for graph traversal.
  - `min_signal`: Minimum weight threshold for graph edges to be traversed.
  - `discovery_boost`: Score boost applied to memories found via graph traversal.

### Query Rewriter (`retrieval.rewriter`)
Configures the LLM-based query expansion and rewriting layer.

- **`enabled`**: Toggle the rewriter on/off.
- **`model_path`**: Path to the GGUF model file (e.g., `models/Phi-3-mini-4k-instruct-q4.gguf`).

> [!IMPORTANT]
> **Manual Model Download Required**:
> To use the query rewriter, you must download the LLM model manually. If you have the `huggingface-cli` installed, run:
> ```bash
> hf download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir models
> ```

- **`device`**: Execution device (`cpu` or `gpu`).
- **`threads`**: Number of CPU threads for inference.
- **`max_words`**: Maximum word count for the rewritten query.

### Ranking (`retrieval.ranking`)
Controls how discovered candidates are scored and sorted.

- **`intent`**:
  - `fact_markers`: Keywords (e.g., "what", "how") that suggest a fact-seeking intent.
  - `weights`: Strategy-specific weights for `similarity`, `importance`, and temporal `decay`.
    - `fact_seeking`: Optimized for accuracy.
    - `exploration`: Optimized for breadth/relevance.
- **`scoring`**:
  - `anchor_boost`: Boost factor for high-level "Observation Anchor" memories.
  - `graph_boost_multiplier`: Scaling factor for graph signal in the final score.
  - `default_similarities`: Baseline similarity scores for different memory classes.
- **`decay`**:
  - `half_life_hours`: Hours after which a memory's temporal score is halved.
  - `min_decay`: The lowest possible decay multiplier (floor).

### Budgeting (`retrieval.budget`)
Controls the final selection and formatting for prompt injection.

- **`relevance_floor`**: Minimum final score (0.0 to 1.0) required for a memory to be included in the results.
- **`default_token_budget`**: Default max tokens allowed for the retrieval block.
- **`labels`**: Score multipliers or cutoffs for importance labels (e.g., `critical`).

### Pipeline (`retrieval.pipeline`)
Determines the order and selection of active retrieval handlers.
- **`discovery`**: List of active discovery stages (e.g., `["anchoring", "vector"]`).
- **`ranking`**: List of active ranking stages (e.g., `["intent", "scoring"]`).
- **`budget`**: List of active budgeting stages.

---

## 3. Enrichment Configuration (`enrichment`)
Controls the ingestion pipeline: how new interactions are processed, embedded, and summarized.

### Models (`enrichment.models`)
- **`embedding`**: SentenceTransformer model for vector generation (e.g., `all-MiniLM-L6-v2`).
- **`summarization`**: Model for generating memory abstracts (e.g., `distilbart`).
- **`classifier`**: Model for classifying memory types and intent.

### Scoring (`enrichment.scoring`)
- **`heuristics`**:
  - `importance_boost`: Base boost for heuristic matches.
  - `keywords`: Keyword lists for categories like `error`, `urgency`, `security`, and `code`.
- **`weights`**: Score values assigned to importance tiers (`critical`, `important`, etc.).

### Profiling (`enrichment.profiling`)
- **`min_word_count`**: Minimum word count to trigger summarization.
- **`max_summary_length`**: Maximum token length for summaries.
- **`summary_beams`**: Search beams for summarization generation.
- **`retention`**:
  - `low_importance_threshold`: Threshold for pruning low-value memories.
  - `default_days`: Days to keep low-importance memories before archiving.

### Pipeline (`enrichment.pipeline`)
- **`active_stages`**: Enabled ingestion stages (e.g., `["heuristics", "classifier", "model_importance"]`).
