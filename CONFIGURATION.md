# ReverieCore Configuration Guide

This document explains the configuration options available in `.reveriecore.yaml`. The configuration is organized into four primary functional layers: **System**, **Retrieval**, **Enrichment**, and **Maintenance**.

---

## 1. System Settings (`system`)
Global settings for the ReverieCore plugin.

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `user_identity` | `string` | `"USER"` | The primary user's name/ID, used for provenance tracking. |
| `memory_char_limit` | `int` | `32768` | The absolute maximum character limit for memory injection into the prompt. |

### Telemetry (`system.telemetry`)
ReverieCore supports OpenTelemetry for performance tracking and pipeline visualization.

- **`enabled`**: Toggle telemetry tracking on/off. Defaults to `true`.
- **`endpoint`**: The OTLP collector URL (e.g., `http://localhost:4318/v1/traces`). 
- **`protocol`**: The OTLP protocol to use. Supports `http/protobuf` and `http/json`.
- **`headers`**: A dictionary of custom headers to send with OTLP requests (e.g., `Authorization: "Bearer <token>"`).
- **`resource_attributes`**: A dictionary of global attributes to attach to all traces (e.g., `environment: "production"`).

> [!NOTE]
> **Graceful Failure**: If the configured telemetry endpoint is unreachable at startup, ReverieCore will log a warning and automatically disable telemetry for the session to prevent performance degradation or log spam.


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
- **`intent_classifier`**:
  - `intent_strategy`: The classification logic for intent detection. Supports `binary` (aggressive) or `trinary` (conservative). Defaults to `binary` for retrieval.
  - `confidence_threshold`: Minimum probability score required before the detected intent is used for edge filtering. Defaults to `0.25`.

### Query Rewriter (`retrieval.rewriter`)
Configures the LLM-based query expansion and rewriting layer. Activated by adding `"rewriter"` to the `retrieval.pipeline`.

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

### Pruning (`retrieval.pruning`)
Filters candidates based on score quality immediately after re-ranking.

- **`top_n`**: Absolute maximum number of candidates to keep.
- **`relative_threshold`**: Contextual quality gate (0.0 to 1.0). Discards candidates that aren't at least X% as good as the top scorer.
- **`min_absolute_score`**: The "hard floor". Discards anything below this score regardless of relative performance.

### Budgeting (`retrieval.budget`)
Controls the final selection and formatting for prompt injection.

- **`relevance_floor`**: Minimum final score (0.0 to 1.0) required for a memory to be included in the results.
- **`default_token_budget`**: Default max tokens allowed for the retrieval block.
- **`labels`**: Score multipliers or cutoffs for importance labels (e.g., `critical`).

### Pipeline (`retrieval.pipeline`)
Determines the order and selection of active retrieval handlers.
- **`discovery`**: List of active discovery stages (e.g., `["anchoring", "vector"]`).
- **`ranking`**: List of active ranking stages (e.g., `["intent", "scoring", "rerank", "pruning"]`).
- **`budget`**: List of active budgeting stages.

---

## 3. Enrichment Configuration (`enrichment`)
Controls the ingestion pipeline: how new interactions are processed, embedded, and summarized.

### Components (enrichment.classifier, etc.)
Structured settings for the individual AI models used during ingestion.

- **`classifier`**:
  - `model`: The zero-shot classification model (e.g., `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`).
  - `intent_strategy`: The logic for handling model output. Use `trinary` (default) for conservative ingestion to avoid false positives, or `binary` for forced-choice classification.
- **`embedding`**:
  - `model`: SentenceTransformer model for vector generation (e.g., `all-MiniLM-L6-v2`).
- **`summarization`**:
  - `model`: Model for generating memory abstracts (e.g., `sshleifer/distilbart-cnn-12-6`).

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

---

## 4. Maintenance Configuration (`maintenance`)
Controls the `MesaService` background maintenance tasks.

### Mesa (`maintenance.mesa`)
- **`pipeline`**: List of active maintenance stages. Supports `["soft_prune", "consolidate", "deep_clean"]`. To disable background maintenance completely, provide an empty list `[]`.
- **`dry_run`**: If true, log actions without executing them (useful for debugging).
- **`interval_seconds`**: Frequency of maintenance cycles.
- **`centrality_threshold`**: Minimum connections a memory must have to avoid being archived.
- **`retention_days`**: Days to keep low-importance memories before archiving.
- **`importance_cutoff`**: Score threshold (0.0 - 10.0) below which memories are considered "stale".
- **`consolidation_threshold`**: Number of memories required to trigger hierarchical consolidation.
- **`purge_enabled`**: Enable Tier 2 deep cleaning (permanent deletion).
- **`deep_clean_interval_days`**: Frequency of deep cleaning and database `VACUUM`.
- **`archive_retention_days`**: Days to keep archived memories before permanent deletion.
- **`pruning`**:
  - `retention_threshold`: Quality threshold for background maintenance.
  - `batch_size`: Number of nodes processed per maintenance batch.
