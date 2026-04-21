# ADR: Unified Cognitive Intent Architecture

## Status
Accepted (2026-04-20)

## Context
ReverieCore initially utilized `facebook/bart-large-mnli` for sentiment and importance classification. While functional, this model lacked multi-lingual NLI support and had a performance ceiling that hindered cross-lingual reasoning. 

A proposal was made to transition to **`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`** to solve these issues. While an initial draft of the implementation plan suggested a "dual-model" approach, it was refined to treat mDeBERTa as a direct, drop-in replacement to maintain a lean VRAM footprint.

## Decision
We will utilize a three-tier model architecture to balance local performance with deep semantic reasoning:

### 1. Local NLI/Classification (mDeBERTa-v3)
Everything related to **Cognitive Intent**, **Sentiment**, and **Importance Scoring** is handled locally by `mDeBERTa-v3-base-mnli-xnli`. This model provides native multi-lingual support via its XNLI training, allowing the plugin to handle non-English queries without translation overhead.

### 2. Local Embeddings (Sentence-Transformers)
Vector search relies on **`all-MiniLM-L6-v2`** for generating dense embeddings. This remains a highly efficient, low-latency local model.

### 3. Delegated Entity Extraction (Primary Hermes LLM)
Complex **Entity Extraction** and **Graph Relationship Discovery** are delegated to the user's primary Hermes LLM (configured in `config.yaml`).
- **Rationale:** Entity extraction requires a larger context window and more complex reasoning than standard classification. By utilizing the model the user has already loaded for their agent (e.g., Gemma 2, Llama 3), we avoid loading a specialized extraction model that would consume additional VRAM. 
- **Implementation:** Performed via the `internal_llm_client` sidecar, ensuring that the plugin stays lean while benefiting from "state-of-the-art" reasoning capabilities.

## Consequences

### Positive
- **Architectural Leaness:** Minimizes VRAM usage by not loading redundant large models for extraction.
- **Language Agnostic:** NLI classification is natively multi-lingual.
- **Resource Harmony:** The plugin's resource usage scales proportionally with the LLM the user has already chosen for their agent.

### Negative
- **Dependency on Hermes LLM**: Graph enrichment quality is tied to the strength of the user's primary model (though it falls back gracefully if the model is too small to perform extraction).
