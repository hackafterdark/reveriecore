# ADR: Unified Cognitive Intent Architecture

## Status
Accepted (2026-04-20)

## Context
ReverieCore's initial intelligence layer utilized multiple models (`facebook/bart-large-mnli` for sentiment and `mDeBERTa` for intent). This dual-model approach introduced significant VRAM overhead (~1.2GB floor) and increased latency during cognitive enrichment. Furthermore, BART lacked native multi-lingual NLI support, requiring language-specific workarounds.

## Decision
We will consolidate all enrichment tasks—Intent Classification, Importance Scoring, and memory categorization—onto a single **`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`** model.

### Key Implementation Details
1. **Zero-Shot Classification:** We utilize the model's XNLI (Cross-lingual Natural Language Inference) capabilities to perform domain-agnostic classification across ANY language.
2. **Standardized Labels:** The system is restricted to goal-oriented labels (*Retrieving specific facts*, *Synthesizing information*, *Exploration*) rather than topic-orientated labels to ensure generalized applicability.
3. **Hardware Efficiency:** By retiring the BART model, we reduce the total VRAM footprint by approximately **500MB**, making the plugin more accessible for local development.

## Consequences

### Positive
- **Architectural Simplicity:** A single model handles the entire "Cognitive Enrichment" pipeline.
- **Resource Optimization:** Significantly lower memory usage on the user's system.
- **Language Agnostic:** Native support for non-English queries without translation overhead.

### Negative
- **Transition Cost:** Required refactoring the `EnrichmentService` to map previous sentiment logic into the NLI entailment format.
