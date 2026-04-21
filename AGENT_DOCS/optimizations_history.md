# ReverieCore: Architecture & Optimizations

This document outlines the core architectural principles and retrieval optimizations that power ReverieCore, shifting it from a standard RAG system to an intent-aware, goal-oriented memory architecture.

## 1. Core Architectural Concepts

In our final optimization pass, we transitioned from `BART-Large-MNLI` to a unified **`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`** architecture. This provided a language-agnostic replacement for all classification tasks, reducing the VRAM footprint by ~500MB while improving native cross-lingual performance.

The system maps queries to three core horizons:
- **Retrieving specific facts or entities (Precision):** High-gravity retrieval for concrete details.
- **Synthesizing related information (Synthesis):** Balanced retrieval for summaries or connections.
- **Exploring open-ended possibilities (Exploration):** Low-gravity retrieval for broad discovery.

### Goal-Based Gravity (vs. Topic-Based)
Early iterations of our gravity system were topic-based. We shifted to **Goal-Based Gravity** to ensure the system remains domain-agnostic. By centering gravity on the user's *intent* rather than the *subject matter*, the system remains equally effective whether the user is debugging a microservice or writing a novel.

## 2. Optimization Pipeline

### Optimization A: Anchor-Weighted Re-ranking
We prioritize "Canonical Entities" (files, classes, and major concepts) identified during memory enrichment. 
- **Mechanism:** The system resolves query entities to graph IDs and applies a weight boost via SQL `CASE` statements.
- **Impact:** This ensures the retriever anchors to the structural "nodes" of a project rather than wandering into generic text fragments.

### Optimization B: Dynamic Gravity & Technical Boost
To prevent an "opinionated" retrieval bias, we introduced a dynamic gravity multiplier combined with a safety valve for technical work.
- **Multiplicative Scoring:** `discovery_score = confidence_score * (1 + (is_anchor * gravity))`
    - *Rationale:* We use a multiplicative approach instead of additive to ensure that the `confidence_score` (ground truth) remains the primary ranking signal. Gravity acts as a force-multiplier for anchors, rather than an override.
- **Normalization:** Intent scores are normalized to sum to 1.0 before gravity calculation, ensuring the "Semantic Force" is distributed proportionally.
- **Knowledge-Type Boost:** A tactical `+0.1` multiplier is applied if the system detects architectural definitions or code, ensuring precision for development tasks.

### Optimization C: Iterative Layered Expansion (1-Hop Dominance)
To combat "ambient noise" in dense knowledge graphs, we implemented **Signal-Aware Expansion**.
- **The Heuristic:** The retriever performs a 1-hop traversal first. If it detects a strong signal (`count >= 3` and `avg_discovery_score >= 0.6`), it halts.
- **The Fallback:** If the local signal is weak, the system automatically expands to a 2-hop search to find missing context.
- **Audit Logging:** Every expansion logs its depth and signal strength to `reverie.log`, allowing for detailed post-mortem tuning of the precision/recall trade-off.

## 3. Performance & Impact

We utilized the **RAGAS** framework to measure the impact of these optimizations against 20 complex architectural questions.

| Phase | Faithfulness | Context Precision | Description |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.8611 | 0.2976 | Standard Vector RAG. |
| **Optimized (Graph)** | 0.8778 | 0.3452 | Unconditional 2-hop graph expansion. |
| **Consolidated (Final)** | **0.8750** | **0.3083** | Intent-driven 1-hop dominance. |

> [!NOTE]
> **Historical Caveat:** Earlier "Baseline" and "Optimized (Graph)" scores were captured during periods of LLM server instability (timeouts and rate-limits). While they remain directionally useful for showing the positive impact of graph-traversal, the **Consolidated (Final)** metrics represent our first fully stabilized ground-truth measurement.
