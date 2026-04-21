# ReverieCore State Assessment

Here is the assessment of the current state of the ReverieCore codebase based on the files provided and the `AGENT_DOCS` specifications.

## 1. Safety Check: Data Truncation (`memory_char_limit`)
**Result:** **Your data is safe. The plugin is NOT artificially truncating data on write.**

In `provider.py`, the variable `self.memory_char_limit = 2200` (which you correctly bumped to 32768 in the Hermes config) is **only** used for the retrieval budget:
```python
# provider.py (Line 150)
baseline_tokens = (self.memory_char_limit // 4)
```
This value calculates the `token_budget` passed to `retrieval.py` to ensure we don't stuff too much context into Hermes' prompt. 

During **storage** (`sync_turn`), the plugin takes the full `user_content` and `assistant_content` from Hermes and saves it directly into the SQLite `content_full` column without applying any length bounds. 
If data was being truncated previously, it was indeed happening upstream in Hermes' core architecture before it reached ReverieCore. Your fix in `config.yaml` was the correct move. 

*Recommendation:* Moving to a true `memory_token_budget` explicitly would be cleaner than dividing the character limit by 4, as you noted.

---

## 2. What Hasn't Been Implemented Yet (Orphaned Features)

Comparing the codebase to the `AGENT_DOCS` architectural plans reveals that while almost all code logic has been written, we have a major disconnected module:

### 🔴 The Pruning Service is Orphaned
The logic for memory consolidation and garbage collection exists inside `pruning.py` (`MemoryPruningService.run_maintenance()`). However, **it is never actually executed**. 
- In `provider.py`, there is no scheduled background thread, nor is it called during `shutdown` or `initialize`. 
- **Consequence:** Ephemeral memories (like low-importance chat chatter) that were assigned an `expires_at` date will never be deleted, and fragmented memories will never be clustered/synthesized, defeating the consolidation described in your design docs.

### 🟡 Minor Discrepancies vs. Documentation
- **Association Manager vs Graph Query:** `memory_plugin_plan.md` outlines creating an `association_manager.py`. We actually split this logic into `enrichment.py` (graph extraction) and `graph_query.py` (graph retrieval/traversal). This is arguably a better architectural design, but it diverges from the literal plan.
- **Importance Scorer Execution:** The `memory_importance_score.md` suggests a multi-faceted algorithm combining sentiment, keywords, and time decay. What is currently implemented in `enrichment.py:calculate_importance` is a Zero-Shot classification via BART that buckets memories into `critical/important/minor/trivial` to assign a hard 1.0-5.0 score. Time decay happens dynamically during *retrieval* instead. Again, this works well, but diverges slightly from the documentation.

## 3. Current Working State (What IS Implemented)

- ✅ **Intelligence Layer:** BART is successfully classifying memory types and importance (`enrichment.py`).
- ✅ **Graph Extraction:** The two-pass entity and triple (relationship) extraction is robustly implemented with idempotency safeguards (`enrichment.py`).
- ✅ **Hybrid RAG:** `retrieval.py` flawlessly implements the pipeline described in `how_memory_retrieval_works.md`, checking vector similarity, falling back to graph traversal if needed, and employing the "Comfort/Tight/Danger" budget zones based on Hermes' remaining token signals.
- ✅ **Asynchronous Operations:** Saving memories and pre-fetching occur safely on background threads (`_sync_thread`, `_prefetch_thread`) locking correctly via context limits.

## Summary
You are in a very strong working state. The core database, schema, extraction, and retrieval logic are fully operational and structurally sound. 

**Next Steps (When ready for changes):**
1. Wire up `pruning.py` into a background worker or chron-job inside `provider.py`.
2. Refactor `memory_char_limit` to explicitly use token counts for cleaner context limits.
