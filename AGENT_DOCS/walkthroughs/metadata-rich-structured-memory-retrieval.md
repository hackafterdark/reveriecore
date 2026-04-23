# Implementation Plan: Metadata-Rich & Structured Memory Retrieval

This plan details the shift to a structured Markdown context format, a 0-10 importance scale, and the inclusion of categorical metadata for the agent's inference.

## User Review Required

> [!IMPORTANT]
> **Importance Scale Shift**: I am moving the internal importance scoring from a 1-5 scale to a 0-10 scale. This requires updating both the `EnrichmentService` (production of score) and the `Retriever` (consumption for ranking/decay).

> [!NOTE]
> **Metadata Mapping**: The "Location" field in the header will be dynamically pulled from the `metadata` JSON field. If a location is not present, the field will be omitted or marked as "N/A".

## Proposed Changes

### 1. Intelligence & Database Layer

#### [MODIFY] [enrichment.py](file:///home/tom/.hermes/plugins/reveriecore/enrichment.py)
- Update `calculate_importance` to return scores on a **0.0 - 10.0** scale.
- Ensure the zero-shot classification weights are adjusted to span this wider range.

#### [MODIFY] [database.py](file:///home/tom/.hermes/plugins/reveriecore/database.py)
- Ensure `get_memory` and other fetchers consistently return the `guid` and `metadata` fields.

### 2. Retrieval & Context Injection

#### [MODIFY] [retrieval.py](file:///home/tom/.hermes/plugins/reveriecore/retrieval.py)
- **Scale Update**: Adjust `_calculate_decay` and ranking normalization to use the 10.0 max (instead of 5.0).
- **Label Mapping**: Add a helper to map 0-10 scores to:
  - `0-3`: **Incidental**
  - `4-7`: **Relevant**
  - `8-10`: **Critical**
- **Structured Formatting**: Update the `search` method to build the following block for each memory:
  ```markdown
  ### MEMORY ID: <guid>
  - Timestamp: <learned_at>
  - Category: <memory_type>
  - Importance: <categorical_label>
  - Location: <metadata.location>
  - Context: 
    <content>
  ```

### 3. Provider & Environment Capture

#### [MODIFY] [provider.py](file:///home/tom/.hermes/plugins/reveriecore/provider.py)
- Update `sync_turn` to handle incoming environmental metadata (geolocation, etc.) and store it in the `metadata` JSON field.

---

## Verification Plan

### Automated Tests
- **Unit Test**: `test_importance_mapping.py` [NEW] to verify that specific scores (e.g., 2.5, 5.0, 9.0) result in the correct labels (**Incidental**, **Relevant**, **Critical**).
- **Format Test**: Verify that the `search` output exactly matches the requested Markdown structure.

### Manual Verification
- Add a memory with `metadata={"location": "San Francisco"}` and verify it renders correctly in the context window.

---

## Verification Plan

### Automated Tests
- **Unit Test**: Verify that `retriever.search` returns a string containing the expected metadata tags.
- **Integration Test**: Verify that passing a `metadata` dict to `sync_turn` results in that data being stored in the DB and retrieved in subsequent turns.

### Manual Verification
- Use the `memory` tool to `add` a memory with specific metadata and verify it appears in the "Relevant Memories" block of the next turn.
