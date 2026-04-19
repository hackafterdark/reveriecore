# Provenance, Identity, and Scoping

ReverieCore implements a professional-grade identity model that separates **Provenance** (who created the data) from **Namespace** (where the data belongs) and **Auditability** (how and when it was created).

## Overview

Unlike basic memory systems that use a single "owner" field, ReverieCore uses a tiered identity system to support multi-agent environments, shared workspaces, and robust auditing.

| Field | Meaning | Scope | Source (Hermes) |
| :--- | :--- | :--- | :--- |
| **`author_id`** | **The Human**. Who is the ultimate creator? | Provenance | `user_id` |
| **`owner_id`** | **The Profile**. Which "Identity" owns this memory? | Namespace | `agent_identity` |
| **`actor_id`** | **The Service**. What process performed the write? | Actorship | `REVERIE_SYNC_SERVICE` |
| **`session_id`** | **The Context**. Which specific conversation? | Audit | `session_uuid` |
| **`workspace`**| **The Location**. What was the working directory? | Audit | `terminal.cwd` |

---

## 1. Namespace Isolation (`owner_id`)

The `owner_id` acts as a **hard boundary** for memory retrieval. It is mapped to the Hermes **Profile Name**.

* **Sandboxing**: An agent running under the `coder` profile will only retrieve memories where `owner_id = 'coder'`. It cannot "see" memories from a `personal` profile.
* **Global Access**: Memories can be made visible across profiles by setting `privacy = 'PUBLIC'` or associating them with the global `PERSONAL_WORKSPACE` owner.

## 2. Provenance tracking (`author_id` & `actor_id`)

This separation allows for complex multi-agent analysis:
* If a sub-agent learns something, the `author_id` remains the human user, but the `actor_id` identifies the specific agent or service (`REVERIE_SYNC_SERVICE`) that handled the storage.
* This prevents "identity bleed" where an agent might mistake a sub-agent's observations for its own primary memory.

## 3. Audit Trail (`session_id` & `workspace`)

These fields ensure every piece of learned information is traceable:
* **Session Tracking**: Use `session_id` to look up the exact conversation logs that led to a memory being stored.
* **Workspace Context**: Use `workspace` to understand the environmental state (project folder) the agent was in when the memory was captured.

## Best Practices for Developers

When extending the `memories` table or adding new tools:
1. **Always filter by `owner_id`**: Ensure your SQL queries include `WHERE owner_id = ? OR privacy = 'PUBLIC'`.
2. **Explicit Actorship**: If a new tool (e.g., a "Maintenance Agent") modifies the database, set the `actor_id` to a descriptive string (e.g., `DATABASE_MAINTENANCE_AGENT`).
