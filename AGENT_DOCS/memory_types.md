### 🧠 Memory Type Taxonomy for `memory_type` Field

Defining memory types is crucial because it tells the agent *how to prioritize* and *how to use* the retrieved memory. A memory of a `CONVERSATION` is used differently than a `RUNTIME_ERROR`.

This table documents the different classification types for entries in the `memories` table. The agent's logic will use this tag to determine the memory's function (e.g., a `RUNTIME_ERROR` should trigger a diagnostic workflow, while a `USER_PREFERENCE` should trigger a preference check).

| Memory Type | Definition | Purpose / Agent Behavior | Example Content |
| :--- | :--- | :--- | :--- |
| **`CONVERSATION`** | Raw transcript or summary of a user/agent interaction. | **Knowledge Retrieval:** Provides context for past discussions. The primary source of general knowledge. | "User asked about the new database setup on April 18th, 2026." |
| **`TASK`** | A specific goal, sub-goal, or completed action from a project. | **Project Management:** Allows the agent to check its past workflow and track dependencies. | "Goal: Refactor the user authentication module." |
| **`OBSERVATION`** | A neutral, factual finding about the environment or system state. | **Environment Awareness:** Stores facts that are true about the world (e.g., "The server is running on AWS region us-east-1."). | "Found three unused environment variables in `config.yaml`." |
| **`USER_PREFERENCE`** | Explicit, stable preferences given by the user (e.g., naming conventions, preferred tools, preferred color scheme). | **Personalization:** Direct input for the agent to apply to future decisions, ensuring consistency with the user. | "Tom prefers using the `sqlite-vec` embedded database." |
| **`RUNTIME_ERROR`** | A record of an exception, crash, or test failure during execution. | **Debugging/Learning:** Triggers the `systematic_debugging` workflow. This memory is highly valuable for iterative improvement. | "Error: ModuleNotFoundError: No module named 'sentence_transformers'." |
| **`CODE_SNIPPET`** | A piece of code that is deemed functional, efficient, or representative of a solution. | **Reusable Knowledge:** Allows the agent to retrieve and reuse best practices or specific implementation patterns. | "Snippet for quick JSON validation using a regex helper." |
| **`LEARNING_EVENT`** | A conceptual breakthrough or a new piece of domain knowledge acquired (e.g., "Learned that SQLite supports vector search via extension"). | **Skill/Knowledge Acquisition:** Used to update internal knowledge bases or trigger skill refinement. | "Discovered that `sqlite-vec` is lightweight enough for embedded use." |
| **`EXPIRED_TASK`** | A task that was set but never completed within a specified time frame. | **Proactiveness:** Triggers a follow-up reminder or a re-triage of the task to prevent it from being forgotten. | "Task: Review API docs. Expiration Date: 2026-05-01." |