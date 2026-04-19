To test the `session_id`, `workspace`, `owner_id`, `author_id`, `actor_id` fields and verify `MemoryType` classifier is working across the full spectrum, here's some prompts that force the agent to categorize data differently.

Copy and paste these into the Hermes CLI. They are designed to trigger specific logic paths in your `EnrichmentService`.

### 1. The "Code Architecture" Test (Trigger: `CODE_SNIPPET`)
> "Hey, help me finalize the architectural boundary for the memory provider. I'm thinking of using a trait-based approach in Golang for the retrieval interface. Something like this:
> `type MemoryRetriever interface { Search(ctx context.Context, query string) ([]Memory, error) }`
> Let's store this interface in the project documentation for the `reveriecore` workspace."

### 2. The "Operational Frustration" Test (Trigger: `RUNTIME_ERROR` + High Sentiment Score)
> "I am getting a persistent `DatabaseLocked` error when running the sync process. It feels like the `ReverieMemoryProvider` is holding an open transaction during the enrichment phase. This is extremely annoying and it's blocking my progress—please remember that we need to investigate the connection pooling ASAP."

### 3. The "Productivity Goal" Test (Trigger: `TASK`)
> "Update my roadmap: My main task is to implement the Multi-Dimensional Retriever. Sub-task: Create the `session_id` filter. Once that is done, I need to migrate the existing `reveriecore.db` to the new schema."

### 4. The "Personal Workflow" Test (Trigger: `USER_PREFERENCE`)
> "Remember this for all future agent configurations: I prefer a strict `SessionID` chaining policy. Never let a memory get created without an explicit link to the current session. Also, always favor explicit logging over silent background processing."

### 5. The "Technical Discovery" Test (Trigger: `LEARNING_EVENT`)
> "I just learned something interesting about the `sqlite-vec` extension—it handles local memory-mapped files much better if I set the `mmap` size explicitly on the connection string. That’s a key insight for the `reveriecore` performance optimization."

### 6. The "System Context" Test (Trigger: `OBSERVATION`)
> "Observation: The current `REVERIE_SYNC_SERVICE` is performing about 20% faster after the lazy-loading refactor. Status: All memory chunks are successfully indexed in the new `vec_chunks` table for the `personal_workspace` identity."
