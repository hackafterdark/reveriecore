# ADR 007: Decoupling Configuration via `reveriecore.yaml`

## Status
Accepted

## Context
ReverieCore was originally designed as a tight integration for the Hermes agent ecosystem. Consequently, its configuration was embedded within the Hermes `config.yaml` and passed through as runtime `kwargs` during plugin initialization.

As the project matures into a high-performance, general-purpose RAG framework, several limitations of this approach have emerged:
1.  **Platform Lock-in**: Running ReverieCore as a standalone library or in non-Hermes environments (like CLI tools or different agent frameworks) was difficult without mocking the Hermes configuration structure.
2.  **Configuration Bloat**: The Hermes `config.yaml` was becoming a "junk drawer" for memory-specific tuning parameters (thresholds, pipeline selection, model names), making it harder for users to maintain.
3.  **Extensibility**: The move toward a "Framework" vision (see [ADR 006](006-reverie-framework-pipeline-architecture.md)) requires a structured way to define complex pipelines that doesn't fit well into a shared global configuration.

## Decision
We will move all ReverieCore-specific configuration to a dedicated `reveriecore.yaml` file and implement a prioritized discovery mechanism.

### 1. Dedicated Configuration File
All internal settings, including the Handler Registry mappings, pipeline stages, and scoring thresholds, will be mastered in `reveriecore.yaml`.

### 2. Prioritized Discovery Logic
The system will resolve the configuration path in the following order:
1.  **Hermes Pointer**: If running within Hermes, check `config.yaml` for `memory.reveriecore_cfg`. This allows Hermes to act as a lightweight orchestrator without owning the data.
2.  **Environment Variable**: Check `REVERIECORE_CONFIG` for explicit overrides (useful for CI/CD and containers).
3.  **Global Default**: Fallback to `~/.reveriecore.yaml` for a "zero-config" developer experience.

### 3. Runtime State Decoupling (`EnvironmentalContext`)
While `reveriecore.yaml` handles static configuration, the dynamic state of a "turn" (e.g., current location, remaining tokens) is now encapsulated in an `EnvironmentalContext` object. This ensures that the engine does not rely on global state or platform-specific dictionary structures passed during runtime.

## Consequences

### Positive
- **Portability**: ReverieCore can now be deployed in any Python environment by simply providing a `.yaml` file.
- **Organization**: Clean separation between agent-level settings (identity, session) and engine-level settings (RAG precision, pipeline order).
- **Reduced Friction**: Hermes developers can now tune memory performance without restarting the main agent process or modifying core agent configs.

### Negative
- **Fragmentation**: Users now have to manage two configuration files (`config.yaml` and `reveriecore.yaml`).
- **Initialization Overhead**: Small increase in startup time due to multi-file I/O and YAML parsing.
