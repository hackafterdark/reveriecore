# Agent Rules - ReverieCore Memory Plugin

This document outlines the critical rules and best practices for developing and maintaining the ReverieCore memory plugin. Adherence to these rules ensures stability, discoverability, and data integrity.

## 1. Plugin Loading & Discoverability
- **Immediate Verification**: After any code change, verify the plugin's status using:
  ```bash
  hermes memory status
  ```
- **Error Detection**: If the status says **Plugin: NOT installed ✗**, the plugin has failed to load. This typically indicates a syntax error or an import resolution issue.
- **Relative Imports**: Always use **package-relative imports** (e.g., `from .database import DatabaseManager`) within the core plugin modules. Absolute imports (e.g., `from database import ...`) will fail when the plugin is loaded as a package by Hermes.

## 2. Integrity & Testing
- **Test Mandate**: No new feature or logic change is complete without corresponding test cases in the `tests/` directory.
- **Verification Suite**: Use the following command to run the full test suite in the Hermes venv:
  ```bash
  /home/tom/.hermes/hermes-agent/venv/bin/pytest tests/
  ```
- **Regression Testing**: Maintain `tests/test_plugin_loader.py` to ensure that packaging changes don't break Hermes runtime compatibility.
- **Benchmarks**: For performance-critical changes (retrieval, RAG), update and run `tests/benchmark_ragas.py`.

## 3. Database & Concurrency
- **Singleton Reset (Testing)**: When writing tests, always include a fixture to reset the `DatabaseManager` singleton to ensure isolation:
  ```python
  @pytest.fixture(autouse=True)
  def reset_singleton():
      DatabaseManager._instance = None
      yield
      DatabaseManager._instance = None
  ```
- **Concurrency Safety**: Always use `DatabaseManager.write_lock()` for write operations to respect SQLite thread-safety and the 10s `busy_timeout`.
- **Identity Integrity**: All memories and entities must have a stable `guid`. Never rely purely on auto-incrementing `id` for sync or archival logic.

## 4. Memory-as-Code (Mirroring)
- **Archive Fidelity**: The `archive/` (Mirror) is the source of truth for long-term persistence. Ensure that all archival processes in `MesaService` trigger `MirrorService.export_node()`.
- **Bi-directional Sync**: Maintain the ability to import from Markdown archives without losing graph associations.
