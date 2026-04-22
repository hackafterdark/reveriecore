# Agent Rules - Reverie Core Memory Plugin

This document outlines the critical rules and best practices for developing and maintaining the Reverie Core memory plugin. Adherence to these rules ensures stability, discoverability, and data integrity.

## 1. Plugin Loading & Discoverability
- **Verification**: The plugin's status can be verified using:
  hermes memory status

- **Error Detection**: If the status says Plugin: NOT installed ✗, the plugin has failed to load. This typically indicates a syntax error or an import resolution issue.
- **Relative Imports**: Use package-relative imports (e.g., from .database import DatabaseManager) within the core plugin modules. This ensures the plugin resolves correctly when loaded by Hermes.

## 2. Integrity & Testing
- **Test Mandate**: No new feature or logic change is complete without corresponding test cases in the tests/ directory.
- **Verification Suite**: Use the following command to run the full test suite in the Hermes venv:
  /home/tom/.hermes/hermes-agent/venv/bin/pytest tests/

- **Regression Testing**: Maintain tests/test_plugin_loader.py to ensure that packaging changes don't break Hermes runtime compatibility.
- **Benchmarks**: For performance-critical changes (retrieval, RAG), update and run tests/benchmark_ragas.py.

## 3. Database & Concurrency
- **Concurrency Safety**: Always use DatabaseManager.write_lock() for write operations to respect SQLite thread-safety and the 10s busy_timeout.
- **Identity Integrity**: All memories and entities must have a stable guid. Never rely purely on auto-incrementing id for sync or archival logic.

## 4. Memory-as-Code (Mirroring)
- **Archive Fidelity**: The archive/ (Mirror) is the source of truth for long-term persistence. Ensure that all archival processes in MesaService trigger MirrorService.export_node().
- **Bi-directional Sync**: Maintain the ability to import from Markdown archives without losing graph associations.

## 5. Persistence & Testing Guidelines

### Avoid "Silent" Failures
Never swallow exceptions in persistence logic. Broad try/except blocks in provider.py or service layers hide underlying database failures and cause silent rollbacks.
* **DO:** Allow sqlite3.OperationalError or other database exceptions to bubble up to the caller.
* **DO:** Use cursor.rowcount after UPDATE or DELETE operations to verify that rows were affected. If rowcount == 0, raise an explicit ValueError.

### Transactional Integrity
All modifications (main table + vector index) must be wrapped in self.write_lock().
* **Atomicity:** This ensures if a vector insertion fails, the main table update is rolled back automatically. 
* **Pattern:** Prepare data (inference, synthesis, formatting) before entering the write_lock() block. Minimize time spent inside the transaction to prevent locking issues.

### Test Isolation & Verification
* **Fresh Cursors:** When asserting database state in tests, always fetch a fresh cursor from provider._db.get_cursor() after the mutation.
* **API-Driven Verification:** Favor provider.get_memory(id) over raw SQL queries for verification. This ensures you are testing the system's "source of truth" and avoids stale snapshot issues.
* **Mocking Safety:** Never hardcode paths to the host's home directory (e.g., ~/.hermes). Always use the centralized hermes_environment fixture in tests/conftest.py which provides a scoped, temporary workspace.