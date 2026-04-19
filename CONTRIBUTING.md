# Contributing to Reverie Core

First off, thank you for considering contributing to ReverieCore! It's people like you that make ReverieCore a great memory engine for the community.

Please take a moment to review this document to understand our development process and how you can get involved.

---

## 🏗️ Architectural Changes (The RFC Process)

If you are proposing a **major or architectural change**, it must go through a Request for Comments (RFC) process before any code is written.

1.  **Open a GitHub Issue**: Create a new issue in the repository.
2.  **Apply the `RFC` Label**: Tag the issue with the `RFC` label.
3.  **Include Motivations**: Your RFC should clearly explain the "why" behind the change, the proposed implementation details, and any potential alternatives considered.
4.  **Community Discussion**: The RFC will be open for discussion by the community and maintainers.

> [!NOTE]
> **Side Project Disclaimer**: ReverieCore is a side project. While your RFCs are very much appreciated, there is no guarantee they will be approved or reviewed in a timely fashion. We ask for your patience!

---

## 🛠️ Contribution Guidelines

### 1. Maintain Agent Documentation
ReverieCore is designed to be understood and used by both humans and AI agents. If your change affects how the system works, **you must update the relevant documentation in the [AGENT_DOCS](./AGENT_DOCS) directory.** This ensures that agents can effectively "understand" and navigate the memory system.

### 2. Pull Request Quality & Behavior
We value quality over quantity. To maintain a healthy repository:
- **No PR Spamming**: Do not submit multiple small, trivial, or unrelated PRs.
- **Explain Your Changes**: Every Pull Request must include a clear explanation of what was changed and why.
- **AI Agent Usage**: While "vibe coding" and using AI assistants to help write code is completely fine, please do not flood the repository with automated PRs from AI agents. Use AI as a tool, not a machine for PR volume.
- **Human Review**: All changes—regardless of source—must be reviewed and approved by a human maintainer.

### 3. Testing
Test cases should accompany code changes where feasible and appropriate. If you are adding a new feature or fixing a bug, please include a corresponding test in the `tests/` directory.

---

## 🚀 How to Submit a Contribution

1.  **Fork the repository** and create your branch from `main`.
2.  **Install dependencies**: `pip install -r requirements.txt`.
3.  **Make your changes**, ensuring you follow the documentation and testing guidelines above.
4.  **Commit your changes** with a descriptive commit message.
5.  **Submit a Pull Request** to the `main` branch.

By contributing, you agree that your contributions will be licensed under the project's [MIT License](./LICENSE).

---

## 🏗️ Project Structure

For developers looking to dive into the code, here is an overview of the key components:

- **`database.py`**: Handles all SQLite database operations, including the initialization of the `vec0` virtual tables and standard relational schemas.
- **`enrichment.py`**: The "Intelligence Layer." Contains the logic for zero-shot classification, importance scoring, and semantic profiling using local transformer models.
- **`retrieval.py`**: Implementation of the hybrid search algorithm (Vector Similarity + Importance Re-ranking).
- **`provider.py`**: The entry point for the Hermes plugin system. Implements the `MemoryProvider` interface.
- **`schemas.py`**: Central repository for Pydantic models, Enums (Memory Types, Association Types), and constants.
- **`AGENT_DOCS/`**: Crucial technical documentation intended for both human developers and AI agents to understand the system's inner workings.
- **`tests/`**: Contains unit and integration tests. Please ensure any new logic is covered here.

## 🛠️ Developer Prerequisites

To set up a development environment for ReverieCore, you will need:

1.  **Python 3.8+**: The core logic is built with modern Python type hinting.
2.  **SQLite with Vector Support**: We use the `sqlite-vec` extension. On most systems, the `sqlite-vec` Python package will handle the binary loading, but ensure your system's `sqlite3` supports loadable extensions.
3.  **Machine Learning Dependencies**:
    - `transformers`: For BART models.
    - `sentence-transformers`: For vector embeddings.
    - `torch`: Backend for the model inference.
4.  **Local RAM**: We recommend at least 8GB of RAM to comfortably load the BART-Large model for enrichment tasks during testing.
