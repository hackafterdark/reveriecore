"""ReverieCore Memory Plugin — Local RAG MemoryProvider.

Features locally-hosted embeddings (SentenceTransformers), SQLite-vec for 
vector search, and sentiment-based importance scoring.
"""

from __future__ import annotations

import os
# Must be set BEFORE other imports to ensure libraries pick these up immediately
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import logging
import threading
import time
import atexit
from typing import Any, Dict, List, Optional
from pathlib import Path

import warnings
import sys
import logging.handlers

logger = logging.getLogger(__name__)

# 1. Immediate suppression for the specific pkg_resources warning
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# 2. Prevent the import of the offending library from firing the warning again
# by "faking" the module registration if it hasn't happened yet
sys.modules['pkg_resources'] = type('pkg_resources', (object,), {
    'declare_namespace': lambda name: None
})

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

# Local imports
from .database import DatabaseManager
from .enrichment import EnrichmentService
from .retrieval import Retriever
from .schemas import MemoryType

logger = logging.getLogger(__name__)

class ReverieMemoryProvider(MemoryProvider):
    """Local RAG memory with SQLite-vec and intelligence-based enrichment."""

    def __init__(self):
        super().__init__()
        self._db = None
        self._enrichment = None
        self._retriever = None
        
        # Identity Context
        self.session_id = ""
        self.author_id = "USER"
        self.owner_id = "PERSONAL_WORKSPACE"
        self.actor_id = "REVERIE_SYNC_SERVICE"
        self.workspace = "UNKNOWN_WORKSPACE"
        self.agent_context = "primary"
        
        # Budget Management
        # Lower defaults to be safer for local users (~32k context)
        # We assume 100% available until Hermes sends a signal via on_turn_start
        self.total_context = 32768
        self.remaining_tokens = 32768
        self.memory_char_limit = 2200 # Baseline from config (approx 550 tokens)
        self.model_name = "unknown"
        
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        self._is_shutdown = False

    @property
    def name(self) -> str:
        return "reveriecore"

    def is_available(self) -> bool:
        # Locally hosted, so it's always available once initialized
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize models and database."""
        self._setup_logging()
        
        from hermes_constants import get_hermes_home
        
        # Database is pinned to the preferred .hermes directory
        db_path = get_hermes_home() / "reveriecore.db"
        
        # Identity Context (Provenance via Author/Actor, Scoping via Owner)
        self.session_id = session_id
        self.author_id = kwargs.get("user_id") or kwargs.get("user_identity") or "USER"
        self.owner_id = kwargs.get("agent_identity") or "PERSONAL_WORKSPACE"
        self.actor_id = "REVERIE_SYNC_SERVICE"
        self.workspace = kwargs.get("agent_workspace") or "UNKNOWN_WORKSPACE"
        self.agent_context = kwargs.get("agent_context") or "primary"
        
        try:
            self._db = DatabaseManager(str(db_path))
            self._enrichment = EnrichmentService()
            # Restore knowledge graph anchoring by passing enrichment to Retriever
            self._retriever = Retriever(self._db, enrichment=self._enrichment)
            
            # Capture memory_char_limit if passed from config
            if "memory_char_limit" in kwargs:
                self.memory_char_limit = int(kwargs["memory_char_limit"])
            
            # Register for automatic cleanup on exit
            atexit.register(self.shutdown)
            
            logger.info(f"ReverieCore initialized for {self.author_id} in {self.owner_id} (Actor: {self.actor_id})")
        except Exception as e:
            logger.error(f"ReverieCore failed to initialize: {e}")

    def system_prompt_block(self) -> str:
        return (
            "# Long-Term Memory (ReverieCore)\n"
            "Active. You have access to a local knowledge base of past interactions.\n"
            "Relevant context is automatically injected during the 'Preparing memory' phase.\n"
            "NOTE: Older or less relevant memories may be returned as abstracts (summaries) to save context window space."
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called by Hermes at the start of each turn with runtime context."""
        self.remaining_tokens = kwargs.get("remaining_tokens") or self.remaining_tokens
        
        model_meta = kwargs.get("model_metadata") or {}
        if model_meta:
            self.total_context = model_meta.get("context_length") or self.total_context
            self.model_name = model_meta.get("model") or self.model_name
            
        logger.debug(f"ReverieCore Turn {turn_number}: Remaining Budget: {self.remaining_tokens}/{self.total_context}")

    def _calculate_budget(self) -> int:
        """
        Determines the safe token budget for memory injection.
        
        Dynamic Zones:
        - Comfort (>50% remaining): Full allocation (up to memory_char_limit/4 tokens)
        - Tight (20-50% remaining): Reduced allocation
        - Danger (<20% remaining): Minimal allocation (Abstracts only)
        """
        # Convert char limit to estimated tokens (rough baseline)
        baseline_tokens = (self.memory_char_limit // 4)
        
        remaining_ratio = self.remaining_tokens / self.total_context if self.total_context > 0 else 0.5
        
        if remaining_ratio > 0.5:
            # Comfort Zone
            target = baseline_tokens
        elif remaining_ratio > 0.2:
            # Tight Zone
            target = int(baseline_tokens * 0.7)
        else:
            # Danger Zone
            target = int(baseline_tokens * 0.4)
            
        # Hard Cap: Never claim more than 1/3 of what is physically left
        # to ensure the current turn has room to complete.
        return max(100, min(target, self.remaining_tokens // 3))

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Returns the result of the last queued prefetch."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)
            
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
            
        if not result:
            return ""
        return f"## Relevant Memories\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Runs a semantic search in the background during the agent's 'thinking' phase."""
        if not self._retriever or not self._enrichment:
            return

        def _run():
            try:
                # 1. Calculate Budget & Strategy
                budget = self._calculate_budget()
                remaining_ratio = self.remaining_tokens / self.total_context if self.total_context > 0 else 0.5
                strategy = "abstract_only" if remaining_ratio < 0.2 else "balanced"
                
                # 2. Embed query
                vec = self._enrichment.generate_embedding(query)
                # 3. Search
                results = self._retriever.search(
                    vec, 
                    query_text=query,
                    limit=3, 
                    token_budget=budget,
                    strategy=strategy,
                    allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"]
                )
                if results:
                    lines = [f"- {r['content_full']}" for r in results]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug(f"ReverieCore prefetch failed: {e}")

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="reverie-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Asynchronously cleans, scores, embeds, and saves the conversation turn."""
        if not self._enrichment or not self._db:
            return

        def _sync():
            try:
                # For turn saving, we focus on the assistant's response or a composite
                full_text = f"User: {user_content}\nAssistant: {assistant_content}"
                
                # 1. Intelligence Layer Analysis
                mem_type = self._enrichment.classify_type(full_text)
                importance = self._enrichment.calculate_importance(full_text)["score"]
                profile = self._enrichment.generate_semantic_profile(full_text)
                vec = self._enrichment.generate_embedding(profile) # Embed the profile for cleaner signal
                
                # Token Counts
                tc_full = self._enrichment.count_tokens(full_text)
                tc_abstract = self._enrichment.count_tokens(profile)
                
                # 2. Relational Store
                # Fixed: Using write_lock transaction for safety
                with self._db.write_lock() as cursor:
                    cursor.execute("""
                        INSERT INTO memories (
                            content_full, content_abstract, 
                            token_count_full, token_count_abstract,
                            memory_type, importance_score, 
                            author_id, owner_id, actor_id, 
                            session_id, workspace
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        full_text, profile, 
                        tc_full, tc_abstract,
                        mem_type.value, importance, 
                        self.author_id, self.owner_id, self.actor_id, 
                        self.session_id, self.workspace
                    ))
                    
                    mem_id = cursor.lastrowid
                    
                    # 3. Vector Store
                    import sqlite_vec
                    cursor.execute("""
                        INSERT INTO memories_vec (rowid, embedding)
                        VALUES (?, ?)
                    """, (mem_id, sqlite_vec.serialize_float32(vec)))
                    
                # 4. Graph Extraction (Triggered if high importance)
                if importance >= 3.0:
                    logger.info(f"Triggering enhanced graph extraction for memory {mem_id}")
                    # This is still inside the _sync thread, which is backgrounded
                    self._enrichment.extract_graph_data(full_text, mem_id, self._db)

                # debug
                actual_path = Path(self._db.db_path).resolve()
                logger.info(f"DEBUG: Successfully committed to database at: {actual_path}")
                
                logger.debug(f"Memory saved: ID {mem_id}, Type {mem_type.value}, Score {importance}")
                
            except Exception as e:
                logger.warning(f"ReverieCore sync failed: {e}")

        # Non-blocking sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="reverie-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # REMOVED: Custom search tool removed to restore native Brain Icon experience
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        
        if self._prefetch_thread or self._sync_thread:
            logger.info("ReverieCore: Synchronizing final memories and cleaning up (this may take a few seconds)...")
            
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)
        if self._db:
            self._db.close()

    def _setup_logging(self):
        """Configures a dedicated log file and silences CLI noise."""
        from hermes_constants import get_hermes_home
        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "reverie.log"

        # 1. Broad package-level logging
        # We try to find the root package name (e.g. 'reveriecore') 
        pkg_name = __name__.split('.')[0]
        parent_logger = logging.getLogger(pkg_name)
        parent_logger.setLevel(logging.INFO)
        parent_logger.propagate = False # Keep plugin logs out of Hermes stdout

        # Clear existing handlers to prevent duplicates on re-init
        for h in parent_logger.handlers[:]:
            parent_logger.removeHandler(h)

        # 2. File Handler
        try:
            handler = logging.handlers.RotatingFileHandler(
                str(log_file), 
                maxBytes=10*1024*1024, # 10MB
                backupCount=5
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            parent_logger.addHandler(handler)
            
            # Special: Explicitly point the module-level logger to this handler
            # in case propagation is being blocked upstream
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        except Exception as e:
            # Fallback to stderr if file logging fails
            print(f"ReverieCore: Failed to setup logging file {log_file}: {e}", file=sys.stderr)

        # 3. Silence Noisy Dependencies
        noisy_libs = [
            "transformers", "huggingface_hub", "urllib3", "torch", 
            "sentence_transformers", "filelock"
        ]
        for lib in noisy_libs:
            l = logging.getLogger(lib)
            l.setLevel(logging.ERROR)
            l.propagate = False 
            
        # Global environment overrides
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except ImportError:
            pass

        logger.info(f"ReverieCore Logging initialized. Package: {pkg_name}, File: {log_file}")
