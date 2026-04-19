"""ReverieCore Memory Plugin — Local RAG MemoryProvider.

Features locally-hosted embeddings (SentenceTransformers), SQLite-vec for 
vector search, and sentiment-based importance scoring.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

import warnings
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # silence the plugin's debug/info logs

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

SEARCH_SCHEMA = {
    "name": "reverie_search",
    "description": (
        "Search your long-term memory for relevant facts, past conversations, "
        "or project context. Uses semantic similarity and importance ranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What you want to remember."},
            "limit": {"type": "integer", "description": "Max results (default: 5)."},
        },
        "required": ["query"],
    },
}

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

    @property
    def name(self) -> str:
        return "reveriecore"

    def is_available(self) -> bool:
        # Locally hosted, so it's always available once initialized
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize models and database."""
        logger.warning("!!! REVERIECORE_LOAD_CHECK: VERSION_V3_LAZY !!!")
        from hermes_constants import get_hermes_home
        
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
            self._retriever = Retriever(self._db)
            
            # Capture memory_char_limit if passed from config
            if "memory_char_limit" in kwargs:
                self.memory_char_limit = int(kwargs["memory_char_limit"])
            
            logger.info(f"ReverieCore initialized for {self.author_id} in {self.owner_id} (Actor: {self.actor_id})")
        except Exception as e:
            logger.error(f"ReverieCore failed to initialize: {e}")

    def system_prompt_block(self) -> str:
        return (
            "# Long-Term Memory (ReverieCore)\n"
            "Active. You have access to a local knowledge base of past interactions.\n"
            "Use 'reverie_search' to find specific historical context when needed.\n"
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
                # 1. Calculate Budget & Strategy
                budget = self._calculate_budget()
                remaining_ratio = self.remaining_tokens / self.total_context if self.total_context > 0 else 0.5
                strategy = "abstract_only" if remaining_ratio < 0.2 else "balanced"
                
                # 2. Embed query
                vec = self._enrichment.generate_embedding(query)
                # 3. Search
                results = self._retriever.search(
                    vec, 
                    limit=3, 
                    token_budget=budget,
                    strategy=strategy,
                    allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"]
                )
                if results:
                    lines = [f"- {r['content']}" for r in results]
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
                importance = self._enrichment.calculate_importance(full_text)
                profile = self._enrichment.generate_semantic_profile(full_text)
                vec = self._enrichment.generate_embedding(profile) # Embed the profile for cleaner signal
                
                # Token Counts
                tc_full = self._enrichment.count_tokens(full_text)
                tc_abstract = self._enrichment.count_tokens(profile)
                
                # 2. Relational Store
                cursor = self._db.get_cursor()
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
                
                self._db.commit()
                
                # debug
                actual_path = Path(self._db.db_path).resolve()
                logger.info(f"DEBUG: Successfully committed to database at: {actual_path}")
                
                mem_id = cursor.lastrowid
                logger.info(f"DEBUG: Memory saved with ID: {mem_id}")
                logger.debug(f"Memory saved: ID {mem_id}, Type {mem_type.value}, Score {importance}")
                
            except Exception as e:
                logger.warning(f"ReverieCore sync failed: {e}")

        # Non-blocking sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="reverie-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "reverie_search":
            query = args.get("query", "")
            limit = int(args.get("limit", 5))
            
            if not query:
                return tool_error("Missing parameter: query")
                
            try:
                # Calculate Budget & Strategy
                budget = self._calculate_budget()
                remaining_ratio = self.remaining_tokens / self.total_context if self.total_context > 0 else 0.5
                strategy = "abstract_only" if remaining_ratio < 0.2 else "balanced"
                
                vec = self._enrichment.generate_embedding(query)
                results = self._retriever.search(
                    vec, 
                    limit=limit,
                    token_budget=budget,
                    strategy=strategy,
                    allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"]
                )
                
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                
                return json.dumps({
                    "results": [r['content'] for r in results],
                    "count": len(results),
                    "budget_used": sum(r['tokens'] for r in results),
                    "strategy": strategy
                })
            except Exception as e:
                return tool_error(f"Search failed: {e}")
                
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._db:
            self._db.close()
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)
