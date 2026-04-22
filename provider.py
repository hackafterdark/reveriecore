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

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

# Local imports
from .database import DatabaseManager
from .enrichment import EnrichmentService
from .retrieval import Retriever
from .schemas import MemoryType, AssociationType
from .pruning import MesaService
from .mirror import MirrorService

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
        self._mesa_service = None
        self._mirror_service = None
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
            
            # Initialize Mesa Maintenance Service
            # Pull thresholds from config/kwargs with defaults
            c_thresh = kwargs.get("mesa_centrality_threshold", 2)
            a_days = kwargs.get("mesa_retention_days", 14)
            i_cutoff = kwargs.get("mesa_importance_cutoff", 4.0)
            interval = kwargs.get("mesa_interval_seconds", 3600)
            
            # Initialize MirrorService (Memory-as-Code)
            archive_path = get_hermes_home() / "reverie_archive"
            self._mirror_service = MirrorService(self._db, self._enrichment, archive_root=archive_path)
            self._mirror_service.start()
            
            self._mesa_service = MesaService(
                self._db, 
                self._enrichment,
                mirror=self._mirror_service,
                centrality_threshold=int(c_thresh),
                age_days=int(a_days),
                importance_cutoff=float(i_cutoff),
                interval_seconds=int(interval)
            )
            self._mesa_service.start()
            
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
                    lines = [f"- {r['content']}" for r in results]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug(f"ReverieCore prefetch failed: {e}")

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="reverie-prefetch")
        self._prefetch_thread.start()

    def _save_memory_sync(self, user_content: str, assistant_content: str, session_id: str = "") -> None:
        """Internal synchronous method to process and save memory."""
        try:
            # For turn saving, we focus on the assistant's response or a composite
            full_text = f"User: {user_content}\nAssistant: {assistant_content}"
            
            # 1. Intelligence Layer Analysis
            mem_type = self._enrichment.classify_type(full_text)
            importance_data = self._enrichment.calculate_importance(full_text)
            importance = importance_data["score"]
            profile = self._enrichment.generate_semantic_profile(full_text)
            vec = self._enrichment.generate_embedding(profile) # Embed the profile for cleaner signal
            
            # 2. Canonical Merge Check
            # Search for duplicates with > 0.95 similarity
            duplicates = self._retriever.find_duplicates(
                vec, 
                threshold=0.95, 
                allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"]
            )
            
            if duplicates:
                dup = duplicates[0]
                dup_id = dup["id"]
                logger.info(f"Canonical Merge: Match found (ID: {dup_id}, Similarity: {dup['similarity']:.3f}). Synthesizing...")
                
                # Merge content using LLM synthesis
                merged_text = self._enrichment.synthesize_memories(
                    {dup_id: dup["content_full"], -1: full_text}, 
                    "canonical_knowledge"
                )
                
                # Re-analyze synthesized content
                new_profile = self._enrichment.generate_semantic_profile(merged_text)
                new_vec = self._enrichment.generate_embedding(new_profile)
                new_importance = self._enrichment.calculate_importance(merged_text)["score"]
                
                # New Token Counts
                tc_full = self._enrichment.count_tokens(merged_text)
                tc_abstract = self._enrichment.count_tokens(new_profile)
                
                self._db.update_memory(
                    dup_id, 
                    merged_text, 
                    new_profile, 
                    new_vec, 
                    tc_full, 
                    tc_abstract,
                    importance_score=new_importance
                )







                logger.info(f"Memory {dup_id} Canonicalized and Updated.")
                return 

            # 3. Standard Relational Store (if no merge found)
            tc_full = self._enrichment.count_tokens(full_text)
            tc_abstract = self._enrichment.count_tokens(profile)
            
            with self._db.write_lock() as cursor:
                import uuid
                cursor.execute("""
                    INSERT INTO memories (
                        content_full, content_abstract, 
                        token_count_full, token_count_abstract,
                        memory_type, importance_score, 
                        author_id, owner_id, actor_id, 
                        session_id, workspace, guid
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    full_text, profile, 
                    tc_full, tc_abstract,
                    mem_type.value, importance, 
                    self.author_id, self.owner_id, self.actor_id, 
                    session_id or self.session_id, self.workspace, str(uuid.uuid4())
                ))
                
                mem_id = cursor.lastrowid
                
                # 4. Vector Store
                import sqlite_vec
                cursor.execute("""
                    INSERT INTO memories_vec (rowid, embedding)
                    VALUES (?, ?)
                """, (mem_id, sqlite_vec.serialize_float32(vec)))
                
            # 5. Graph Extraction (Triggered if high importance)
            if importance >= 3.0:
                logger.info(f"Triggering enhanced graph extraction for memory {mem_id}")
                self._enrichment.extract_graph_data(full_text, mem_id, self._db)

            logger.debug(f"Memory saved: ID {mem_id}, Type {mem_type.value}, Score {importance}")
            
        except Exception as e:
            logger.warning(f"ReverieCore sync failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Asynchronously cleans, scores, embeds, and saves the conversation turn."""
        if not self._enrichment or not self._db:
            return

        # Non-blocking sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)

        self._sync_thread = threading.Thread(
            target=self._save_memory_sync, 
            args=(user_content, assistant_content, session_id),
            daemon=True, 
            name="reverie-sync"
        )
        self._sync_thread.start()


    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Provides the memory management tool schema to Hermes."""
        return [
            {
                "name": "memory",
                "description": "Exposes manual control over the ReverieCore memory vault. Use this to delete or replace specific records after finding them via semantic search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "remove", "replace"],
                            "description": "Selected action. Note: 'remove' and 'replace' require a search step first if memory_id is unknown."
                        },
                        "text": {
                            "type": "string",
                            "description": "The memory content (for 'add') or search query (for 'remove'/'replace') to identify potential candidates."
                        },
                        "memory_id": {
                            "type": "integer",
                            "description": "The explicit database ID of the memory to remove or replace (obtained via search)."
                        },
                        "replacement": {
                            "type": "string",
                            "description": "The new content to use when replacing an existing memory."
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "recall_reverie",
                "description": "Drill down into the specific, nuanced experiences (fragments) that form an Observation Anchor. Use this when a retrieved summary indicates that Child IDs are available and you need gritty details to maintain precision.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "The Child ID to recall (as listed in the parent Observation's context)."
                        }
                    },
                    "required": ["memory_id"]
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "recall_reverie":
            return self._handle_recall_reverie(args.get("memory_id"))

        if tool_name != "memory":
            return tool_error(f"Unknown tool: {tool_name}")

        action = args.get("action")
        text = args.get("text", "")
        mem_id = args.get("memory_id")
        replacement = args.get("replacement")

        if action == "add":
            if not text:
                return tool_error("Text content is required for the 'add' action.")
            # Map manual 'add' to sync_turn for standard processing
            self.sync_turn("Manual Entry", text)
            return "Memory has been successfully queued for enrichment and storage."

        if action == "remove":
            if not mem_id:
                # Stage 1: Search for candidates
                if not text:
                    return tool_error("Search text is required to find memories for removal.")
                return self._handle_management_search(text, "remove")
            else:
                # Stage 2: Confirmed removal by ID
                memory = self._db.get_memory(mem_id)
                if not memory:
                    return tool_error(f"Memory ID {mem_id} not found.")
                self._db.delete_memory(mem_id)
                return f"Memory ID {mem_id} ('{memory['content_full'][:50]}...') has been permanently deleted."

        if action == "replace":
            if not mem_id:
                # Stage 1: Search for candidates
                if not text:
                    return tool_error("Search text is required to find memories for replacement.")
                return self._handle_management_search(text, "replace")
            else:
                # Stage 2: Confirmed replacement by ID
                if not replacement:
                    return tool_error("Replacement text is required for the 'replace' action.")
                
                existing = self._db.get_memory(mem_id)
                if not existing:
                    return tool_error(f"Memory ID {mem_id} not found.")

                # Re-enrich the replacement text
                profile = self._enrichment.generate_semantic_profile(replacement)
                vec = self._enrichment.generate_embedding(profile)
                importance = self._enrichment.calculate_importance(replacement)["score"]
                tc_f = self._enrichment.count_tokens(replacement)
                tc_a = self._enrichment.count_tokens(profile)
                
                self._db.update_memory(mem_id, replacement, profile, vec, tc_f, tc_a, importance_score=importance)
                return f"Memory ID {mem_id} has been successfully replaced with the new content."

        return tool_error(f"Unsupported memory action: {action}")

    def _handle_recall_reverie(self, mem_id: Optional[int]) -> str:
        """Handles the 'drill-down' request with strict multi-tenant validation."""
        if not mem_id:
            return tool_error("memory_id is required for recall_reverie.")
            
        try:
            # 1. Fetch metadata and ownership
            memory = self._db.get_memory(mem_id)
            if not memory:
                return tool_error(f"Memory fragment {mem_id} not found.")
                
            # 2. Strict Security Validation
            # Case A: Explicit Ownership
            if memory.get("owner_id") == self.owner_id or memory.get("owner_id") == "PERSONAL_WORKSPACE":
                is_authorized = True
            else:
                # Case B: Provenance check (Is it a child of an authorized Observation?)
                is_authorized = self._db.check_provenance_access(mem_id, self.owner_id)
                
            if not is_authorized:
                logger.warning(f"SECURITY ALERT: Actor {self.actor_id} (Prop: {self.owner_id}) attempted to unauthorized recall of memory {mem_id}")
                return tool_error("Access Denied. You are not authorized to recall this memory fragment.")

            # 3. Return the full content
            content = memory.get("content_full", "")
            m_type = memory.get("memory_type", "FRAGMENT")
            return f"### RECALLED NUANCE (ID: {mem_id}, Type: {m_type})\n\n{content}"

        except Exception as e:
            logger.error(f"recall_reverie failed for {mem_id}: {e}")
            return tool_error(f"Internal error during recall: {str(e)}")

    def _handle_management_search(self, query: str, action_type: str) -> str:
        """Helper to find top 3 semantic matches for confirmation workflow."""
        # 1. Embed query intent
        vec = self._enrichment.generate_embedding(query)
        # 2. Search
        # We use a large token budget to ensure we get meaningful snippets
        results = self._retriever.search(
            vec, 
            query_text=query,
            limit=3, 
            token_budget=4000, 
            allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"]
        )
        
        if not results:
            return f"No memories found matching '{query}'. Please try more specific terms."
            
        lines = [f"### Potential Matches for {action_type.capitalize()}"]
        lines.append(f"I found the following memories that might match your request. To proceed, call the `memory` tool again with the specific `memory_id` and action='{action_type}'.")
        lines.append("")
        for res in results:
            # Clean snippet for display
            snippet = res["content"].split("\n")[0] # Just the first line/user part often
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            
            lines.append(f"- **ID: {res['id']}** (Confidence: {res['score']:.2f})")
            lines.append(f"  Snippet: \"{snippet}\"")
            lines.append("")
            
        return "\n".join(lines)

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True
        
        if self._prefetch_thread or self._sync_thread:
            logger.info("ReverieCore: Synchronizing final memories and cleaning up (this may take a few seconds)...")
            
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=10.0)
        
        if self._mesa_service:
            self._mesa_service.stop()
            
        if self._mirror_service:
            self._mirror_service.stop()

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

    def export_all_memories(self):
        """Manual trigger to mirror all active memories to disk."""
        if not self._db or not self._mirror_service:
            return
        logger.info("ReverieCore: Starting bulk export of all memories...")
        cursor = self._db.get_cursor()
        cursor.execute("SELECT id FROM memories WHERE status = 'ACTIVE'")
        for (mid,) in cursor.fetchall():
            self._mirror_service.export_node(mid)
        logger.info("ReverieCore: Bulk export complete.")

    def import_from_archive(self, path: Optional[str] = None):
        """Manual trigger to ingest memories from a local archive."""
        if not self._mirror_service:
            return
        p = Path(path) if path else None
        self._mirror_service.import_archive(p)
