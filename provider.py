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
from .schemas import MemoryType, RelationType
from .pruning import MesaService
from .mirror import MirrorService
from opentelemetry import trace, context
from opentelemetry.trace import StatusCode
from .telemetry import initialize_telemetry, get_tracer

tracer = get_tracer(__name__)
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
        self.env = None
        self._last_retrieved_memories = []

    @property
    def name(self) -> str:
        return "reveriecore"

    def is_available(self) -> bool:
        # Locally hosted, so it's always available once initialized
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize models and database."""
        try:
            self._setup_logging()
            
            from hermes_constants import get_hermes_home
            from .config import load_reverie_config
            
            # 1. Load ReverieCore specific config
            self.config = load_reverie_config()
            system_cfg = self.config.get("system", {})
            telemetry_cfg = system_cfg.get("telemetry", {})
            
            # 2. Initialize Telemetry with config
            # We initialize AFTER loading config so user preferences are respected
            t_endpoint = telemetry_cfg.get("endpoint") if isinstance(telemetry_cfg, dict) else None
            t_enabled = telemetry_cfg.get("enabled", True) if isinstance(telemetry_cfg, dict) else True
            
            initialize_telemetry(
                service_name="reveriecore",
                endpoint=t_endpoint,
                enabled=t_enabled
            )

            
            # Database is pinned to the preferred .hermes directory
            db_path = get_hermes_home() / "reveriecore.db"
            
            # Identity Context (Provenance via Author/Actor, Scoping via Owner)
            self.session_id = session_id
            self.author_id = system_cfg.get("user_identity") or kwargs.get("user_id") or kwargs.get("user_identity") or "USER"
            self.owner_id = kwargs.get("agent_identity") or "PERSONAL_WORKSPACE"
            self.actor_id = "REVERIE_SYNC_SERVICE"
            self.workspace = kwargs.get("agent_workspace") or "UNKNOWN_WORKSPACE"
            self.agent_context = kwargs.get("agent_context") or "primary"
            
            try:
                self._db = DatabaseManager(str(db_path))
                self._enrichment = EnrichmentService()
                
                # Retrieval Context
                from .retrieval import RetrievalConfig, MaintenanceConfig
                ret_cfg = RetrievalConfig.from_dict(self.config)
                self._retriever = Retriever(self._db, enrichment=self._enrichment)
                
                # Maintenance Context
                maint_cfg = MaintenanceConfig.from_dict(self.config)
                
                # Capture memory_char_limit if passed from config (prioritize reveriecore.yaml)
                self.memory_char_limit = int(system_cfg.get("memory_char_limit", kwargs.get("memory_char_limit", self.memory_char_limit)))
                
                # Register for automatic cleanup on exit
                atexit.register(self.shutdown)
                
                # Initialize MirrorService (Memory-as-Code)
                archive_path = get_hermes_home() / "reverie_archive"
                self._mirror_service = MirrorService(self._db, self._enrichment, archive_root=archive_path)
                self._mirror_service.start()
                
                # Initialize Mesa Maintenance Service
                self._mesa_service = MesaService(
                    self._db, 
                    self._enrichment,
                    mirror=self._mirror_service,
                    config=maint_cfg.mesa
                )
                self._mesa_service.start()
                
                # Initial Environmental Context
                from .config import EnvironmentalContext
                self.env = EnvironmentalContext(
                    user_id=self.author_id,
                    agent_id=self.owner_id,
                    session_id=self.session_id,
                    remaining_tokens=self.remaining_tokens,
                    total_context=self.total_context
                )
                
                logger.info(f"ReverieCore initialized for {self.author_id} in {self.owner_id} (Actor: {self.actor_id})")
            except Exception as e:
                logger.error(f"ReverieCore initialization failed during sub-service startup: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            # Emergency fallback: write to a file we know we can find
            with open("/tmp/reverie_startup_error.log", "a") as f:
                import traceback
                f.write(f"\nFATAL STARTUP ERROR: {e}\n{traceback.format_exc()}\n")
            raise

    def system_prompt_block(self) -> str:
        return (
            "# Long-Term Memory (ReverieCore)\n"
            "Active. You have access to a local knowledge base of past interactions.\n"
            "Relevant context is automatically injected during the 'Preparing memory' phase.\n"
            "NOTE: Older or less relevant memories may be returned as abstracts (summaries) to save context window space."
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called by Hermes at the start of each turn with runtime context."""
        with tracer.start_as_current_span("reverie.provider.on_turn_start") as span:
            span.set_attribute("agent.turn_number", turn_number)
            self.remaining_tokens = kwargs.get("remaining_tokens") or self.remaining_tokens
        
        model_meta = kwargs.get("model_metadata") or {}
        if model_meta:
            self.total_context = model_meta.get("context_length") or self.total_context
            self.model_name = model_meta.get("model") or self.model_name
            
        # Capture environmental metadata (Location, Environment, etc.)
        self._last_turn_metadata = {}
        for key in ["location", "geolocation", "environment", "weather"]:
            if key in kwargs:
                self._last_turn_metadata[key] = kwargs[key]
            elif key in model_meta:
                self._last_turn_metadata[key] = model_meta[key]

        # Handle Dynamic Personality/Soul Updates
        soul_update = kwargs.get("soul_prompt") or kwargs.get("personality") or model_meta.get("personality")
        if soul_update and self._enrichment:
            self._enrichment.set_soul(soul_update)

        # Update Environmental Context
        if self.env:
            self.env.remaining_tokens = self.remaining_tokens
            self.env.total_context = self.total_context
            self.env.metadata = self._last_turn_metadata
            self.env.session_id = self.session_id
            
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
        with tracer.start_as_current_span("reverie.provider.prefetch") as span:
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
        with tracer.start_as_current_span("reverie.provider.queue_prefetch") as span:
            if not self._retriever or not self._enrichment:
                return
            
            current_ctx = context.get_current()
            
            def _run():
                token = context.attach(current_ctx)
                try:
                    with tracer.start_as_current_span("reverie.provider.prefetch_task") as task_span:
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
                                allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"],
                                env=self.env
                            )
                            if results:
                                self._last_retrieved_memories = results
                                lines = [f"- {r['content']}" for r in results]
                                with self._prefetch_lock:
                                    self._prefetch_result = "\n".join(lines)
                        except Exception as e:
                            logger.debug(f"ReverieCore prefetch failed: {e}")
                finally:
                    context.detach(token)
            
            self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="reverie-prefetch")
            self._prefetch_thread.start()

    def _save_memory_sync(self, user_content: str, assistant_content: str, session_id: str = "", metadata: dict = None) -> None:
        """Internal synchronous method to process and save memory."""
        with tracer.start_as_current_span("reverie.provider.save_memory_task") as span:
            # 0. Measure Retrieval Utility (Level 4 Metric)
            is_useful = False
            if self._last_retrieved_memories:
                assistant_words = set(assistant_content.lower().split())
                for mem in self._last_retrieved_memories:
                    # Strip the ID/Timestamp header for comparison
                    content_body = mem.get("content", "").split("Context:\n")[-1].lower()
                    overlap = [w for w in assistant_words if len(w) > 4 and w in content_body]
                    if len(overlap) >= 3: # Threshold: 3 shared technical/rare words
                        is_useful = True
                        break
            span.set_attribute("rag.retrieval.is_useful", is_useful)
            
            try:
                # For turn saving, we focus on the assistant's response or a composite
                full_text = f"User: {user_content}\nAssistant: {assistant_content}"
                
                # 1. Intelligence Layer Analysis (Modular Pipeline)
                ctx = self._enrichment.enrich(full_text, env=self.env)
                
                mem_type = ctx.memory_type
                importance = ctx.importance_score
                profile = ctx.profile
                vec = ctx.embedding
                
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
                    
                    # Re-analyze synthesized content (Modular Pipeline)
                    m_ctx = self._enrichment.enrich(merged_text, env=self.env)
                    
                    self._db.update_memory(
                        dup_id, 
                        merged_text, 
                        m_ctx.profile, 
                        m_ctx.embedding, 
                        m_ctx.token_count_full, 
                        m_ctx.token_count_abstract,
                        importance_score=m_ctx.importance_score,
                        metadata=metadata
                    )
                    logger.info(f"Memory {dup_id} Canonicalized and Updated.")
                    return 

                # 3. Standard Relational Store (if no merge found)
                tc_full = ctx.token_count_full
                tc_abstract = ctx.token_count_abstract
                
                with self._db.write_lock() as cursor:
                    import uuid
                    query = """
                        INSERT INTO memories (
                            content_full, content_abstract, 
                            token_count_full, token_count_abstract,
                            memory_type, importance_score, 
                            author_id, owner_id, actor_id, 
                            session_id, workspace, guid, metadata
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    params = (
                        full_text, profile, 
                        tc_full, tc_abstract,
                        mem_type.value, importance, 
                        self.author_id, self.owner_id, self.actor_id, 
                        session_id or self.session_id, self.workspace, str(uuid.uuid4()),
                        json.dumps(metadata) if metadata else None
                    )
                    with self._db.trace_query("INSERT", "memories", query, params) as span:
                        cursor.execute(query, params)
                        mem_id = cursor.lastrowid
                    
                    # 4. Vector Store
                    import sqlite_vec
                    vec_query = "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)"
                    vec_params = (mem_id, sqlite_vec.serialize_float32(vec))
                    with self._db.trace_query("INSERT", "memories_vec", vec_query, vec_params) as span:
                        cursor.execute(vec_query, vec_params)
                    
                # 5. Graph Extraction (Extract for all memories to ensure archive completeness)
                logger.info(f"Triggering enhanced graph extraction for memory {mem_id}")
                self._enrichment.extract_graph_data(full_text, mem_id, self._db)

                # 6. Mirror-as-Code: Export to Markdown archive immediately
                if self._mirror_service:
                    self._mirror_service.export_node(mem_id)

                logger.debug(f"Memory saved and exported: ID {mem_id}, Type {mem_type.value}, Score {importance}")
                
            except Exception as e:
                logger.warning(f"ReverieCore sync failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", metadata: dict = None) -> None:
        """Asynchronously cleans, scores, embeds, and saves the conversation turn."""
        if not self._enrichment or not self._db:
            return

        # Use captured metadata if none provided explicitly
        final_metadata = metadata or getattr(self, "_last_turn_metadata", None)

        # Non-blocking sync
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)

        current_ctx = context.get_current()
        def _run():
            token = context.attach(current_ctx)
            try:
                self._save_memory_sync(user_content, assistant_content, session_id, final_metadata)
            finally:
                context.detach(token)

        self._sync_thread = threading.Thread(
            target=_run, 
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
            },
            {
                "name": "mirror_archive",
                "description": "Synchronize the internal database with the local Markdown archive (Memory-as-Code). Use 'export' to back up all active memories to disk, or 'import' to ingest memories from an existing archive.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["export", "import"],
                            "description": "The action to perform: 'export' (DB -> Disk) or 'import' (Disk -> DB)."
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional: Specific path to import from (defaults to the standard archive root)."
                        }
                    },
                    "required": ["action"]
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        with tracer.start_as_current_span("reverie.provider.handle_tool_call") as span:
            span.set_attribute("gen_ai.tool.name", tool_name)
            if tool_name == "recall_reverie":
                return self._handle_recall_reverie(args.get("memory_id"))

            if tool_name == "mirror_archive":
                action = args.get("action")
                if action == "export":
                    self.export_all_memories()
                    return "Bulk export to Markdown archive completed successfully."
                elif action == "import":
                    path = args.get("path")
                    self.import_from_archive(path)
                    return "Archive import and synchronization completed successfully."
                return tool_error(f"Unsupported mirror action: {action}")

            if tool_name != "memory":
                return tool_error(f"Unknown tool: {tool_name}")

            action = args.get("action")
            span.set_attribute("memory_action", action)
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

                # Re-enrich the replacement text (Modular Pipeline)
                ctx = self._enrichment.enrich(replacement, env=self.env)
                
                self._db.update_memory(
                    mem_id, 
                    replacement, 
                    ctx.profile, 
                    ctx.embedding, 
                    ctx.token_count_full, 
                    ctx.token_count_abstract, 
                    importance_score=ctx.importance_score
                )
                return f"Memory ID {mem_id} has been successfully replaced with the new content."

        return tool_error(f"Unsupported memory action: {action}")

    def _handle_recall_reverie(self, mem_id: Optional[int]) -> str:
        """Handles the 'drill-down' request with strict multi-tenant validation."""
        with tracer.start_as_current_span("reverie.provider.recall_reverie") as span:
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
        with tracer.start_as_current_span("reverie.provider.management_search") as span:
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
        with tracer.start_as_current_span("reverie.provider.export_all_memories") as span:
            if not self._db or not self._mirror_service:
                return
            logger.info("ReverieCore: Starting bulk export of all memories...")
            cursor = self._db.get_cursor()
            query = "SELECT id FROM memories WHERE status = 'ACTIVE'"
            with self._db.trace_query("SELECT", "memories", query) as span:
                cursor.execute(query)
                for (mid,) in cursor.fetchall():
                    self._mirror_service.export_node(mid)
            logger.info("ReverieCore: Bulk export complete.")

    def import_from_archive(self, path: Optional[str] = None):
        """Manual trigger to ingest memories from a local archive."""
        with tracer.start_as_current_span("reverie.provider.import_from_archive") as span:
            if not self._mirror_service:
                return
            p = Path(path) if path else None
            self._mirror_service.import_archive(p)
