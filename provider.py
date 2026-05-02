import os
# Must be set BEFORE other imports to ensure libraries pick these up immediately
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import logging
import threading
import time
import atexit
import warnings
import sys
import logging.handlers
from typing import Any, Dict, List, Optional
from pathlib import Path

# OpenTelemetry imports
from opentelemetry import trace, context
from opentelemetry.trace import StatusCode

# Internal package imports
from .database import DatabaseManager
from .enrichment import EnrichmentService
from .schemas import MemoryType, RelationType, RetrievalConfig, MaintenanceConfig
from .pruning import MesaService
from .mirror import MirrorService
from .telemetry import initialize_telemetry, get_tracer

# Base class and registry
from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

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
        self.total_context = 32768
        self.remaining_tokens = 32768
        self.memory_char_limit = 2200
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
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize models and database."""
        try:
            self._setup_logging()
            
            from hermes_constants import get_hermes_home
            from .config import load_reverie_config
            from .retrieval import Retriever # LATE IMPORT TO AVOID CIRCULARITY
            
            # 1. Load ReverieCore specific config
            self.config = load_reverie_config()
            
            # 2. Get local storage paths
            hermes_home = get_hermes_home()
            db_path = Path(hermes_home) / "reveriecore.db"
            
            # 3. Handle specific config overrides
            system_cfg = self.config.get("system", {})
            self.author_id = system_cfg.get("author_id", kwargs.get("author_id", self.author_id))
            self.owner_id = system_cfg.get("owner_id", kwargs.get("owner_id", self.owner_id))
            self.actor_id = "REVERIE_SYNC_SERVICE"
            self.workspace = kwargs.get("agent_workspace") or "UNKNOWN_WORKSPACE"
            self.agent_context = kwargs.get("agent_context") or "primary"
            
            try:
                self._db = DatabaseManager(str(db_path))
                self._enrichment = EnrichmentService()
                
                # Retrieval Context
                self._retriever = Retriever(self._db, enrichment=self._enrichment)
                
                # Maintenance Context
                maint_cfg = MaintenanceConfig.from_dict(self.config)
                
                # Capture memory_char_limit
                self.memory_char_limit = int(system_cfg.get("memory_char_limit", kwargs.get("memory_char_limit", self.memory_char_limit)))
                
                # Register for automatic cleanup on exit
                atexit.register(self.shutdown)
                
                # Setup Maintenance service
                self._mesa_service = MesaService(self._db, self._enrichment, config=maint_cfg.mesa)
                
                # Setup Mirror service
                mirror_cfg = self.config.get("mirror", {})
                self._mirror_service = MirrorService(self._db, self._enrichment, config=mirror_cfg)
                
                # Start background maintenance if not in a test env
                if not kwargs.get("disable_background"):
                    self._start_background_tasks()
                
                self.session_id = session_id
                logger.info(f"ReverieMemoryProvider initialized successfully for session: {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize database or services: {e}", exc_info=True)
                raise
                
        except Exception as e:
            logger.error(f"Critical initialization error: {e}", exc_info=True)
            raise

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

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
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
            self.sync_turn("Manual Entry", text)
            return "Memory has been successfully queued for enrichment and storage."

        if action == "remove":
            if not mem_id:
                if not text:
                    return tool_error("Search text is required to find memories for removal.")
                return self._handle_management_search(text, "remove")
            else:
                memory = self._db.get_memory(mem_id)
                if not memory:
                    return tool_error(f"Memory ID {mem_id} not found.")
                self._db.delete_memory(mem_id)
                return f"Memory ID {mem_id} ('{memory['content_full'][:50]}...') has been permanently deleted."

        if action == "replace":
            if not mem_id:
                if not text:
                    return tool_error("Search text is required to find memories for replacement.")
                return self._handle_management_search(text, "replace")
            else:
                if not replacement:
                    return tool_error("Replacement text is required for the 'replace' action.")
                
                existing = self._db.get_memory(mem_id)
                if not existing:
                    return tool_error(f"Memory ID {mem_id} not found.")

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

    def on_turn_start(self, turn_context: Dict[str, Any]) -> None:
        self.remaining_tokens = turn_context.get("remaining_tokens", self.total_context)
        self.model_name = turn_context.get("model", "unknown")
        self.env = turn_context.get("env")
        if turn_context.get("history") and not self._is_shutdown:
            self._start_prefetch(turn_context["history"])

    def on_turn_end(self, turn_context: Dict[str, Any]) -> None:
        pass

    def recall(self, query: str, **kwargs) -> str:
        with tracer.start_as_current_span("reverie.provider.recall") as span:
            span.set_attribute("memory.query", query)
            if not self._retriever:
                logger.error("Recall called before initialization")
                return "Error: Memory provider not initialized."

            with self._prefetch_lock:
                if self._prefetch_result and query in self._prefetch_result:
                    res = self._prefetch_result
                    self._prefetch_result = ""
                    return res

            try:
                query_vector = self._enrichment.generate_embedding(query)
                limit = kwargs.get("limit", 5)
                token_budget = kwargs.get("token_budget", int(self.remaining_tokens * 0.15))
                results = self._retriever.search(
                    query_vector, 
                    query_text=query,
                    limit=limit,
                    token_budget=token_budget,
                    strategy=kwargs.get("strategy", "balanced"),
                    env=self.env
                )
                self._last_retrieved_memories = results
                if not results:
                    return "No relevant memories found."
                return "\n\n".join([r["content"] for r in results])
            except Exception as e:
                logger.error(f"Recall failed: {e}", exc_info=True)
                return f"Error during memory retrieval: {e}"

    def remember(self, content: str, **kwargs) -> bool:
        with tracer.start_as_current_span("reverie.provider.remember") as span:
            span.set_attribute("memory.content_len", len(content))
            if not self._db or not self._enrichment:
                return False

            try:
                m_type = kwargs.get("memory_type", "OBSERVATION")
                if isinstance(m_type, str):
                    try:
                        m_type = MemoryType(m_type)
                    except:
                        m_type = MemoryType.OBSERVATION
                importance = kwargs.get("importance", 5)
                memory_id = self._db.add_memory(
                    content=content,
                    memory_type=m_type.value,
                    importance_score=importance,
                    owner_id=self.owner_id
                )
                embedding = self._enrichment.generate_embedding(content)
                self._db.update_embedding(memory_id, embedding)
                return True
            except Exception as e:
                logger.error(f"Remember failed: {e}", exc_info=True)
                return False

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", metadata: dict = None) -> None:
        """Asynchronously cleans, scores, embeds, and saves the conversation turn."""
        if not self._enrichment or not self._db:
            return

        final_metadata = metadata or getattr(self, "_last_turn_metadata", None)

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

    def _save_memory_sync(self, user_content: str, assistant_content: str, session_id: str = "", metadata: dict = None) -> None:
        """Internal synchronous method to process and save memory."""
        with tracer.start_as_current_span("reverie.provider.save_memory_task") as span:
            is_useful = False
            if self._last_retrieved_memories:
                assistant_words = set(assistant_content.lower().split())
                for mem in self._last_retrieved_memories:
                    content_body = mem.get("content", "").split("Context:\n")[-1].lower()
                    overlap = [w for w in assistant_words if len(w) > 4 and w in content_body]
                    if len(overlap) >= 3:
                        is_useful = True
                        break
            span.set_attribute("retrieval.is_useful", is_useful)
            
            try:
                full_text = f"User: {user_content}\nAssistant: {assistant_content}"
                ctx = self._enrichment.enrich(full_text, env=self.env)
                
                m_type = ctx.memory_type
                importance = ctx.importance_score
                profile = ctx.profile
                vec = ctx.embedding
                
                duplicates = self._retriever.find_duplicates(
                    vec, 
                    threshold=0.95, 
                    allowed_owners=[self.owner_id, "PERSONAL_WORKSPACE"]
                )
                
                if duplicates:
                    dup = duplicates[0]
                    dup_id = dup["id"]
                    logger.info(f"Canonical Merge: Match found (ID: {dup_id}). Synthesizing...")
                    
                    merged_text = self._enrichment.synthesize_memories(
                        {dup_id: dup["content_full"], -1: full_text}, 
                        "canonical_knowledge"
                    )
                    
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
                    return 

                tc_full = ctx.token_count_full
                tc_abstract = ctx.token_count_abstract
                
                import uuid
                memory_id = self._db.add_memory(
                    content_full=full_text,
                    content_abstract=profile,
                    token_count_full=tc_full,
                    token_count_abstract=tc_abstract,
                    memory_type=m_type.value,
                    importance_score=importance,
                    author_id=self.author_id,
                    owner_id=self.owner_id,
                    actor_id=self.actor_id,
                    session_id=session_id or self.session_id,
                    workspace=self.workspace,
                    guid=str(uuid.uuid4()),
                    metadata=json.dumps(metadata) if metadata else None
                )
                
                import sqlite_vec
                self._db.update_embedding(memory_id, vec)
                
            except Exception as e:
                logger.error(f"save_memory_sync failed: {e}", exc_info=True)

    def _handle_recall_reverie(self, mem_id: Optional[int]) -> str:
        """Handles the 'drill-down' request with strict multi-tenant validation."""
        with tracer.start_as_current_span("reverie.provider.recall_reverie") as span:
            if not mem_id:
                return tool_error("memory_id is required for recall_reverie.")
            
        try:
            memory = self._db.get_memory(mem_id)
            if not memory:
                return tool_error(f"Memory fragment {mem_id} not found.")
                
            if memory.get("owner_id") == self.owner_id or memory.get("owner_id") == "PERSONAL_WORKSPACE":
                is_authorized = True
            else:
                is_authorized = self._db.check_provenance_access(mem_id, self.owner_id)
                
            if not is_authorized:
                logger.warning(f"SECURITY ALERT: Actor {self.actor_id} attempted unauthorized recall of memory {mem_id}")
                return tool_error("Access Denied. You are not authorized to recall this memory fragment.")

            content = memory.get("content_full", "")
            m_type = memory.get("memory_type", "FRAGMENT")
            return f"### RECALLED NUANCE (ID: {mem_id}, Type: {m_type})\n\n{content}"

        except Exception as e:
            logger.error(f"recall_reverie failed for {mem_id}: {e}")
            return tool_error(f"Internal error during recall: {str(e)}")

    def _handle_management_search(self, query: str, action_type: str) -> str:
        """Helper to find top 3 semantic matches for confirmation workflow."""
        with tracer.start_as_current_span("reverie.provider.management_search") as span:
            vec = self._enrichment.generate_embedding(query)
        
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
            snippet = res["content"].split("\n")[0]
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
        from hermes_constants import get_hermes_home
        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "reverie.log"

        pkg_name = __name__.split('.')[0]
        parent_logger = logging.getLogger(pkg_name)
        parent_logger.setLevel(logging.INFO)
        parent_logger.propagate = False

        for h in parent_logger.handlers[:]:
            parent_logger.removeHandler(h)

        try:
            handler = logging.handlers.RotatingFileHandler(
                str(log_file), 
                maxBytes=10*1024*1024,
                backupCount=5
            )
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            parent_logger.addHandler(handler)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        except Exception as e:
            print(f"ReverieCore: Failed to setup logging file {log_file}: {e}", file=sys.stderr)

        noisy_libs = ["transformers", "huggingface_hub", "urllib3", "torch", "sentence_transformers", "filelock"]
        for lib in noisy_libs:
            logging.getLogger(lib).setLevel(logging.ERROR)

    def _start_background_tasks(self):
        if not self._mesa_service:
            return
        def run_mesa():
            while not self._is_shutdown:
                try:
                    self._mesa_service.run_cycle()
                except Exception as e:
                    logger.error(f"Background Mesa cycle failed: {e}")
                time.sleep(self._mesa_service.interval_seconds)
        self._sync_thread = threading.Thread(target=run_mesa, daemon=True)
        self._sync_thread.start()

    def _start_prefetch(self, history: List[Dict[str, Any]]):
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        if not last_user:
            return
        def do_prefetch():
            try:
                res = self.recall(last_user, limit=3)
                with self._prefetch_lock:
                    self._prefetch_result = res
            except: pass
        self._prefetch_thread = threading.Thread(target=do_prefetch, daemon=True)
        self._prefetch_thread.start()

    def export_all_memories(self):
        with tracer.start_as_current_span("reverie.provider.export_all_memories") as span:
            if not self._db or not self._mirror_service:
                return
            cursor = self._db.get_cursor()
            query = "SELECT id FROM memories WHERE status = 'ACTIVE'"
            cursor.execute(query)
            for (mid,) in cursor.fetchall():
                self._mirror_service.export_node(mid)

    def import_from_archive(self, path: Optional[str] = None):
        with tracer.start_as_current_span("reverie.provider.import_from_archive") as span:
            if not self._mirror_service:
                return
            p = Path(path) if path else None
            self._mirror_service.import_archive(p)
