import os
import json
import logging
from pathlib import Path
from datetime import datetime
import threading
from typing import List, Dict, Optional
import uuid

logger = logging.getLogger(__name__)

class MirrorService:
    """
    Bi-directional synchronization Service between SQLite and Markdown.
    Implements Memory-as-Code for data sovereignty and disaster recovery.
    """
    
    def __init__(self, db, enrichment, archive_root: Optional[Path] = None):
        self.db = db
        self.enrichment = enrichment
        
        if archive_root is None:
            # Fallback to current directory/archive if not provided
            self.archive_root = Path("reverie_archive")
        else:
            self.archive_root = archive_root
            
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self._reembedding_queue = []
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._queue_lock = threading.Lock()

    def start(self):
        """Starts the background worker for lazy re-vectorization."""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="reverie-mirror-worker")
        self._worker_thread.start()
        logger.info("MirrorService background worker started.")

    def stop(self):
        """Signals the background worker to stop."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            node_id = None
            with self._queue_lock:
                if self._reembedding_queue:
                    node_id = self._reembedding_queue.pop(0)
            
            if node_id:
                try:
                    self._revectorize(node_id)
                except Exception as e:
                    logger.error(f"Lazy re-vectorization failed for memory {node_id}: {e}")
            else:
                if self._stop_event.wait(5.0):
                    break

    def _revectorize(self, memory_id: int):
        memory = self.db.get_memory(memory_id)
        if not memory:
            return
            
        # Standard approach: Embed the abstract (semantic profile) if available
        content = memory.get("content_abstract") or memory.get("content_full")
        if not content:
            return
        
        logger.debug(f"Lazy re-vectorizing memory {memory_id}...")
        vec = self.enrichment.generate_embedding(content)
        
        # We need token counts for update_memory
        tc_full = self.enrichment.count_tokens(memory.get("content_full", ""))
        tc_abstract = self.enrichment.count_tokens(memory.get("content_abstract", ""))
        
        self.db.update_memory(
            memory_id, 
            memory.get("content_full"), 
            memory.get("content_abstract"), 
            vec, 
            tc_full, 
            tc_abstract
        )

    def export_node(self, memory_id: int):
        """Fetches a memory and mirrors it to the local Markdown archive."""
        try:
            # 1. Fetch from DB
            memory = self.db.get_memory(memory_id)
            if not memory:
                return

            guid = memory.get("guid")
            if not guid:
                logger.warning(f"Cannot export memory {memory_id}: missing GUID.")
                return

            # 2. Get Associations
            associations = self.db.get_associations_for_node(memory_id, 'MEMORY')
            assoc_data = []
            for a in associations:
                # We need GUIDs for the targets/sources to maintain portability
                target_guid = self._get_guid_for_node(a['target_id'], a['target_type'])
                source_guid = self._get_guid_for_node(a['source_id'], a['source_type'])
                assoc_data.append({
                    "source": source_guid,
                    "target": target_guid,
                    "type": a['association_type'],
                    "confidence": a['confidence_score']
                })

            # 3. Build Frontmatter
            learned_at = memory.get("learned_at")
            if isinstance(learned_at, str):
                dt = datetime.fromisoformat(learned_at.replace('Z', '+00:00'))
            else:
                dt = learned_at or datetime.now()
            
            frontmatter = {
                "version": "1.0",
                "guid": guid,
                "type": memory.get("memory_type"),
                "importance": memory.get("importance_score"),
                "status": memory.get("status"),
                "owner": memory.get("owner_id"),
                "learned_at": learned_at,
                "associations": [],
                "metadata": json.loads(memory.get("metadata") or "{}")
            }

            # Add associations
            assocs = self.db.get_associations_for_node(memory_id, "MEMORY")
            for a in assocs:
                # We resolve local IDs to stable GUIDs for the archive
                source_guid = None
                target_guid = None
                
                # Resolve Source
                if a["source_type"] == "MEMORY":
                    s = self.db.get_memory(a["source_id"])
                    source_guid = s["guid"] if s else None
                else:
                    s = self.db.get_entity(a["source_id"])
                    source_guid = s["guid"] if s else None
                    
                # Resolve Target
                if a["target_type"] == "MEMORY":
                    t = self.db.get_memory(a["target_id"])
                    target_guid = t["guid"] if t else None
                else:
                    t = self.db.get_entity(a["target_id"])
                    target_guid = t["guid"] if t else None
                    
                if source_guid and target_guid:
                    frontmatter["associations"].append({
                        "source_guid": source_guid,
                        "source_type": a["source_type"],
                        "target_guid": target_guid,
                        "target_type": a["target_type"],
                        "association_type": a["association_type"],
                        "confidence": a.get("confidence_score", 1.0)
                    })

            # 4. Hive Pathing: archive/year=YYYY/month=MM/day=DD/{guid}.md
            path = self.archive_root / f"year={dt.year:04d}" / f"month={dt.month:02d}" / f"day={dt.day:02d}"
            path.mkdir(parents=True, exist_ok=True)
            file_path = path / f"{guid}.md"

            # 5. Write File
            yaml_block = self._dump_yaml(frontmatter)
            content = memory.get("content_full", "")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write(yaml_block)
                f.write("---\n\n")
                if memory.get("content_abstract"):
                    f.write(f"> {memory.get('content_abstract')}\n\n")
                f.write(content)
            
            logger.info(f"Exported memory {memory_id} to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export memory {memory_id}: {e}")

    def import_archive(self, archive_path: Optional[Path] = None):
        """Crawls the archive and ingests missing or updated memories into SQLite."""
        self.archive_root = archive_path or self.archive_root
        if not self.archive_root.exists():
            return

        logger.info(f"Starting import from archive: {self.archive_root}")
        
        # 1. First Pass: Import all memories and entities
        all_frontmatters = []
        for file_path in self.archive_root.glob("**/*.md"):
            try:
                fm = self._import_file(file_path)
                if fm:
                    all_frontmatters.append(fm)
            except Exception as e:
                logger.error(f"Failed to import {file_path}: {e}")
                
        # 2. Second Pass: Restore associations
        self._restore_associations(all_frontmatters)
        
        logger.info(f"Mirror import complete. Processed {len(all_frontmatters)} files.")

    def _restore_associations(self, all_frontmatters: list[dict]):
        """Re-links memories and entities using stable GUIDs."""
        for fm in all_frontmatters:
            assocs = fm.get("associations", [])
            for a in assocs:
                try:
                    self._link_association(a)
                except Exception as e:
                    logger.error(f"Failed to restore association {a}: {e}")

    def _link_association(self, a: dict):
        """Resolves GUIDs to local IDs and creates the association record."""
        source_guid = a["source_guid"]
        target_guid = a["target_guid"]
        
        # Resolve local IDs
        source_id = None
        if a["source_type"] == "MEMORY":
            s = self.db.get_memory_by_guid(source_guid)
            source_id = s["id"] if s else None
        else:
            s = self.db.get_entity_by_guid(source_guid)
            source_id = s["id"] if s else None
            
        target_id = None
        if a["target_type"] == "MEMORY":
            t = self.db.get_memory_by_guid(target_guid)
            target_id = t["id"] if t else None
        else:
            t = self.db.get_entity_by_guid(target_guid)
            target_id = t["id"] if t else None
            
        if source_id and target_id:
            with self.db.write_lock() as cursor:
                # Check if exists
                cursor.execute("""
                    SELECT id FROM memory_associations 
                    WHERE source_id = ? AND source_type = ? 
                    AND target_id = ? AND target_type = ? 
                    AND association_type = ?
                """, (source_id, a["source_type"], target_id, a["target_type"], a["association_type"]))
                
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO memory_associations (
                            source_id, source_type, target_id, target_type, association_type, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (source_id, a["source_type"], target_id, a["target_type"], a["association_type"], a.get("confidence", 1.0)))

    def _import_file(self, file_path: Path) -> Optional[dict]:
        """Imports a single markdown file into the database."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple YAML parser (since we don't have pyyaml)
            parts = content.split("---")
            if len(parts) < 3:
                return None
            
            yaml_content = parts[1]
            body = "---".join(parts[2:]).strip()
            
            # Extract Abstract from body if present (formatted as blockquote)
            abstract = None
            if body.startswith("> "):
                body_parts = body.split("\n\n", 1)
                if len(body_parts) > 1:
                    abstract = body_parts[0][2:].strip()
                    body = body_parts[1].strip()
                else:
                    # Body is ONLY abstract?
                    abstract = body[2:].strip()
                    body = ""

            # Naive YAML loader
            frontmatter = {}
            for line in yaml_content.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Handle basic types
                    if v.startswith("[") and v.endswith("]"):
                        try:
                            v = json.loads(v)
                        except: pass
                    elif v.startswith("{"):
                        try:
                            v = json.loads(v)
                        except: pass
                    elif v.lower() == "true": v = True
                    elif v.lower() == "false": v = False
                    elif v.isdigit(): v = int(v)
                    elif v.startswith('"') and v.endswith('"'): v = v[1:-1]
                    frontmatter[k] = v
            
            guid = frontmatter.get("guid")
            if not guid:
                return None
            
            status = frontmatter.get("status", "ACTIVE")
            
            # Handle PURGED status
            if status == "PURGED":
                existing = self.db.get_memory_by_guid(guid)
                if existing:
                    self.db.delete_memory(existing["id"])
                    logger.info(f"Purged memory {guid} based on archive status.")
                return frontmatter

            # Logic for Restore/Update
            existing = self.db.get_memory_by_guid(guid)
            metadata_json = json.dumps(frontmatter.get("metadata", {}))
            
            if existing:
                # Update existing
                with self.db.write_lock() as cursor:
                    cursor.execute("""
                        UPDATE memories SET 
                            content_full = ?, content_abstract = ?, status = ?, learned_at = ?, metadata = ?
                        WHERE guid = ?
                    """, (body, abstract, status, frontmatter.get("learned_at"), metadata_json, guid))
            else:
                # Insert new
                with self.db.write_lock() as cursor:
                    cursor.execute("""
                        INSERT INTO memories (
                            content_full, content_abstract, guid, memory_type, 
                            importance_score, status, learned_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (body, abstract, guid, frontmatter.get("type", "CONVERSATION"),
                          frontmatter.get("importance", 1.0), status,
                          frontmatter.get("learned_at"), metadata_json))
                    mem_id = cursor.lastrowid
                    
                    # Queue for re-vectorization
                    with self._queue_lock:
                        self._reembedding_queue.append(mem_id)
            
            return frontmatter
        except Exception as e:
            logger.error(f"Error importing {file_path}: {e}")
            return None

    def _get_guid_for_node(self, node_id: int, node_type: str) -> Optional[str]:
        cursor = self.db.get_cursor()
        if node_type == 'MEMORY':
            cursor.execute("SELECT guid FROM memories WHERE id = ?", (node_id,))
        else:
            cursor.execute("SELECT guid FROM entities WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _dump_yaml(self, data: Dict) -> str:
        """Naive YAML dumper for simple structures."""
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {json.dumps(v)}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"

    def _load_yaml(self, lines: List[str]) -> Dict:
        """Naive YAML loader for simple key-value pairs and JSON fields."""
        data = {}
        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            # Try parsing as JSON if it looks like a list or dict
            if v.startswith("[") or v.startswith("{"):
                try:
                    v = json.loads(v)
                except:
                    pass
            data[k] = v
        return data
