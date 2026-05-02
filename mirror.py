import os
import json
import logging
from pathlib import Path
from datetime import datetime
import threading
from typing import List, Dict, Optional, Any
import uuid

from .telemetry import get_tracer

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)

class MirrorService:
    """
    Bi-directional synchronization Service between SQLite and Markdown.
    Implements Memory-as-Code for data sovereignty and disaster recovery.
    """
    
    def __init__(self, db, enrichment, archive_root: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.db = db
        self.enrichment = enrichment
        
        # Priority: archive_root arg > config['archive_root'] > default
        if archive_root:
            self.archive_root = archive_root
        elif config and "archive_root" in config:
            self.archive_root = Path(config["archive_root"])
        else:
            # Fallback to current directory/archive if not provided
            self.archive_root = Path("reverie_archive")
            
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
                with tracer.start_as_current_span("reverie.mirror.worker_task") as span:
                    span.set_attribute("memory.id", node_id)
                    try:
                        self._revectorize(node_id)
                    except Exception as e:
                        logger.error(f"Lazy re-vectorization failed for memory {node_id}: {e}")
                        span.record_exception(e)
            else:
                if self._stop_event.wait(5.0):
                    break

    def _revectorize(self, memory_id: int):
        with tracer.start_as_current_span("reverie.mirror.revectorize") as span:
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

    def _get_relative_path(self, memory_dict: dict) -> str:
        """Derives the Hive-style relative path for a memory."""
        learned_at = memory_dict.get("learned_at")
        guid = memory_dict.get("guid")
        if isinstance(learned_at, str):
            try:
                dt = datetime.fromisoformat(learned_at.replace('Z', '+00:00'))
            except ValueError:
                dt = datetime.now()
        else:
            dt = learned_at or datetime.now()
        return f"year={dt.year:04d}/month={dt.month:02d}/day={dt.day:02d}/{guid}.md"

    def export_node(self, memory_id: int):
        """Fetches a memory and mirrors it to the local Markdown archive."""
        with tracer.start_as_current_span("reverie.mirror.export_node") as span:
            span.set_attribute("memory.id", memory_id)
            try:
                # 1. Fetch from DB
                memory = self.db.get_memory(memory_id)
                if not memory:
                    return

                guid = memory.get("guid")
                if not guid:
                    logger.warning(f"Cannot export memory {memory_id}: missing GUID.")
                    return

                # 1.5 Parse DateTime for formatting
                learned_at = memory.get("learned_at")
                if isinstance(learned_at, str):
                    try:
                        # Handle both ISO and SQLite formats
                        dt = datetime.fromisoformat(learned_at.replace(' ', 'T').replace('Z', '+00:00'))
                    except ValueError:
                        dt = datetime.now()
                else:
                    dt = learned_at or datetime.now()

                # 2. Build Frontmatter
                rel_path = self._get_relative_path(memory)
                frontmatter = {
                    "version": "1.0",
                    "guid": guid,
                    "path": rel_path,
                    "type": memory.get("memory_type"),
                    "importance": memory.get("importance_score"),
                    "status": memory.get("status"),
                    "owner": memory.get("owner_id"),
                    "learned_at": dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "abstract": memory.get("content_abstract"),
                    "relations": [],
                    "metadata": json.loads(memory.get("metadata") or "{}")
                }

                # 3. Add Relations (both direct and evidenced)
                rel_rows_direct = self.db.get_relations_for_node(memory_id, "MEMORY")
                rel_rows_evidenced = self.db.get_relations_by_evidence(memory_id)
                
                # Combine and deduplicate by ID
                seen_rel_ids = set()
                all_rels = []
                for r in rel_rows_direct + rel_rows_evidenced:
                    if r["id"] not in seen_rel_ids:
                        all_rels.append(r)
                        seen_rel_ids.add(r["id"])

                for a in all_rels:
                    other_guid = None
                    other_node_type = None
                    other_name = None
                    other_label = None
                    other_description = None
                    role = "target"
                    
                    # Determine "the other node"
                    if a["source_id"] == memory_id and a["source_type"] == "MEMORY":
                        # We are the source, other is target
                        role = "target"
                        other_node_type = a["target_type"]
                        if other_node_type == "MEMORY":
                            t = self.db.get_memory(a["target_id"])
                            if t:
                                other_guid = t["guid"]
                        else:
                            t = self.db.get_entity(a["target_id"])
                            if t:
                                other_guid = t["guid"]
                                other_name = t.get("name")
                                other_label = t.get("label")
                                other_description = t.get("description")
                    else:
                        # We are the evidence but not source or target (Triple E1 -> E2)
                        # We'll export the relation from the perspective of the source
                        role = "evidence"
                        other_node_type = a["source_type"]
                        if other_node_type == "MEMORY":
                            s = self.db.get_memory(a["source_id"])
                            if s: other_guid = s["guid"]
                        else:
                            s = self.db.get_entity(a["source_id"])
                            if s:
                                other_guid = s["guid"]
                                other_name = s.get("name")
                                other_label = s.get("label")
                                other_description = s.get("description")
                    
                    # If we were target, we also want the source's info
                    # But actually, the user format is 1 relation item per "other" node.
                    # For triples, we should probably add the target info to the metadata or something?
                    # For now, let's just make sure MENTIONS are handled which are 1-to-1.
                    
                    if other_guid:
                        # Construct with specific key order
                        assoc_entry = {}
                        if other_node_type == "ENTITY":
                            assoc_entry["name"] = other_name or "unknown"
                            assoc_entry["label"] = other_label or "unknown"
                        
                        assoc_entry["type"] = a["relation_type"]
                        assoc_entry["node_type"] = other_node_type
                        assoc_entry["confidence"] = a.get("confidence_score", 1.0)
                        assoc_entry["guid"] = other_guid
                        assoc_entry["role"] = role

                        if other_node_type == "ENTITY" and other_description:
                            assoc_entry["description"] = other_description
                            
                        frontmatter["relations"].append(assoc_entry)
                    else:
                        logger.warning(f"Skipping relation {a['id']} for memory {memory_id}: could not resolve other GUID (Type: {other_node_type}, ID: {a['source_id'] if role=='source' else a['target_id']})")

                # 4. Write File
                dt_path = self.archive_root / Path(rel_path).parent
                dt_path.mkdir(parents=True, exist_ok=True)
                file_path = self.archive_root / rel_path

                yaml_block = self._dump_yaml(frontmatter)
                content = memory.get("content_full", "")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("---\n")
                    f.write(yaml_block)
                    f.write("---\n\n")
                    f.write(content)
                
                logger.info(f"Exported memory {memory_id} to {file_path}")

            except Exception as e:
                logger.error(f"Failed to export memory {memory_id}: {e}")

    def import_archive(self, archive_path: Optional[Path] = None):
        """Crawls the archive and ingests missing or updated memories into SQLite."""
        with tracer.start_as_current_span("reverie.mirror.import_archive") as span:
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
                    
            # 2. Second Pass: Restore relations
            self._restore_relations(all_frontmatters)
            
            logger.info(f"Mirror import complete. Processed {len(all_frontmatters)} files.")

    def _restore_relations(self, all_frontmatters: list[dict]):
        """Re-links memories and entities using stable GUIDs."""
        for fm in all_frontmatters:
            current_guid = fm.get("guid")
            if not current_guid:
                continue
            rels = fm.get("relations") or fm.get("associations") or []
            for r in rels:
                try:
                    self._link_relation(current_guid, r)
                except Exception as e:
                    logger.error(f"Failed to restore relation {r}: {e}")

    def _link_relation(self, current_guid: str, r: dict):
        """Resolves GUIDs to local IDs and creates the relation record."""
        # Handle new format
        if "role" in r:
            if r["role"] == "target":
                source_guid = current_guid
                source_type = "MEMORY" # Assuming current is always MEMORY for now
                target_guid = r["guid"]
                target_type = r.get("node_type", "MEMORY")
            else:
                source_guid = r["guid"]
                source_type = r.get("node_type", "MEMORY")
                target_guid = current_guid
                target_type = "MEMORY"
            rel_type = r["type"]
        else:
            # Fallback for old format
            source_guid = r.get("source_guid")
            source_type = r.get("source_type", "MEMORY")
            target_guid = r.get("target_guid")
            target_type = r.get("target_type", "MEMORY")
            rel_type = r.get("relation_type") or r.get("relation_type")
        
        if not source_guid or not target_guid:
            return
            
        # Resolve local IDs
        source_id = None
        if source_type == "MEMORY":
            s = self.db.get_memory_by_guid(source_guid)
            source_id = s["id"] if s else None
        else:
            s = self.db.get_entity_by_guid(source_guid)
            if not s and source_type == "ENTITY" and "name" in r:
                with self.db.write_lock() as cursor:
                    cursor.execute("""
                        INSERT INTO entities (name, label, guid, description) VALUES (?, ?, ?, ?)
                    """, (r["name"], r.get("label", "ENTITY"), source_guid, r.get("description")))
                    source_id = cursor.lastrowid
            else:
                source_id = s["id"] if s else None
            
        target_id = None
        if target_type == "MEMORY":
            t = self.db.get_memory_by_guid(target_guid)
            target_id = t["id"] if t else None
        else:
            t = self.db.get_entity_by_guid(target_guid)
            if not t and target_type == "ENTITY" and "name" in r:
                # Disaster Recovery: Recreate entity from relation metadata
                with self.db.write_lock() as cursor:
                    cursor.execute("""
                        INSERT INTO entities (name, label, guid, description) VALUES (?, ?, ?, ?)
                    """, (r["name"], r.get("label", "ENTITY"), target_guid, r.get("description")))
                    target_id = cursor.lastrowid
            else:
                target_id = t["id"] if t else None
            
        if source_id and target_id:
            with self.db.write_lock() as cursor:
                # Check if exists
                cursor.execute("""
                    SELECT id FROM memory_relations 
                    WHERE source_id = ? AND source_type = ? 
                    AND target_id = ? AND target_type = ? 
                    AND relation_type = ?
                """, (source_id, source_type, target_id, target_type, rel_type))
                
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO memory_relations (
                            source_id, source_type, target_id, target_type, relation_type, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (source_id, source_type, target_id, target_type, rel_type, r.get("confidence", 1.0)))

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
            
            # Use the improved _load_yaml
            frontmatter = self._load_yaml(yaml_content.split("\n"))
            
            # Extract Abstract: prioritize frontmatter, fallback to blockquote for legacy support
            abstract = frontmatter.get("abstract")
            if not abstract and body.startswith("> "):
                body_parts = body.split("\n\n", 1)
                if len(body_parts) > 1:
                    abstract = body_parts[0][2:].strip()
                    body = body_parts[1].strip()
                else:
                    # Body is ONLY abstract?
                    abstract = body[2:].strip()
                    body = ""
            
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
        """Naive YAML dumper. Supports multi-line indented lists."""
        lines = []
        for k, v in data.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    if isinstance(item, dict):
                        first = True
                        for ik, iv in item.items():
                            val = json.dumps(iv) if isinstance(iv, (dict, list, str)) else iv
                            if first:
                                lines.append(f"  - {ik}: {val}")
                                first = False
                            else:
                                lines.append(f"    {ik}: {val}")
                    else:
                        lines.append(f"  - {item}")
            elif isinstance(v, dict):
                lines.append(f"{k}: {json.dumps(v)}")
            elif isinstance(v, str):
                # Ensure strings are properly escaped/quoted for YAML
                lines.append(f"{k}: {json.dumps(v)}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"

    def _load_yaml(self, lines: List[str]) -> Dict:
        """Naive YAML loader supporting multi-line indented lists."""
        data = {}
        current_key = None
        current_item = None
        
        for line in lines:
            if not line.strip() or line.strip().startswith("#"):
                continue
            
            indent = len(line) - len(line.lstrip())
            clean = line.strip()
            
            if clean.startswith("- "):
                val = clean[2:].strip()
                if ":" in val:
                    ik, iv = val.split(":", 1)
                    current_item = {ik.strip(): self._parse_val(iv.strip())}
                    if current_key:
                        if not isinstance(data.get(current_key), list):
                            data[current_key] = []
                        data[current_key].append(current_item)
                else:
                    if current_key:
                        if not isinstance(data.get(current_key), list):
                            data[current_key] = []
                        data[current_key].append(self._parse_val(val))
                continue
            
            if ":" in clean:
                k, v = clean.split(":", 1)
                k = k.strip()
                v = v.strip()
                
                if indent >= 4 and current_item is not None:
                    current_item[k] = self._parse_val(v)
                else:
                    current_key = k
                    if not v:
                        data[current_key] = []
                    else:
                        data[current_key] = self._parse_val(v)
                    current_item = None
        return data

    def _parse_val(self, v: str) -> Any:
        """Helper to parse YAML values."""
        if not v: return None
        if v.startswith('"') and v.endswith('"'): return v[1:-1]
        if v.lower() == "true": return True
        if v.lower() == "false": return False
        if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
            try: return json.loads(v)
            except: pass
        try:
            if "." in v: return float(v)
            return int(v)
        except: pass
        return v
