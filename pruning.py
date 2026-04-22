import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import os
import json
import uuid
from .database import DatabaseManager
from .enrichment import EnrichmentService
from .schemas import MemoryType

logger = logging.getLogger(__name__)


class MesaService:
    """
    Two-Tier Maintenance Service:
    Tier 1: Soft-Prune (Marks fragmented/stale memories as 'ARCHIVED')
    Tier 2: Deep Clean (Permanently deletes old archives and executes VACUUM)
    """

    def __init__(self, db: DatabaseManager, 
                 enrichment: EnrichmentService,
                 mirror: Any = None,
                 centrality_threshold: int = 2, 
                 age_days: int = 14, 
                 importance_cutoff: float = 4.0,
                 interval_seconds: int = 3600):
        self.db = db
        self.enrichment = enrichment
        self.mirror = mirror
        self.centrality_threshold = centrality_threshold
        self.max_age_days = age_days
        self.importance_cutoff = importance_cutoff
        self.interval_seconds = interval_seconds
        self.consolidation_threshold = 5 # Higher threshold for pattern recognition
        
        self.purge_enabled = os.environ.get("REVERIE_MESA_PURGE_ENABLED", "True").lower() == "true"
        self.dry_run = os.environ.get("REVERIE_MESA_DRY_RUN", "False").lower() == "true"
        
        self._stop_event = threading.Event()
        self._thread = None
        self.last_deep_clean = None

    def start(self):
        """Starts the maintenance loop in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="reverie-mesa")
        self._thread.start()
        logger.info(f"MesaService started (Interval: {self.interval_seconds}s, DryRun: {self.dry_run})")

    def stop(self):
        """Signals the background thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("MesaService stopped.")

    def _run_loop(self):
        """Main background loop."""
        while not self._stop_event.wait(self.interval_seconds):
            try:
                # 1. Soft Prune (Tier 1: Fragmentation Cleanup)
                self.run_soft_prune()
                
                # 1.5. Hierarchical Consolidation (Tier 1.5: The Tree of Nuance)
                self.run_hierarchical_consolidation()
                
                # 2. Deep Clean (Tier 2: Purge & Vacuum - Once a month)
                if self.purge_enabled and self._should_deep_clean():
                    self.run_deep_clean()
            except Exception as e:
                logger.error(f"MesaService loop error: {e}")

    def _should_deep_clean(self) -> bool:
        """Triggers deep clean every 30 days."""
        if self.last_deep_clean is None:
            return True
        return (datetime.now() - self.last_deep_clean).days >= 30

    def run_soft_prune(self):
        """Identifies fragmented memories and marks them as ARCHIVED."""
        logger.info("MesaService: Running Tier 1 (Soft Prune)...")
        
        # Logic: 
        # - importance_score < cutoff
        # - last_accessed_at < now - age_days
        # - status = 'ACTIVE'
        # - edge count < centrality_threshold
        
        try:
            # Subquery to find candidates
            candidate_query = f"""
                SELECT m.id FROM memories m
                LEFT JOIN (
                    SELECT source_id as node_id FROM memory_associations WHERE source_type = 'MEMORY'
                    UNION ALL
                    SELECT target_id as node_id FROM memory_associations WHERE target_type = 'MEMORY'
                ) a ON m.id = a.node_id
                WHERE m.importance_score < ?
                AND m.last_accessed_at < datetime('now', ?)
                AND m.status = 'ACTIVE'
                GROUP BY m.id
                HAVING COUNT(a.node_id) < ?
            """
            
            age_filter = f"-{self.max_age_days} days"
            cursor = self.db.get_cursor()
            cursor.execute(candidate_query, (self.importance_cutoff, age_filter, self.centrality_threshold))
            candidate_ids = [row[0] for row in cursor.fetchall()]
            
            if not candidate_ids:
                logger.debug("MesaService: No fragmentation detected.")
                return

            if self.dry_run:
                logger.info(f"MesaService [DRY RUN]: Would archive {len(candidate_ids)} memories: {candidate_ids}")
                return

            placeholders = ",".join(["?"] * len(candidate_ids))
            with self.db.write_lock() as cursor:
                cursor.execute(f"UPDATE memories SET status = 'ARCHIVED' WHERE id IN ({placeholders})", tuple(candidate_ids))
            
            logger.info(f"MesaService: Archived {len(candidate_ids)} fragmented memories.")

            # Mirror-as-Code: Export to local archive
            if self.mirror:
                for mid in candidate_ids:
                    self.mirror.export_node(mid)
            
        except Exception as e:
            logger.error(f"MesaService Soft Prune failed: {e}")

    def run_hierarchical_consolidation(self):
        """Identifies clusters of stale/fragmented memories and crystallizes them into Observation Anchors."""
        logger.info("MesaService: Running Tier 1.5 (Hierarchical Consolidation)...")
        try:
            cursor = self.db.get_cursor()
            
            # Find entities mentioned in clusters of stale/fragmented ACTIVE memories
            # Criteria: 
            # - status = 'ACTIVE'
            # - shared entity
            # - count >= threshold
            # - memories are candidates for pruning (stale OR low centrality)
            
            age_filter = f"-{self.max_age_days} days"
            
            query = f"""
                SELECT e.id, e.name, COUNT(ma.source_id) as c_count, GROUP_CONCAT(ma.source_id) as member_ids
                FROM memory_associations ma
                JOIN entities e ON ma.target_id = e.id
                JOIN memories m ON ma.source_id = m.id
                WHERE ma.source_type = 'MEMORY' 
                AND ma.target_type = 'ENTITY'
                AND m.status = 'ACTIVE'
                AND (m.last_accessed_at < datetime('now', ?) OR m.importance_score < ?)
                GROUP BY e.id
                HAVING c_count >= ?
            """
            cursor.execute(query, (age_filter, self.importance_cutoff, self.consolidation_threshold))
            clusters = cursor.fetchall()

            for ent_id, ent_name, count, member_ids_str in clusters:
                logger.debug(f"Cluster: Entity {ent_name}, Count {count}")
                member_ids = [int(i) for i in member_ids_str.split(',')]
                self._consolidate_to_hierarchy(member_ids, ent_name, ent_id)

        except Exception as e:
            logger.error(f"Hierarchical consolidation failed: {e}")

    def _consolidate_to_hierarchy(self, member_ids: List[int], entity_name: str, entity_id: int):
        """Creates a high-level Observation Anchor and archives children with CHILD_OF links."""
        logger.info(f"MesaService: Consolidating '{entity_name}' hierarchy ({len(member_ids)} fragments)...")
        
        try:
            # 0. Fetch content
            id_to_text = {}
            with self.db.write_lock() as cursor:
                placeholders = ",".join(["?"] * len(member_ids))
                cursor.execute(f"SELECT id, content_full FROM memories WHERE id IN ({placeholders})", tuple(member_ids))
                for mid, txt in cursor.fetchall():
                    id_to_text[mid] = txt
                
            if not id_to_text:
                return

            # 1. Hierarchical Synthesis
            summary_text = self.enrichment.synthesize_memories(id_to_text, entity_name)
            
            # 2. Extract Profile & Importance
            profile = self.enrichment.generate_semantic_profile(summary_text)
            imp_data = self.enrichment.calculate_importance(summary_text)
            
            with self.db.write_lock() as cursor:
                # 3. Save Observation Anchor
                metadata = json.dumps({"source_ids": member_ids, "consensus_target": entity_name})
                cursor.execute("""
                    INSERT INTO memories (
                        content_full, content_abstract, importance_score, memory_type, status, metadata, guid
                    ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
                """, (summary_text, profile, 4.5, "OBSERVATION", metadata, str(uuid.uuid4()))) # Force high importance for anchors
                
                anchor_id = cursor.lastrowid
                
                # Vector
                vec = self.enrichment.generate_embedding(profile)
                import sqlite_vec
                cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                            (anchor_id, sqlite_vec.serialize_float32(vec)))

                # 4. Link Hierarchy and Archive
                for mid in member_ids:
                    # CHILD_OF link
                    cursor.execute("""
                        INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type)
                        VALUES (?, 'MEMORY', ?, 'MEMORY', 'CHILD_OF')
                    """, (mid, anchor_id))
                    
                    # Also keep SUPERSEDES for backward compatibility
                    cursor.execute("""
                        INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type)
                        VALUES (?, 'MEMORY', ?, 'MEMORY', 'SUPERSEDES')
                    """, (anchor_id, mid))
                    
                    # Archive source
                    cursor.execute("UPDATE memories SET status = 'ARCHIVED' WHERE id = ?", (mid,))
                    
                    # Mirror-as-Code: Export child archive
                    if self.mirror:
                        self.mirror.export_node(mid)

                # 5. Link anchor to entity
                cursor.execute("""
                    INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type)
                    VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')
                """, (anchor_id, entity_id))

            logger.info(f"MesaService: Hierarchical crystallization complete. Anchor: {anchor_id}")

            # Mirror-as-Code: Export new anchor
            if self.mirror:
                self.mirror.export_node(anchor_id)

        except Exception as e:
            logger.error(f"Hierarchical crystallization for {entity_name} failed: {e}")
            self.db.conn.rollback()

    def run_deep_clean(self):
        """Tier 2: Purges old archives and VACUUMs the database."""
        logger.info("MesaService: Running Tier 2 (Deep Clean)...")
        
        try:
            if self.dry_run:
                logger.info("MesaService [DRY RUN]: Skipping Deep Clean.")
                return

            with self.db.write_lock() as cursor:
                # 1. Delete ARCHIVED memories older than 90 days
                cursor.execute("DELETE FROM memories WHERE status = 'ARCHIVED' AND learned_at < datetime('now', '-90 days')")
                purged_count = cursor.rowcount
                
                # 2. Cleanup orphaned vector entries (if any)
                cursor.execute("DELETE FROM memories_vec WHERE rowid NOT IN (SELECT id FROM memories)")
                
                logger.info(f"MesaService: Purged {purged_count} records.")

            # 3. VACUUM to reclaim space (MUST be outside transaction)
            try:
                # Using a fresh cursor directly from connection to be safe
                self.db.conn.execute("VACUUM")
                logger.info("MesaService: VACUUM executed successfully.")
            except Exception as ev:
                logger.warning(f"MesaService: VACUUM failed (likely concurrent access): {ev}")
                
            self.last_deep_clean = datetime.now()
            
        except Exception as e:
            logger.error(f"MesaService Deep Clean failed: {e}")
