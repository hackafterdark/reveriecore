from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import threading
import os
import json
import uuid
import logging
from .database import DatabaseManager
from .enrichment import EnrichmentService
from .schemas import MemoryType, RetrievalContext, RetrievalHandler, MesaConfig
from .telemetry import get_tracer
from .mirror import MirrorService

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class PruningEngine:
    """Stateless utility for filtering retrieval candidates based on score quality."""

    @staticmethod
    def prune(candidates: Dict[int, Dict[str, Any]], 
              top_n: int = 3, 
              relative_threshold: float = 0.5, 
              min_absolute_score: float = 0.3) -> Dict[int, Dict[str, Any]]:
        """
        Filters candidates using a multi-gate approach:
        1. Top-N limit.
        2. Relative threshold (percentage of the top score).
        3. Absolute floor (hard cutoff).
        """
        if not candidates:
            return {}

        # 1. Sort by score
        sorted_items = sorted(
            candidates.items(), 
            key=lambda x: x[1].get("score", 0.0), 
            reverse=True
        )

        top_score = sorted_items[0][1].get("score", 0.0)
        
        # 2. Filter
        pruned = {}
        for cid, cand in sorted_items[:top_n]:
            score = cand.get("score", 0.0)
            
            # Relative Gate: Must be within X% of the best result
            is_relative_match = score >= (top_score * relative_threshold)
            
            # Absolute Gate: Must be above the hard floor
            is_absolute_match = score >= min_absolute_score
            
            if is_relative_match and is_absolute_match:
                pruned[cid] = cand
        
        return pruned


class PruningHandler(RetrievalHandler):
    """Stage E: Quality-based Pruning (The 'Junk' Sieve)"""

    def process(self, context: RetrievalContext, retriever: Any) -> None:
        if not context.candidates:
            return

        # Config resolution (expects a PruningConfig-like object or dict)
        if hasattr(self.config, "top_n"):
            top_n = self.config.top_n
            rel_thresh = self.config.relative_threshold
            min_abs = self.config.min_absolute_score
        else:
            # Fallback for manual dict injection
            cfg = self.config or {}
            top_n = cfg.get("top_n", 3)
            rel_thresh = cfg.get("relative_threshold", 0.5)
            min_abs = cfg.get("min_absolute_score", 0.3)

        original_count = len(context.candidates)
        
        # Execute Pruning
        context.candidates = PruningEngine.prune(
            context.candidates,
            top_n=top_n,
            relative_threshold=rel_thresh,
            min_absolute_score=min_abs
        )
        
        pruned_count = original_count - len(context.candidates)
        
        # Telemetry
        context.metrics["pruning"] = {
            "pruned": pruned_count,
            "remaining": len(context.candidates),
            "top_n_limit": top_n,
            "relative_threshold": rel_thresh,
            "min_absolute_score": min_abs
        }
        
        if pruned_count > 0:
            logger.info(f"PruningHandler: Filtered {pruned_count} noise candidates. Remaining: {len(context.candidates)}")


class MesaService:
    """
    Two-Tier Maintenance Service:
    Tier 1: Soft-Prune (Marks fragmented/stale memories as 'ARCHIVED')
    Tier 2: Deep Clean (Permanently deletes old archives and executes VACUUM)
    """

    def __init__(self, db: DatabaseManager, 
                 enrichment: EnrichmentService,
                 mirror: Any = None,
                 config: Optional[Any] = None):
        self.db = db
        self.enrichment = enrichment
        self.mirror = mirror
        
        # 1. Use injected config or safe defaults
        if config:
            self.config = config
        else:
            # Fallback for legacy or direct instantiation
            from .retrieval import MesaConfig
            self.config = MesaConfig()

        self.centrality_threshold = self.config.centrality_threshold
        self.max_age_days = self.config.retention_days
        self.importance_cutoff = self.config.importance_cutoff
        self.interval_seconds = self.config.interval_seconds
        self.consolidation_threshold = self.config.consolidation_threshold
        
        self.purge_enabled = self.config.purge_enabled
        self.dry_run = self.config.dry_run
        
        self._stop_event = threading.Event()
        self._thread = None
        self.last_deep_clean = None

    def start(self):
        """Starts the maintenance loop in a background daemon thread."""
        if not self.config.pipeline:
            logger.info("MesaService has no active pipeline stages. Skipping background loop.")
            return
            
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="reverie-mesa")
        self._thread.start()
        logger.info(f"MesaService started (Interval: {self.interval_seconds}s, Stages: {self.config.pipeline}, DryRun: {self.dry_run})")

    def stop(self):
        """Signals the background thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("MesaService stopped.")

    def _run_loop(self):
        """Main background loop following Activation by Inclusion."""
        while not self._stop_event.wait(self.interval_seconds):
            with tracer.start_as_current_span("reverie.mesa.maintenance_cycle") as span:
                try:
                    for stage in self.config.pipeline:
                        with tracer.start_as_current_span(f"reverie.mesa.stage.{stage}") as s_span:
                            if stage == "soft_prune":
                                self.run_soft_prune()
                            elif stage == "consolidate":
                                self.run_hierarchical_consolidation()
                            elif stage == "deep_clean":
                                if self.purge_enabled and self._should_deep_clean():
                                    self.run_deep_clean()
                except Exception as e:
                    logger.error(f"MesaService loop error: {e}")
                    span.record_exception(e)

    def _should_deep_clean(self) -> bool:
        """Triggers deep clean every 30 days."""
        if self.last_deep_clean is None:
            return True
        return (datetime.now() - self.last_deep_clean).days >= self.config.deep_clean_interval_days

    def run_soft_prune(self):
        """Identifies fragmented memories and marks them as ARCHIVED."""
        with tracer.start_as_current_span("reverie.mesa.soft_prune") as span:
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
                    SELECT source_id as node_id FROM memory_relations WHERE source_type = 'MEMORY'
                    UNION ALL
                    SELECT target_id as node_id FROM memory_relations WHERE target_type = 'MEMORY'
                ) a ON m.id = a.node_id
                WHERE m.importance_score < ?
                AND m.last_accessed_at < datetime('now', ?)
                AND m.status = 'ACTIVE'
                GROUP BY m.id
                HAVING COUNT(a.node_id) < ?
            """
            
            age_filter = f"-{self.max_age_days} days"
            cursor = self.db.get_cursor()
            with self.db.trace_query("SELECT", "memories", candidate_query, (self.importance_cutoff, age_filter, self.centrality_threshold)) as span:
                cursor.execute(candidate_query, (self.importance_cutoff, age_filter, self.centrality_threshold))
                candidate_ids = [row[0] for row in cursor.fetchall()]
            
            if not candidate_ids:
                logger.debug("MesaService: No fragmentation detected.")
                return

            if self.dry_run:
                logger.info(f"MesaService [DRY RUN]: Would archive {len(candidate_ids)} memories: {candidate_ids}")
                return

            placeholders = ",".join(["?"] * len(candidate_ids))
            update_query = f"UPDATE memories SET status = 'ARCHIVED' WHERE id IN ({placeholders})"
            with self.db.write_lock() as cursor:
                with self.db.trace_query("UPDATE", "memories", update_query, tuple(candidate_ids), batch_size=len(candidate_ids)) as span:
                    cursor.execute(update_query, tuple(candidate_ids))
            
            logger.info(f"MesaService: Archived {len(candidate_ids)} fragmented memories.")

            # Mirror-as-Code: Export to local archive
            if self.mirror:
                for mid in candidate_ids:
                    self.mirror.export_node(mid)
            
        except Exception as e:
            logger.error(f"MesaService Soft Prune failed: {e}")

    def run_hierarchical_consolidation(self):
        """Identifies clusters of stale/fragmented memories and crystallizes them into Observation Anchors."""
        with tracer.start_as_current_span("reverie.mesa.hierarchical_consolidation") as span:
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
                FROM memory_relations ma
                JOIN entities e ON ma.target_id = e.id
                JOIN memories m ON ma.source_id = m.id
                WHERE ma.source_type = 'MEMORY' 
                AND ma.target_type = 'ENTITY'
                AND m.status = 'ACTIVE'
                AND (m.last_accessed_at < datetime('now', ?) OR m.importance_score < ?)
                GROUP BY e.id
                HAVING c_count >= ?
            """
            with self.db.trace_query("SELECT", "memory_relations", query, (age_filter, self.importance_cutoff, self.consolidation_threshold)) as span:
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
                fetch_query = f"SELECT id, content_full FROM memories WHERE id IN ({placeholders})"
                with self.db.trace_query("SELECT", "memories", fetch_query, tuple(member_ids)) as span:
                    cursor.execute(fetch_query, tuple(member_ids))
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
                insert_query = """
                    INSERT INTO memories (
                        content_full, content_abstract, importance_score, memory_type, status, metadata, guid
                    ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?)
                """
                guid = str(uuid.uuid4())
                with self.db.trace_query("INSERT", "memories", insert_query, (summary_text, profile, 4.5, "OBSERVATION", metadata, guid)) as span:
                    cursor.execute(insert_query, (summary_text, profile, 4.5, "OBSERVATION", metadata, guid)) # Force high importance for anchors
                
                anchor_id = cursor.lastrowid
                
                # Vector
                vec = self.enrichment.generate_embedding(profile)
                import sqlite_vec
                insert_vec_query = "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)"
                with self.db.trace_query("INSERT", "memories_vec", insert_vec_query, (anchor_id, "BLOB")) as span:
                    cursor.execute(insert_vec_query, (anchor_id, sqlite_vec.serialize_float32(vec)))

                # 4. Link Hierarchy and Archive
                for mid in member_ids:
                    # CHILD_OF link
                    rel_query = """
                        INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type)
                        VALUES (?, 'MEMORY', ?, 'MEMORY', 'CHILD_OF')
                    """
                    params = (mid, anchor_id)
                    with self.db.trace_query("INSERT", "memory_relations", rel_query, params) as span:
                        cursor.execute(rel_query, params)
                    
                    # Also keep SUPERSEDES for backward compatibility
                    sup_query = """
                        INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type)
                        VALUES (?, 'MEMORY', ?, 'MEMORY', 'SUPERSEDES')
                    """
                    with self.db.trace_query("INSERT", "memory_relations", sup_query, (anchor_id, mid)) as span:
                        cursor.execute(sup_query, (anchor_id, mid))
                    
                    # Archive source
                    with self.db.trace_query("UPDATE", "memories", "UPDATE memories SET status = 'ARCHIVED' WHERE id = ?", (mid,)) as span:
                        cursor.execute("UPDATE memories SET status = 'ARCHIVED' WHERE id = ?", (mid,))
                    
                    # Mirror-as-Code: Export child archive
                    if self.mirror:
                        self.mirror.export_node(mid)

                # 5. Link anchor to entity
                mentions_query = """
                    INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type)
                    VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')
                """
                with self.db.trace_query("INSERT", "memory_relations", mentions_query, (anchor_id, entity_id)) as span:
                    cursor.execute(mentions_query, (anchor_id, entity_id))

            logger.info(f"MesaService: Hierarchical crystallization complete. Anchor: {anchor_id}")

            # Mirror-as-Code: Export new anchor
            if self.mirror:
                self.mirror.export_node(anchor_id)

        except Exception as e:
            logger.error(f"Hierarchical crystallization for {entity_name} failed: {e}")
            self.db.conn.rollback()

    def run_deep_clean(self):
        """Tier 2: Purges old archives and VACUUMs the database."""
        with tracer.start_as_current_span("reverie.mesa.deep_clean") as span:
            logger.info("MesaService: Running Tier 2 (Deep Clean)...")
        
        try:
            if self.dry_run:
                logger.info("MesaService [DRY RUN]: Skipping Deep Clean.")
                return

            with self.db.write_lock() as cursor:
                # 1. Delete ARCHIVED memories older than N days
                purge_days = f"-{self.config.archive_retention_days} days"
                purge_query = "DELETE FROM memories WHERE status = 'ARCHIVED' AND learned_at < datetime('now', ?)"
                with self.db.trace_query("DELETE", "memories", purge_query, (purge_days,)) as span:
                    cursor.execute(purge_query, (purge_days,))
                    purged_count = cursor.rowcount
                
                # 2. Cleanup orphaned vector entries (if any)
                cleanup_query = "DELETE FROM memories_vec WHERE rowid NOT IN (SELECT id FROM memories)"
                with self.db.trace_query("DELETE", "memories_vec", cleanup_query) as span:
                    cursor.execute(cleanup_query)
                
                logger.info(f"MesaService: Purged {purged_count} records.")

            # 3. VACUUM to reclaim space (MUST be outside transaction)
            try:
                # Using a fresh cursor directly from connection to be safe
                with self.db.trace_query("VACUUM", None, "VACUUM") as span:
                    self.db.conn.execute("VACUUM")
                logger.info("MesaService: VACUUM executed successfully.")
            except Exception as ev:
                logger.warning(f"MesaService: VACUUM failed (likely concurrent access): {ev}")
                
            self.last_deep_clean = datetime.now()
            
        except Exception as e:
            logger.error(f"MesaService Deep Clean failed: {e}")
