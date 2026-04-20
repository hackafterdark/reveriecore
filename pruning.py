import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .database import DatabaseManager
from .enrichment import EnrichmentService
from .schemas import MemoryType

logger = logging.getLogger(__name__)

class MemoryPruningService:
    """
    Background service for non-destructive memory consolidation.
    
    1. Prunes expired ephemeral memories.
    2. Consolidates fragmented memories sharing canonical entities into high-level summaries.
    3. Archives original sources with SUPERSEDES links.
    """

    def __init__(self, db: DatabaseManager, enrichment: EnrichmentService):
        self.db = db
        self.enrichment = enrichment

    def run_maintenance(self, cluster_threshold: int = 4):
        """主入口: 执行清理和整合."""
        logger.info("Running ReverieCore Memory Maintenance...")
        self.prune_expired()
        self.consolidate_by_entities(threshold=cluster_threshold)

    def prune_expired(self):
        """Truly DELETEs memories where expires_at is in the past."""
        try:
            with self.db.write_lock() as cursor:
                cursor.execute("DELETE FROM memories WHERE expires_at < CURRENT_TIMESTAMP")
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Pruned {deleted_count} expired ephemeral memories.")
        except Exception as e:
            logger.error(f"Expiration pruning failed: {e}")

    def consolidate_by_entities(self, threshold: int = 4):
        """Finds clusters of memories sharing the same entity and summarizes them."""
        try:
            cursor = self.db.get_cursor()
            
            # Find entities mentioned in >= threshold ACTIVE memories that HAVE an expiration (Transient)
            # Permanent memories (expires_at IS NULL) are skipped for now per user request.
            query = """
                SELECT e.id, e.name, COUNT(ma.source_id) as c_count, GROUP_CONCAT(ma.source_id) as member_ids
                FROM memory_associations ma
                JOIN entities e ON ma.target_id = e.id
                JOIN memories m ON ma.source_id = m.id
                WHERE ma.source_type = 'MEMORY' 
                AND ma.target_type = 'ENTITY'
                AND m.status = 'ACTIVE'
                AND m.expires_at IS NOT NULL
                GROUP BY e.id
                HAVING c_count >= ?
            """
            cursor.execute(query, (threshold,))
            clusters = cursor.fetchall()

            for ent_id, ent_name, count, member_ids_str in clusters:
                member_ids = [int(i) for i in member_ids_str.split(',')]
                self._consolidate_cluster(member_ids, ent_name, ent_id)

        except Exception as e:
            logger.error(f"Clustered consolidation failed: {e}")

    def _consolidate_cluster(self, member_ids: List[int], entity_name: str, entity_id: int):
        """Merges a specific set of IDs into a summary memory."""
        logger.info(f"Consolidating cluster for entity '{entity_name}' ({len(member_ids)} memories)...")
        
        try:
            # 0. Fetch content
            texts = []
            with self.db.write_lock() as cursor:
                placeholders = ",".join(["?"] * len(member_ids))
                cursor.execute(f"SELECT content_full FROM memories WHERE id IN ({placeholders})", tuple(member_ids))
                texts = [row[0] for row in cursor.fetchall()]
                
            if not texts:
                logger.warning(f"No texts found for cluster {member_ids}")
                return

            # 1. Synthesize
            summary_text = self.enrichment.synthesize_memories(texts, entity_name)
            
            # 2. Extract Profile & Importance for the new record
            profile = self.enrichment.generate_semantic_profile(summary_text)
            imp_data = self.enrichment.calculate_importance(summary_text)
            importance = imp_data["score"]
            expires_at = imp_data["expires_at"]
            vec = self.enrichment.generate_embedding(profile)
            
            with self.db.write_lock() as cursor:
                # 3. Save Summary
                cursor.execute("""
                    INSERT INTO memories (
                        content_full, content_abstract, importance_score, expires_at, memory_type, status, metadata
                    ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?)
                """, (summary_text, profile, importance, expires_at, MemoryType.OBSERVATION.value, f'{"consolidated_from": {member_ids}}'))
                
                summary_id = cursor.lastrowid
                
                # Vector
                import sqlite_vec
                cursor.execute("INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)", 
                            (summary_id, sqlite_vec.serialize_float32(vec)))

                # 4. Create SUPERSEDES links and Archive sources
                for mid in member_ids:
                    # Link
                    cursor.execute("""
                        INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type)
                        VALUES (?, 'MEMORY', ?, 'MEMORY', 'SUPERSEDES')
                    """, (summary_id, mid))
                    
                    # Archive
                    cursor.execute("UPDATE memories SET status = 'ARCHIVED' WHERE id = ?", (mid,))

                # 5. Link new summary to the entity anchor
                cursor.execute("""
                    INSERT INTO memory_associations (source_id, source_type, target_id, target_type, association_type)
                    VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS')
                """, (summary_id, entity_id))

            logger.info(f"Consolidation complete. Created summary {summary_id}, Archived {len(member_ids)} memories.")

        except Exception as e:
            logger.error(f"Cluster consolidation for {entity_name} failed: {e}")
            self.db.conn.rollback()

    def unarchive_memories(self, memory_ids: List[int]):
        """Restores a list of archived memories back to ACTIVE status."""
        try:
            cursor = self.db.get_cursor()
            placeholders = ",".join(["?"] * len(memory_ids))
            cursor.execute(f"UPDATE memories SET status = 'ACTIVE' WHERE id IN ({placeholders})", tuple(memory_ids))
            self.db.commit()
            logger.info(f"Unarchived {len(memory_ids)} memories.")
        except Exception as e:
            logger.error(f"Unarchiving failed: {e}")
