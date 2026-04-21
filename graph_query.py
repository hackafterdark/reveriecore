import logging
import sqlite3
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

class GraphQueryService:
    """Encapsulates complex Recursive CTE logic for graph traversal."""
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_related_memories(self, start_memory_ids: List[int], anchor_entities: List[str] = None, gravity: float = 1.0, depth: int = 2, per_node_limit: int = 10) -> List[int]:
        """
        Traverses the graph to find memories linked to the start nodes.
        Path: Memory <-> Entity <-> Entity <-> Memory
        
        Uses a step-by-step Python traversal with SQL neighbor lookups 
        to ensure 'Hub Protection' can be reliably enforced.

        Gravity: A multiplier for associations connected to query anchors.
        """
        if not start_memory_ids:
            return []
            
        cursor = self.db.get_cursor()

        # Resolve anchor entities to IDs
        anchor_ids = set()
        if anchor_entities:
            placeholders = ",".join(["?"] * len(anchor_entities))
            cursor.execute(f"SELECT id FROM entities WHERE name IN ({placeholders})", tuple(anchor_entities))
            anchor_ids = {row[0] for row in cursor.fetchall()}
        
        # Track state
        visited = set()
        for mid in start_memory_ids:
            visited.add((mid, 'MEMORY'))
            
        current_layer = []
        for mid in start_memory_ids:
            current_layer.append((mid, 'MEMORY'))
            
        # All found memories (excluding seeds)
        found_memories = set()
        
        for level in range(depth):
            if not current_layer:
                break
                
            next_layer = []
            
            # Process current layer in batches for efficiency (optional, but good)
            # For simplicity, we'll do one query for all nodes in the current layer
            for node_id, node_type in current_layer:
                # anchor_id_placeholders for the CASE statement
                anchor_list = list(anchor_ids) if anchor_ids else [-1]
                placeholders = ",".join(["?"] * len(anchor_list))

                neighbors_query = f"""
                    SELECT 
                        CASE WHEN source_id = ? AND source_type = ? THEN target_id ELSE source_id END as next_id,
                        CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END as next_type,
                        confidence_score,
                        -- Prioritize ENTITY types over MEMORY types for bridging logic
                        CASE WHEN (CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END) = 'ENTITY' THEN 1 ELSE 0 END as type_weight,
                        -- Anchor check
                        CASE WHEN (CASE WHEN source_id = ? AND source_type = ? THEN target_id ELSE source_id END) IN ({placeholders}) 
                             AND (CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END) = 'ENTITY'
                             THEN 1 ELSE 0 END as is_anchor,
                        rowid
                    FROM memory_associations
                    WHERE (source_id = ? AND source_type = ?) OR (target_id = ? AND target_type = ?)
                """
                # Combined score: confidence * (1 + (is_anchor * gravity))
                # We do this in Python or via a slightly more complex SQL. 
                # Let's do it in SQL:
                neighbors_query = f"SELECT next_id, next_type, confidence_score, type_weight, (confidence_score * (1 + (is_anchor * ?))) as discovery_score, rowid FROM ({neighbors_query}) ORDER BY type_weight DESC, discovery_score DESC, rowid ASC LIMIT ?"
                
                params = [
                    node_id, node_type, node_id, node_type, # for next_id/next_type
                    node_id, node_type, # for type_weight
                    node_id, node_type, node_id, node_type, # for is_anchor
                ]
                params.extend(anchor_list) # for the IN clause
                params.extend([node_id, node_type, node_id, node_type]) # for WHERE clause
                params.extend([gravity, per_node_limit]) # for final sort/limit

                cursor.execute(neighbors_query, tuple(params))
                
                for next_id, next_type, _, _, _ in cursor.fetchall():
                    if (next_id, next_type) not in visited:
                        visited.add((next_id, next_type))
                        next_layer.append((next_id, next_type))
                        if next_type == 'MEMORY':
                            found_memories.add(next_id)
                            
            current_layer = next_layer
            # Global cap to prevent extreme growth
            if len(found_memories) >= 50:
                break
                
        return list(found_memories)

    def get_memories_by_entities(self, entity_names: List[str]) -> List[int]:
        """Finds memories that are directly linked to any of the given entity names."""
        if not entity_names: return []
        cursor = self.db.get_cursor()
        placeholders = ','.join(['?'] * len(entity_names))
        
        query = f"""
            SELECT DISTINCT source_id 
            FROM memory_associations 
            WHERE source_type = 'MEMORY' 
            AND target_type = 'ENTITY'
            AND target_id IN (SELECT id FROM entities WHERE name IN ({placeholders}))
            UNION
            SELECT DISTINCT target_id
            FROM memory_associations
            WHERE target_type = 'MEMORY'
            AND source_type = 'ENTITY'
            AND source_id IN (SELECT id FROM entities WHERE name IN ({placeholders}))
        """
        cursor.execute(query, entity_names + entity_names)
        return [row[0] for row in cursor.fetchall()]

    def get_neighbors_summary(self, memory_id: int) -> str:
        """Returns a string summary of entities linked to a memory for context injection."""
        cursor = self.db.get_cursor()
        query = """
            SELECT e.name, e.label, ma.association_type 
            FROM memory_associations ma
            JOIN entities e ON ma.target_id = e.id AND ma.target_type = 'ENTITY'
            WHERE ma.source_id = ? AND ma.source_type = 'MEMORY'
        """
        try:
            cursor.execute(query, (memory_id,))
            rows = cursor.fetchall()
            if not rows:
                return ""
            
            links = [f"[{r[1]}: {r[0]} ({r[2]})]" for r in rows]
            return " Linked Entities: " + ", ".join(links)
        except Exception as e:
            logger.debug(f"Failed to fetch neighbors summary: {e}")
            return ""
