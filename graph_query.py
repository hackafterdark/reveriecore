import logging
import sqlite3
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

class GraphQueryService:
    """Encapsulates complex Recursive CTE logic for graph traversal."""
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_related_memories(self, start_memory_ids: List[int], depth: int = 2, per_node_limit: int = 10) -> List[int]:
        """
        Traverses the graph to find memories linked to the start nodes.
        Path: Memory <-> Entity <-> Entity <-> Memory
        
        Uses a step-by-step Python traversal with SQL neighbor lookups 
        to ensure 'Hub Protection' can be reliably enforced.
        """
        if not start_memory_ids:
            return []
            
        cursor = self.db.get_cursor()
        
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
                neighbors_query = """
                    SELECT 
                        CASE WHEN source_id = ? AND source_type = ? THEN target_id ELSE source_id END as next_id,
                        CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END as next_type,
                        confidence
                    FROM memory_associations
                    WHERE (source_id = ? AND source_type = ?) OR (target_id = ? AND target_type = ?)
                    ORDER BY confidence DESC, rowid ASC
                    LIMIT ?
                """
                cursor.execute(neighbors_query, (node_id, node_type, node_id, node_type, node_id, node_type, node_id, node_type, per_node_limit))
                
                for next_id, next_type, _ in cursor.fetchall():
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
