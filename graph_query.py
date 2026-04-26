import logging
import sqlite3
from typing import List, Dict, Any, Set
from opentelemetry import trace
from .telemetry import get_tracer

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)

class GraphQueryService:
    """Encapsulates complex Recursive CTE logic for graph traversal."""
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_related_memories(self, start_memory_ids: List[int], anchor_entities: List[str] = None, gravity: float = 1.0, depth: int = 2, per_node_limit: int = 10) -> Dict[int, float]:
        """
        Traverses the graph to find memories linked to the start nodes.
        Path: Memory <-> Entity <-> Entity <-> Memory
        
        Returns: Dict[memory_id, max_discovery_score]
        """
        with tracer.start_as_current_span("reverie.graph.traversal") as span:
            span.set_attribute("graph.start_nodes", len(start_memory_ids))
            span.set_attribute("graph.depth", depth)
            if not start_memory_ids:
                return {}
            
        cursor = self.db.get_cursor()

        # Resolve anchor entities to IDs
        anchor_ids = set()
        if anchor_entities:
            placeholders = ",".join(["?"] * len(anchor_entities))
            query = f"SELECT id FROM entities WHERE name IN ({placeholders})"
            with tracer.start_as_current_span("reverie.graph.sql_query") as span:
                span.set_attribute("db.statement", query)
                cursor.execute(query, tuple(anchor_entities))
                anchor_ids = {row[0] for row in cursor.fetchall()}
        
        # Track state: (node_id, node_type) -> max_score
        visited = {}
        for mid in start_memory_ids:
            visited[(mid, 'MEMORY')] = 1.0 # Seeds start with max confidence
            
        current_layer = []
        for mid in start_memory_ids:
            current_layer.append((mid, 'MEMORY'))
            
        # All found memories: id -> score
        found_memories = {}
        
        for level in range(depth):
            if not current_layer:
                break
                
            next_layer = []
            
            for node_id, node_type in current_layer:
                # anchor_id_placeholders for the CASE statement
                anchor_list = list(anchor_ids) if anchor_ids else [-1]
                placeholders = ",".join(["?"] * len(anchor_list))

                # We select discovery_score to propagate confidence
                neighbors_query = f"""
                    SELECT 
                        CASE WHEN source_id = ? AND source_type = ? THEN target_id ELSE source_id END as next_id,
                        CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END as next_type,
                        confidence_score,
                        CASE WHEN (CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END) = 'ENTITY' THEN 1 ELSE 0 END as type_weight,
                        CASE WHEN (CASE WHEN source_id = ? AND source_type = ? THEN target_id ELSE source_id END) IN ({placeholders}) 
                             AND (CASE WHEN source_id = ? AND source_type = ? THEN target_type ELSE source_type END) = 'ENTITY'
                             THEN 1 ELSE 0 END as is_anchor,
                        id
                    FROM memory_relations
                    WHERE (source_id = ? AND source_type = ?) OR (target_id = ? AND target_type = ?)
                """
                neighbors_query = f"SELECT next_id, next_type, confidence_score, type_weight, (confidence_score * (1 + (is_anchor * ?))) as discovery_score, id FROM ({neighbors_query}) ORDER BY type_weight DESC, discovery_score DESC, id ASC LIMIT ?"
                
                # Parameter ordering: 1 (gravity), then inner query params, then 1 (limit)
                params = [gravity] 
                # Inner SELECT params (10 + len(anchor_list))
                params.extend([
                    node_id, node_type, node_id, node_type, # next_id, next_type
                    node_id, node_type,                     # type_weight
                    node_id, node_type,                     # is_anchor part 1
                ])
                params.extend(anchor_list)
                params.extend([node_id, node_type])         # is_anchor part 2
                
                # WHERE clause params (4)
                params.extend([node_id, node_type, node_id, node_type])
                
                # Final LIMIT param (1)
                params.append(per_node_limit)

                with tracer.start_as_current_span("reverie.graph.sql_query") as span:
                    span.set_attribute("db.statement", neighbors_query)
                    cursor.execute(neighbors_query, tuple(params))
                    rows = cursor.fetchall()
                
                for next_id, next_type, _, _, d_score, _ in rows:
                    # If not visited OR we found a higher-score path
                    if (next_id, next_type) not in visited or d_score > visited[(next_id, next_type)]:

                        visited[(next_id, next_type)] = d_score
                        next_layer.append((next_id, next_type))
                        if next_type == 'MEMORY':
                            found_memories[next_id] = max(found_memories.get(next_id, 0), d_score)
                            
            current_layer = next_layer
            # Global cap
            if len(found_memories) >= 50:
                break
                
        return found_memories

    def get_memories_by_entities(self, entity_names: List[str]) -> List[int]:
        """Finds memories that are directly linked to any of the given entity names."""
        with tracer.start_as_current_span("reverie.graph.entity_lookup") as span:
            span.set_attribute("graph.entity_count", len(entity_names))
            if not entity_names: return []
        cursor = self.db.get_cursor()
        placeholders = ','.join(['?'] * len(entity_names))
        
        query = f"""
            SELECT DISTINCT source_id 
            FROM memory_relations 
            WHERE source_type = 'MEMORY' 
            AND target_type = 'ENTITY'
            AND target_id IN (SELECT id FROM entities WHERE name IN ({placeholders}))
            UNION
            SELECT DISTINCT target_id
            FROM memory_relations
            WHERE target_type = 'MEMORY'
            AND source_type = 'ENTITY'
            AND source_id IN (SELECT id FROM entities WHERE name IN ({placeholders}))
        """
        with tracer.start_as_current_span("reverie.graph.sql_query") as span:
            span.set_attribute("db.statement", query)
            cursor.execute(query, entity_names + entity_names)
            return [row[0] for row in cursor.fetchall()]

    def get_neighbors_summary(self, memory_id: int) -> str:
        """Returns a string summary of entities linked to a memory for context injection."""
        cursor = self.db.get_cursor()
        query = """
            SELECT e.name, e.label, ma.relation_type 
            FROM memory_relations ma
            JOIN entities e ON ma.target_id = e.id AND ma.target_type = 'ENTITY'
            WHERE ma.source_id = ? AND ma.source_type = 'MEMORY'
        """
        try:
            with tracer.start_as_current_span("reverie.graph.sql_query") as span:
                span.set_attribute("db.statement", query)
                cursor.execute(query, (memory_id,))
                rows = cursor.fetchall()
            if not rows:
                return ""
            
            links = [f"[{r[1]}: {r[0]} ({r[2]})]" for r in rows]
            return " Linked Entities: " + ", ".join(links)
        except Exception as e:
            logger.debug(f"Failed to fetch neighbors summary: {e}")
            return ""
