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
            
            # Batch process current_layer to respect SQLite parameter limits
            BATCH_SIZE = 400 
            for i in range(0, len(current_layer), BATCH_SIZE):
                batch = current_layer[i:i+BATCH_SIZE]
                
                # Build current_layer VALUES clause
                values_placeholders = ",".join(["(?, ?)"] * len(batch))
                values_params = []
                for nid, ntype in batch:
                    values_params.extend([nid, ntype])
                
                anchor_list = list(anchor_ids) if anchor_ids else [-1]
                anchor_placeholders = ",".join(["?"] * len(anchor_list))
                
                # Bulk Expansion Query:
                # 1. Identifies all neighbors (forward and backward edges) for the batch
                # 2. Ranks neighbors per source node using ROW_NUMBER()
                # 3. Applies the per_node_limit in-database to manage memory
                bulk_query = f"""
                    WITH current_layer_nodes(node_id, node_type) AS (
                        VALUES {values_placeholders}
                    ),
                    candidates AS (
                        -- Forward edges: current_layer is source
                        SELECT 
                            r.target_id as next_id, r.target_type as next_type, r.confidence_score, r.id as rel_id,
                            r.source_id, r.source_type
                        FROM memory_relations r
                        JOIN current_layer_nodes cl ON r.source_id = cl.node_id AND r.source_type = cl.node_type
                        UNION ALL
                        -- Backward edges: current_layer is target
                        SELECT 
                            r.source_id as next_id, r.source_type as next_type, r.confidence_score, r.id as rel_id,
                            r.target_id as source_id, r.target_type as source_type
                        FROM memory_relations r
                        JOIN current_layer_nodes cl ON r.target_id = cl.node_id AND r.target_type = cl.node_type
                    ),
                    scored AS (
                        SELECT 
                            next_id, next_type, confidence_score, rel_id, source_id, source_type,
                            CASE WHEN next_type = 'ENTITY' AND next_id IN ({anchor_placeholders}) THEN 1 ELSE 0 END as is_anchor
                        FROM candidates
                    ),
                    ranked AS (
                        SELECT 
                            next_id, next_type, 
                            (confidence_score * (1 + (is_anchor * ?))) as d_score,
                            ROW_NUMBER() OVER (
                                PARTITION BY source_id, source_type 
                                ORDER BY 
                                    (CASE WHEN next_type = 'ENTITY' THEN 1 ELSE 0 END) DESC, 
                                    (confidence_score * (1 + (is_anchor * ?))) DESC, 
                                    rel_id ASC
                            ) as rn
                        FROM scored
                    )
                    SELECT next_id, next_type, d_score 
                    FROM ranked 
                    WHERE rn <= ?
                    ORDER BY d_score DESC
                """
                
                # Parameters: values_params, anchor_list (x1), gravity (x2), per_node_limit
                params = values_params + anchor_list + [gravity, gravity, per_node_limit]
                
                with tracer.start_as_current_span("reverie.graph.sql_query") as span:
                    span.set_attribute("db.statement", "reverie.graph.bulk_expansion")
                    span.set_attribute("graph.batch_size", len(batch))
                    cursor.execute(bulk_query, tuple(params))
                    rows = cursor.fetchall()
                
                for next_id, next_type, d_score in rows:
                    if (next_id, next_type) not in visited:
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

    def get_neighbors_summaries(self, memory_ids: List[int]) -> Dict[int, str]:
        """Returns a mapping of memory_id -> neighbors summary string."""
        if not memory_ids:
            return {}
            
        with tracer.start_as_current_span("reverie.graph.batch_summary") as span:
            cursor = self.db.get_cursor()
            placeholders = ",".join(["?"] * len(memory_ids))
            query = f"""
                SELECT ma.source_id, e.name, e.label, ma.relation_type 
                FROM memory_relations ma
                JOIN entities e ON ma.target_id = e.id AND ma.target_type = 'ENTITY'
                WHERE ma.source_id IN ({placeholders}) AND ma.source_type = 'MEMORY'
            """
            
            try:
                with tracer.start_as_current_span("reverie.graph.sql_query") as sql_span:
                    sql_span.set_attribute("db.statement", "reverie.graph.batch_summary_query")
                    cursor.execute(query, tuple(memory_ids))
                    rows = cursor.fetchall()
                
                # Group by source_id
                grouped = {}
                for mid, name, label, rel_type in rows:
                    link = f"[{label}: {name} ({rel_type})]"
                    grouped.setdefault(mid, []).append(link)
                
                # Format into strings
                return {mid: " Linked Entities: " + ", ".join(links) for mid, links in grouped.items()}
            except Exception as e:
                logger.debug(f"Failed to fetch batch neighbors summary: {e}")
                return {}
