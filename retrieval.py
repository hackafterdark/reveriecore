import logging
from typing import List, Dict, Any
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class Retriever:
    """RAG Engine: Handles vector search and intelligence-based ranking."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db

    def search(self, query_vector: List[float], limit: int = 5, similarity_weight: float = 0.7, importance_weight: float = 0.3, allowed_owners: List[str] = None) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search: Vector Similarity + Importance Re-ranking.
        
        Filters by owner (if allowed_owners provided) and privacy='PUBLIC'.
        """
        cursor = self.db.get_cursor()
        
        # 1. Fetch top K*2 candidates using vector search
        candidate_limit = limit * 3
        
        try:
            import sqlite_vec
            
            # Build filter clause
            filter_clause = ""
            params = [sqlite_vec.serialize_float32(query_vector), candidate_limit]
            
            if allowed_owners:
                placeholders = ",".join(["?"] * len(allowed_owners))
                filter_clause = f"AND (m.owner_id IN ({placeholders}) OR m.privacy = 'PUBLIC')"
                params.extend(allowed_owners)
            
            query = f"""
                SELECT 
                    m.id, 
                    m.content_full, 
                    m.content_abstract,
                    m.token_count_full,
                    m.token_count_abstract,
                    m.importance_score, 
                    v.distance
                FROM memories_vec v
                JOIN memories m ON v.rowid = m.id
                WHERE v.embedding MATCH ? AND v.k = ?
                {filter_clause}
                ORDER BY v.distance ASC
            """
            
            cursor.execute(query, params)
            
            rows = cursor.fetchall()
            if not rows:
                return []

            # 2. Re-rank based on importance
            results = []
            for row in rows:
                mem_id, content_full, content_abstract, tc_full, tc_abstract, importance, distance = row
                
                # Convert distance to similarity
                similarity = 1.0 / (1.0 + distance) 
                
                # Normalize importance
                norm_importance = min(importance / 5.0, 1.0)
                
                final_score = (similarity * similarity_weight) + (norm_importance * importance_weight)
                
                results.append({
                    "id": mem_id,
                    "content_full": content_full,
                    "content_abstract": content_abstract,
                    "token_count_full": tc_full,
                    "token_count_abstract": tc_abstract,
                    "score": final_score,
                    "importance": importance,
                    "similarity": similarity
                })

            # Sort by final score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results[:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
