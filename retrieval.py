import logging
from typing import List, Dict, Any
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class Retriever:
    """RAG Engine: Handles vector search and intelligence-based ranking."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db

    def search(self, 
               query_vector: List[float], 
               limit: int = 5, 
               token_budget: int = 1000,
               strategy: str = "balanced",
               similarity_weight: float = 0.7, 
               importance_weight: float = 0.3, 
               allowed_owners: List[str] = None) -> List[Dict[str, Any]]:
        """
        Performs a hybrid search: Vector Similarity + Importance Re-ranking.
        
        Optimizes for context window usage by choosing between full content 
        and abstracts based on the provided token_budget.
        """
        cursor = self.db.get_cursor()
        
        # 1. Fetch top K*3 candidates using vector search
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
            candidates = []
            for row in rows:
                mem_id, content_full, content_abstract, tc_full, tc_abstract, importance, distance = row
                
                # Convert distance to similarity
                similarity = 1.0 / (1.0 + distance) 
                
                # Normalize importance
                norm_importance = min(importance / 5.0, 1.0)
                
                final_score = (similarity * similarity_weight) + (norm_importance * importance_weight)
                
                candidates.append({
                    "id": mem_id,
                    "content_full": content_full,
                    "content_abstract": content_abstract,
                    "tc_full": tc_full or (len(content_full) // 4),
                    "tc_abstract": tc_abstract or (len(content_abstract or "") // 4),
                    "score": final_score,
                    "importance": importance
                })

            # Sort by final score descending
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # 3. Budget-Aware Selection
            results = []
            current_tokens = 0
            
            for c in candidates:
                if len(results) >= limit:
                    break
                    
                chosen_content = None
                chosen_tokens = 0
                version = "full"
                
                # Decision logic
                if strategy == "abstract_only" and c["content_abstract"]:
                    chosen_content = c["content_abstract"]
                    chosen_tokens = c["tc_abstract"]
                    version = "abstract"
                else:
                    # Try full first
                    if current_tokens + c["tc_full"] <= token_budget:
                        chosen_content = c["content_full"]
                        chosen_tokens = c["tc_full"]
                        version = "full"
                    # Fallback to abstract if full is too big
                    elif c["content_abstract"] and current_tokens + c["tc_abstract"] <= token_budget:
                        chosen_content = c["content_abstract"]
                        chosen_tokens = c["tc_abstract"]
                        version = "abstract"
                
                if chosen_content:
                    results.append({
                        "id": c["id"],
                        "content": chosen_content,
                        "tokens": chosen_tokens,
                        "version": version,
                        "score": c["score"]
                    })
                    current_tokens += chosen_tokens
            
            logger.info(f"Retrieved {len(results)} memories using {current_tokens}/{token_budget} tokens (Strategy: {strategy})")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
