import math
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
from .database import DatabaseManager
from .graph_query import GraphQueryService

logger = logging.getLogger(__name__)

class Retriever:
    """RAG Engine: Handles vector search, importance-based ranking, and graph traversal."""
    
    def __init__(self, db: DatabaseManager, enrichment: Any = None):
        self.db = db
        self.graph = GraphQueryService(db)
        self.enrichment = enrichment

    def _calculate_decay(self, learned_at_str: str, importance: float, expires_at: Optional[str] = None) -> float:
        """
        Calculates time decay score. 
        - Permanent memories (expires_at is NULL) have NO decay (1.0).
        - High importance memories (>= 4.0) have permanent weight (1.0).
        - Others follow a 48-hour half-life exponential decay.
        """
        if not expires_at or importance >= 4.0:
            return 1.0
        
        try:
            learned_at = datetime.fromisoformat(learned_at_str.replace("Z", "+00:00"))
            now = datetime.utcnow()
            age_hours = (now - learned_at).total_seconds() / 3600.0
            
            decay = math.pow(0.5, age_hours / 48.0)
            return max(0.1, decay)
        except Exception as e:
            logger.debug(f"Decay calculation failed: {e}")
            return 1.0

    def _calculate_gravity(self, query: str, primary_candidates: List[Dict[str, Any]]) -> float:
        """
        Calculates dynamic gravity based on query intent and result context.
        - High Precision Intent: Gravity 1.0 (Anchored)
        - Synthesis Intent: Gravity 0.8 (Balanced)
        - Exploration Intent: Gravity 0.5 (Discoverable)
        - Knowledge Boost: +0.1 if results contain code/architecture.
        """
        if not self.enrichment:
            return 1.0
            
        intent_scores = self.enrichment.classify_intent(query)
        
        # Normalize scores so they sum to 1.0
        total_score = sum(intent_scores.values())
        if total_score > 0:
            for k in intent_scores:
                intent_scores[k] /= total_score
        
        # Weighted average
        base_gravity = (
            (intent_scores.get('retrieving specific facts or entities', 0) * 1.0) +
            (intent_scores.get('synthesizing related information', 0) * 0.8) +
            (intent_scores.get('exploring open-ended possibilities', 0) * 0.5)
        )
        
        # Technical boost
        boost = 0.0
        tech_markers = ["class ", "def ", "struct ", "interface ", "schema", "architecture", "diagram"]
        for c in primary_candidates:
            content = c.get("content_full", "").lower()
            if "```" in content or any(m in content for m in tech_markers):
                boost = 0.1
                break
                
        return min(1.1, base_gravity + boost)

    def search(self, 
               query_vector: List[float], 
               query_text: str = "",
               limit: int = 5, 
               token_budget: int = 1000,
               strategy: str = "balanced",
               similarity_weight: float = 0.5, 
               importance_weight: float = 0.3,
               decay_weight: float = 0.2,
               allowed_owners: List[str] = None,
               include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        Advanced Anchor-Aware Search:
        1. Freshness Check: Bypass graph if 'clean slate' requested.
        2. Semantic Anchoring: Graph-first discovery from query entities.
        3. Broad Fallback: Trigger vector search if graph results < 3.
        4. Re-ranking: Similarity + Importance + Temporal Decay.
        """
        cursor = self.db.get_cursor()
        candidates = []
        seen_ids = set()
        
        status_filter = "('ACTIVE')" if not include_archived else "('ACTIVE', 'ARCHIVED')"
        
        # Detect if 'clean slate' requested (Bypass anchors/gravity)
        keywords = ["clean slate", "new idea", "fresh start", "forget history", "new project"]
        is_fresh = any(k in query_text.lower() for k in keywords)
        
        anchors = []
        if not is_fresh and self.enrichment:
            anchors = self.enrichment.extract_query_anchors(query_text)

        # A. Semantic Anchoring (Graph-First)
        graph_anchored_ids = []
        if anchors:
            graph_anchored_ids = self.graph.get_memories_by_entities(anchors)
            if graph_anchored_ids:
                id_placeholders = ",".join(["?"] * len(graph_anchored_ids))
                query = f"SELECT id, content_full, content_abstract, token_count_full, token_count_abstract, importance_score, learned_at, expires_at FROM memories WHERE id IN ({id_placeholders}) AND status IN {status_filter}"
                cursor.execute(query, tuple(graph_anchored_ids))
                for row in cursor.fetchall():
                    m_id, c_f, c_a, tc_f, tc_a, imp, lat, exp = row
                    decay = self._calculate_decay(lat, imp, exp)
                    
                    # Graph anchors get a boost
                    score = (0.6 * similarity_weight) + (min(imp / 5.0, 1.0) * importance_weight) + (decay * decay_weight) + 0.2
                    
                    candidates.append({
                        "id": m_id, "content_full": c_f, "content_abstract": c_a,
                        "tc_full": tc_f or (len(c_f) // 4), "tc_abstract": tc_a or (len(c_a or "") // 4),
                        "score": score, "importance": imp, "learned_at": lat, "source": "anchor"
                    })
                    seen_ids.add(m_id)

        # B. Vector Fallback (Broad Search)
        # Triggered if freshness is requested OR we have fewer than 3 graph results
        if is_fresh or len(candidates) < 3:
            candidate_limit = limit * 3
            try:
                import sqlite_vec
                filter_clause = ""
                v_params = [sqlite_vec.serialize_float32(query_vector), candidate_limit]
                
                if allowed_owners:
                    placeholders = ",".join(["?"] * len(allowed_owners))
                    filter_clause = f"AND (m.owner_id IN ({placeholders}) OR m.privacy = 'PUBLIC')"
                    v_params.extend(allowed_owners)
                
                v_query = f"""
                    SELECT m.id, m.content_full, m.content_abstract, m.token_count_full, m.token_count_abstract, m.importance_score, m.learned_at, m.expires_at, v.distance
                    FROM memories_vec v JOIN memories m ON v.rowid = m.id
                    WHERE v.embedding MATCH ? AND v.k = ? AND m.status IN {status_filter} {filter_clause}
                    ORDER BY v.distance ASC
                """
                cursor.execute(v_query, v_params)
                for row in cursor.fetchall():
                    m_id, c_f, c_a, tc_f, tc_a, imp, lat, exp, dist = row
                    if m_id in seen_ids: continue
                    
                    similarity = 1.0 / (1.0 + dist)
                    decay = 1.0 if is_fresh else self._calculate_decay(lat, imp, exp)
                    
                    final_score = (similarity * similarity_weight) + (min(imp / 5.0, 1.0) * importance_weight) + (decay * decay_weight)
                    
                    candidates.append({
                        "id": m_id, "content_full": c_f, "content_abstract": c_a,
                        "tc_full": tc_f or (len(c_f) // 4), "tc_abstract": tc_a or (len(c_a or "") // 4),
                        "score": final_score, "importance": imp, "learned_at": lat, "source": "vector"
                    })
                    seen_ids.add(m_id)
            except Exception as e:
                logger.error(f"Vector fallback failed: {e}")

        # C. Graph Augmentation (Connections from top results)
        if not is_fresh:
            seed_ids = [c["id"] for c in sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]]
            if seed_ids:
                # Calculate dynamic gravity before expansion
                gravity = self._calculate_gravity(query_text, candidates[:5])
                linked_ids = self.graph.get_related_memories(seed_ids, anchor_entities=anchors, gravity=gravity)
                new_ids = [i for i in linked_ids if i not in seen_ids]
                if new_ids:
                    id_placeholders = ",".join(["?"] * len(new_ids))
                    fetch_query = f"SELECT id, content_full, content_abstract, token_count_full, token_count_abstract, importance_score, learned_at, expires_at FROM memories WHERE id IN ({id_placeholders}) AND status IN {status_filter}"
                    cursor.execute(fetch_query, tuple(new_ids))
                    for row in cursor.fetchall():
                        m_id, c_f, c_a, tc_f, tc_a, imp, lat, exp = row
                        decay = self._calculate_decay(lat, imp, exp)
                        score = (0.4 * similarity_weight) + (min(imp / 5.0, 1.0) * importance_weight) + (decay * decay_weight)
                        candidates.append({
                            "id": m_id, "content_full": c_f, "content_abstract": c_a,
                            "tc_full": tc_f or (len(c_f) // 4), "tc_abstract": tc_a or (len(c_a or "") // 4),
                            "score": score, "importance": imp, "learned_at": lat, "source": "graph"
                        })
                        seen_ids.add(m_id)

        # D. Budget-Aware Selection (Same as before)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = []
        current_tokens = 0
        for c in candidates:
            if len(results) >= limit: break
            chosen_content = None
            chosen_tokens = 0
            version = "full"
            
            if strategy == "abstract_only" and c["content_abstract"]:
                chosen_content = c["content_abstract"]; chosen_tokens = c["tc_abstract"]; version = "abstract"
            else:
                if current_tokens + c["tc_full"] <= token_budget:
                    chosen_content = c["content_full"]; chosen_tokens = c["tc_full"]; version = "full"
                elif c["content_abstract"] and current_tokens + c["tc_abstract"] <= token_budget:
                    chosen_content = c["content_abstract"]; chosen_tokens = c["tc_abstract"]; version = "abstract"
            
            if chosen_content:
                display_content = chosen_content
                if c["source"] in ["graph", "anchor"]:
                    summary = self.graph.get_neighbors_summary(c["id"])
                    display_content += summary
                results.append({"id": c["id"], "content": display_content, "tokens": chosen_tokens, "version": version, "score": c["score"]})
                current_tokens += chosen_tokens
        
        logger.info(f"Retrieved {len(results)} memories ({current_tokens}/{token_budget} tokens). Strategy: {strategy}, IsFresh: {is_fresh}")
        return results
