import math
import json
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod
from .database import DatabaseManager
from .graph_query import GraphQueryService
from .config import load_reverie_config
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from .telemetry import get_tracer
from .reranking import RerankerHandler

tracer = get_tracer(__name__)


logger = logging.getLogger(__name__)
from .retrieval_base import RetrievalContext, RetrievalHandler

class AnchoringDiscovery(RetrievalHandler):
    """Stage A: Semantic Anchoring (Graph-First)"""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        # Detect if 'clean slate' requested
        keywords = ["clean slate", "new idea", "fresh start", "forget history", "new project"]
        context.is_fresh = any(k in context.query_text.lower() for k in keywords)
        
        if context.is_fresh:
            return

        if retriever.enrichment:
            context.anchors = retriever.enrichment.extract_query_anchors(context.query_text)
            
        if not context.anchors:
            return

        cursor = retriever.db.get_cursor()
        graph_anchored_ids = retriever.graph.get_memories_by_entities(context.anchors)
        
        include_archived = context.config.get("include_archived", False)
        status_filter = "('ACTIVE')" if not include_archived else "('ACTIVE', 'ARCHIVED')"
        
        if graph_anchored_ids:
            id_placeholders = ",".join(["?"] * len(graph_anchored_ids))
            query = f"SELECT id, content_full, content_abstract, token_count_full, token_count_abstract, importance_score, learned_at, expires_at, memory_type, metadata, guid FROM memories WHERE id IN ({id_placeholders}) AND status IN {status_filter}"
            params = tuple(graph_anchored_ids)
            with retriever.db.trace_query("SELECT", "memories", query, params) as span:
                cursor.execute(query, params)
                rows = cursor.fetchall()
            
            for row in rows:
                m_id, c_f, c_a, tc_f, tc_a, imp, lat, exp, m_type, meta, guid = row
                context.candidates[m_id] = {
                    "id": m_id, "content_full": c_f, "content_abstract": c_a,
                    "tc_full": tc_f or (len(c_f) // 4), "tc_abstract": tc_a or (len(c_a or "") // 4),
                    "importance": imp, "learned_at": lat, "expires_at": exp,
                    "source": "anchor", "type": m_type, "metadata": meta, "guid": guid
                }
        
        context.metrics["anchoring"] = {"found": len(graph_anchored_ids) if graph_anchored_ids else 0}

class VectorDiscovery(RetrievalHandler):
    """Stage B: Vector Fallback (Broad Search)"""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        # Triggered if freshness is requested OR we have fewer than 3 graph results
        if not context.is_fresh and len(context.candidates) >= 3:
            return

        cursor = retriever.db.get_cursor()
        candidate_limit = context.limit * 3
        
        allowed_owners = context.config.get("allowed_owners")
        include_archived = context.config.get("include_archived", False)
        status_filter = "('ACTIVE')" if not include_archived else "('ACTIVE', 'ARCHIVED')"
        
        try:
            import sqlite_vec
            filter_clause = ""
            v_params = [sqlite_vec.serialize_float32(context.query_vector), candidate_limit]
            
            if allowed_owners:
                placeholders = ",".join(["?"] * len(allowed_owners))
                filter_clause = f"AND (m.owner_id IN ({placeholders}) OR m.privacy = 'PUBLIC')"
                v_params.extend(allowed_owners)
            
            v_query = f"""
                SELECT m.id, m.content_full, m.content_abstract, m.token_count_full, m.token_count_abstract, m.importance_score, m.learned_at, m.expires_at, v.distance, m.memory_type, m.metadata, m.guid
                FROM memories_vec v JOIN memories m ON v.rowid = m.id
                WHERE v.embedding MATCH ? AND v.k = ? AND m.status IN {status_filter} {filter_clause}
                ORDER BY v.distance ASC
            """
            with retriever.db.trace_query("SELECT", "memories", v_query, tuple(v_params)) as span:
                cursor.execute(v_query, v_params)
                rows = cursor.fetchall()
            
            count = 0
            for row in rows:
                m_id, c_f, c_a, tc_f, tc_a, imp, lat, exp, dist, m_type, meta, guid = row
                similarity = 1.0 / (1.0 + dist)
                
                # Telemetry for Precision Histogram
                with tracer.start_as_current_span("reverie.retrieval.precision_log") as p_span:
                    p_span.set_attribute("retrieval.score", similarity)
                    p_span.set_attribute("memory.id", m_id)
                    # Add content snippet for visual debugging in Jaeger
                    p_span.set_attribute("memory.content_snippet", (c_a or c_f)[:200])

                # Precision Gate
                if similarity < 0.45:
                    continue
                    
                if m_id not in context.candidates:
                    context.candidates[m_id] = {
                        "id": m_id, "content_full": c_f, "content_abstract": c_a,
                        "tc_full": tc_f or (len(c_f) // 4), "tc_abstract": tc_a or (len(c_a or "") // 4),
                        "importance": imp, "learned_at": lat, "expires_at": exp,
                        "similarity": similarity, "source": "vector",
                        "type": m_type, "metadata": meta, "guid": guid
                    }
                    count += 1
            
            context.metrics["vector"] = {"found": count}
        except Exception as e:
            logger.error(f"Vector discovery failed: {e}")

class GraphExpansionDiscovery(RetrievalHandler):
    """Stage C: Graph Augmentation (Connections from top results)"""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        if context.is_fresh or not context.candidates:
            return

        # Sort current candidates by temporary score (or importance if scores not set yet)
        # For expansion, we look at the top 3 results currently in the pool
        seed_ids = [cid for cid, c in sorted(context.candidates.items(), key=lambda x: x[1].get("similarity", x[1]["importance"]/10.0), reverse=True)[:3]]
        
        if not seed_ids:
            return

        # Calculate dynamic gravity
        gravity = retriever._calculate_gravity(context.query_text, list(context.candidates.values())[:5])
        
        # Iterative expansion
        linked_results = retriever.graph.get_related_memories(seed_ids, anchor_entities=context.anchors, gravity=gravity, depth=1)
        depth = 1
        
        count = len(linked_results)
        avg_signal = sum(linked_results.values()) / count if count > 0 else 0.0
        
        if count < 3 or avg_signal < 0.6:
            linked_results = retriever.graph.get_related_memories(seed_ids, anchor_entities=context.anchors, gravity=gravity, depth=2)
            depth = 2
            avg_signal = sum(linked_results.values()) / len(linked_results) if linked_results else 0.0

        cursor = retriever.db.get_cursor()
        new_ids = [i for i in linked_results.keys() if i not in context.candidates]
        
        include_archived = context.config.get("include_archived", False)
        status_filter = "('ACTIVE')" if not include_archived else "('ACTIVE', 'ARCHIVED')"
        
        if new_ids:
            id_placeholders = ",".join(["?"] * len(new_ids))
            fetch_query = f"SELECT id, content_full, content_abstract, token_count_full, token_count_abstract, importance_score, learned_at, expires_at, memory_type, metadata, guid FROM memories WHERE id IN ({id_placeholders}) AND status IN {status_filter}"
            params = tuple(new_ids)
            with retriever.db.trace_query("SELECT", "memories", fetch_query, params) as span:
                cursor.execute(fetch_query, params)
                rows = cursor.fetchall()
            
            for row in rows:
                m_id, c_f, c_a, tc_f, tc_a, imp, lat, exp, m_type, meta, guid = row
                context.candidates[m_id] = {
                    "id": m_id, "content_full": c_f, "content_abstract": c_a,
                    "tc_full": tc_f or (len(c_f) // 4), "tc_abstract": tc_a or (len(c_a or "") // 4),
                    "importance": imp, "learned_at": lat, "expires_at": exp,
                    "discovery_boost": linked_results.get(m_id, 0.5),
                    "source": "graph", "type": m_type, "metadata": meta, "guid": guid
                }
                
        context.metrics["graph_expansion"] = {"depth": depth, "found": len(new_ids), "signal": avg_signal}

class IntentRanker(RetrievalHandler):
    """Detects intent and sets weights."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        query_lower = context.query_text.lower()
        fact_markers = ["what is", "how ", "who ", "where ", "when ", "why ", "list ", "explain ", "identify"]
        
        # Check if weights were manually overridden in config
        manual_sw = context.config.get("similarity_weight")
        manual_iw = context.config.get("importance_weight")
        manual_dw = context.config.get("decay_weight")
        
        if manual_sw is not None and manual_iw is not None and manual_dw is not None:
            context.intent = "Manual Override"
            context.weights = {"similarity": manual_sw, "importance": manual_iw, "decay": manual_dw}
        elif any(m in query_lower for m in fact_markers):
            context.intent = "Fact-Seeking"
            context.weights = {"similarity": 0.7, "importance": 0.1, "decay": 0.2}
        else:
            context.intent = "Exploration"
            context.weights = {"similarity": 0.4, "importance": 0.4, "decay": 0.2}
            
        context.metrics["intent"] = context.intent

class ScoringRanker(RetrievalHandler):
    """Calculates final combined score for all candidates."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        sw = context.weights["similarity"]
        iw = context.weights["importance"]
        dw = context.weights["decay"]
        
        for cid, c in context.candidates.items():
            # 1. Similarity (from vector search or default for graph/anchor)
            sim = c.get("similarity", 0.6 if c["source"] == "anchor" else 0.4)
            
            # 2. Decay
            decay = 1.0 if context.is_fresh else retriever._calculate_decay(c["learned_at"], c["importance"], c["expires_at"])
            
            # 3. Importance (normalized 0-1)
            imp = min(c["importance"] / 10.0, 1.0)
            
            # 4. Boosts
            boost = 0.0
            if c["source"] == "anchor": boost = 0.2
            elif c["source"] == "graph": boost = c.get("discovery_boost", 0.5) * 0.1
            
            # Final Score
            c["score"] = (sim * sw) + (imp * iw) + (decay * dw) + boost

class BudgetHandler(RetrievalHandler):
    """Selects results and formats output strings."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        # 1. Fetch relevance floor from config (default to 0.2 if not set)
        relevance_floor = context.config.get("relevance_floor", 0.2)

        # Sort candidates by score
        sorted_candidates = sorted(context.candidates.values(), key=lambda x: x["score"], reverse=True)
        
        strategy = context.config.get("strategy", "balanced")
        
        # Pre-fetch all neighbor summaries for candidates that need them
        candidate_ids_for_summary = [
            c["id"] for c in sorted_candidates[:context.limit] 
            if c["type"] == "OBSERVATION" or c["source"] in ["graph", "anchor"]
        ]
        all_summaries = retriever.graph.get_neighbors_summaries(candidate_ids_for_summary)
        
        for c in sorted_candidates:
            # 2. Apply Relevance Floor: Skip noise, even if it fits the budget
            if c["score"] < relevance_floor:
                continue
                
            if len(context.results) >= context.limit:
                break
                
            chosen_content = None
            chosen_tokens = 0
            version = "full"
            
            if strategy == "abstract_only" and c["content_abstract"]:
                chosen_content = c["content_abstract"]; chosen_tokens = c["tc_abstract"]; version = "abstract"
            else:
                if context.consumed_tokens + c["tc_full"] <= context.token_budget:
                    chosen_content = c["content_full"]; chosen_tokens = c["tc_full"]; version = "full"
                elif c["content_abstract"] and context.consumed_tokens + c["tc_abstract"] <= context.token_budget:
                    chosen_content = c["content_abstract"]; chosen_tokens = c["tc_abstract"]; version = "abstract"
            
            if chosen_content:
                # Map Metadata
                label = "Incidental"
                if c["importance"] >= 8.0: label = "Critical"
                elif c["importance"] >= 4.0: label = "Relevant"
                
                date_str = "Unknown"
                if c["learned_at"]:
                    date_str = c["learned_at"].split("T")[0] if "T" in c["learned_at"] else c["learned_at"].split(" ")[0]
                
                location = "N/A"
                if c.get("metadata"):
                    try:
                        meta_dict = json.loads(c["metadata"]) if isinstance(c["metadata"], str) else c["metadata"]
                        location = meta_dict.get("location") or meta_dict.get("geolocation") or "N/A"
                    except: pass

                # Build Header
                guid = c.get("guid") or f"mem_{c['id']}"
                header = (
                    f"### MEMORY ID: {guid}\n"
                    f"- Timestamp: {date_str}\n"
                    f"- Category: {c['type']}\n"
                    f"- Importance: {label}\n"
                    f"- Location: {location}\n"
                    f"- Context:\n"
                )
                
                indented_content = "  " + chosen_content.replace("\n", "\n  ")
                display_content = header + indented_content

                if c["id"] in all_summaries:
                    display_content += f"\n  {all_summaries[c['id']].strip()}"
                    
                # Hierarchical Discovery
                if c["type"] == "OBSERVATION" and c.get("metadata"):
                    try:
                        meta = json.loads(c["metadata"]) if isinstance(c["metadata"], str) else c["metadata"]
                        child_ids = meta.get("source_ids", [])
                        if child_ids:
                            display_content += f"\n  [Nuanced Details available via recall_reverie for IDs: {child_ids}]"
                    except: pass

                context.results.append({
                    "id": c["id"], 
                    "content": display_content, 
                    "tokens": chosen_tokens, 
                    "version": version, 
                    "score": c["score"]
                })
                context.consumed_tokens += chosen_tokens

# --- Handler Registry ---
# Maps string names to handler classes for config-driven pipelines
HANDLER_REGISTRY = {
    "anchoring": AnchoringDiscovery,
    "vector": VectorDiscovery,
    "graph_expansion": GraphExpansionDiscovery,
    "intent": IntentRanker,
    "scoring": ScoringRanker,
    "rerank": RerankerHandler,
    "budget": BudgetHandler
}

class Retriever:
    """RAG Engine: Handles vector search, importance-based ranking, and graph traversal."""
    
    def __init__(self, db: DatabaseManager, enrichment: Any = None):
        self.db = db
        self.graph = GraphQueryService(db)
        self.enrichment = enrichment
        self.config = load_reverie_config()
        
        # Pipelines
        self.discovery_pipeline: List[RetrievalHandler] = []
        self.ranking_pipeline: List[RetrievalHandler] = []
        self.budget_pipeline: List[RetrievalHandler] = []
        
        self._setup_pipelines()

    def _setup_pipelines(self):
        """Initializes pipelines from config or defaults."""
        retrieval_cfg = self.config.get("retrieval_pipeline", {})
        
        # 1. Discovery
        discovery_names = retrieval_cfg.get("discovery")
        if discovery_names:
            for name in discovery_names:
                if name in HANDLER_REGISTRY:
                    self.register_handler(HANDLER_REGISTRY[name](), "discovery")
        else:
            # Default Discovery
            self.discovery_pipeline = [AnchoringDiscovery(), VectorDiscovery()]
            
        # 2. Ranking & Expansion
        ranking_names = retrieval_cfg.get("ranking")
        if ranking_names:
            for name in ranking_names:
                if name in HANDLER_REGISTRY:
                    self.register_handler(HANDLER_REGISTRY[name](), "ranking")
        else:
            # Default Ranking
            self.ranking_pipeline = [IntentRanker(), GraphExpansionDiscovery(), ScoringRanker(), RerankerHandler()]
            
        # 3. Budgeting
        budget_names = retrieval_cfg.get("budget")
        if budget_names:
            for name in budget_names:
                if name in HANDLER_REGISTRY:
                    self.register_handler(HANDLER_REGISTRY[name](), "budget")
        else:
            # Default Budgeting
            self.budget_pipeline = [BudgetHandler()]

    def _setup_default_pipelines(self):
        """DEPRECATED: Use _setup_pipelines instead."""
        self._setup_pipelines()

    def register_handler(self, handler: RetrievalHandler, stage: str = "discovery"):
        """Allows external plugins to inject logic into the pipeline."""
        if stage == "discovery":
            self.discovery_pipeline.append(handler)
        elif stage == "ranking":
            self.ranking_pipeline.append(handler)
        elif stage == "budget":
            self.budget_pipeline.append(handler)
        else:
            logger.warning(f"Unknown pipeline stage: {stage}")

    def _calculate_decay(self, learned_at_str: str, importance: float, expires_at: Optional[str] = None) -> float:
        """
        Calculates time decay score. 
        - Permanent memories (expires_at is NULL) have NO decay (1.0).
        - High importance memories (>= 8.0) have permanent weight (1.0).
        - Others follow a 48-hour half-life exponential decay.
        """
        if not expires_at or importance >= 8.0:
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
               similarity_weight: Optional[float] = None, 
               importance_weight: Optional[float] = None,
               decay_weight: Optional[float] = None,
               allowed_owners: List[str] = None,
               include_archived: bool = False,
               env: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Composable Pipeline Orchestrator:
        1. Initialize RetrievalContext.
        2. Run Discovery Handlers (Parallel-ready, currently serial).
        3. Run Ranking & Expansion Handlers (Serial).
        4. Run Budgeting & Selection Handlers (Serial).
        """
        # 1. Initialize Context
        config = {
            "strategy": strategy,
            "similarity_weight": similarity_weight,
            "importance_weight": importance_weight,
            "decay_weight": decay_weight,
            "allowed_owners": allowed_owners,
            "include_archived": include_archived
        }
        with tracer.start_as_current_span("reverie.retrieval") as span:
            span.set_attribute("retrieval.query", query_text)
            span.set_attribute("agent.context.token_budget_allocated", token_budget)
            context = RetrievalContext(query_text, query_vector, limit, token_budget, config, env=env)
            
            # 2. Discovery Phase
            for handler in self.discovery_pipeline:
                with tracer.start_as_current_span(f"reverie.retrieval.handler.{handler.__class__.__name__}") as h_span:
                    try:
                        handler.process(context, self)
                        h_span.set_attribute("retrieval.handler", handler.__class__.__name__)
                        h_span.set_attribute("retrieval.candidate_count", len(context.candidates))
                    except Exception as e:
                        h_span.set_status(StatusCode.ERROR)
                        h_span.record_exception(e)
                        logger.error(f"Discovery handler {handler.__class__.__name__} failed: {e}")
                
            # 3. Ranking & Expansion Phase
            for handler in self.ranking_pipeline:
                with tracer.start_as_current_span(f"reverie.retrieval.handler.{handler.__class__.__name__}") as h_span:
                    try:
                        handler.process(context, self)
                        h_span.set_attribute("retrieval.handler", handler.__class__.__name__)
                        h_span.set_attribute("retrieval.candidate_count", len(context.candidates))
                    except Exception as e:
                        h_span.set_status(StatusCode.ERROR)
                        h_span.record_exception(e)
                        logger.error(f"Ranking handler {handler.__class__.__name__} failed: {e}")
                
            # 4. Budgeting Phase
            for handler in self.budget_pipeline:
                with tracer.start_as_current_span(f"reverie.retrieval.handler.{handler.__class__.__name__}") as h_span:
                    try:
                        handler.process(context, self)
                        h_span.set_attribute("retrieval.handler", handler.__class__.__name__)
                    except Exception as e:
                        h_span.set_status(StatusCode.ERROR)
                        h_span.record_exception(e)
                        logger.error(f"Budgeting handler {handler.__class__.__name__} failed: {e}")
                
            span.set_attribute("retrieval.intent", context.intent)
            span.set_attribute("retrieval.result_count", len(context.results))
            span.set_attribute("agent.context.token_budget_allocated", context.consumed_tokens)
            span.set_attribute("agent.context.token_budget_remaining", context.remaining_budget)
            
            if context.results:
                avg_score = sum(r.get("score", 0.0) for r in context.results) / len(context.results)
                span.set_attribute("retrieval.score", avg_score)
            
            logger.info(f"Retrieved {len(context.results)} memories ({context.consumed_tokens}/{token_budget} tokens). Intent: {context.intent}, Metrics: {context.metrics}")
            
            # 5. Update Access Timestamps
            if context.results:
                self.db.update_access_timestamp([r["id"] for r in context.results])
                
            return context.results

    def find_duplicates(self, query_vector: List[float], threshold: float = 0.95, 
                        allowed_owners: List[str] = None) -> List[Dict[str, Any]]:
        """
        Finds memories with high cosine similarity for deduplication/merging.
        Used during sync_turn to maintain a canonical knowledge base.
        """
        cursor = self.db.get_cursor()
        import sqlite_vec
        
        # We check top 5 candidates to see if any cross the threshold
        v_params = [sqlite_vec.serialize_float32(query_vector), 5]
        filter_clause = ""
        if allowed_owners:
            placeholders = ",".join(["?"] * len(allowed_owners))
            filter_clause = f"AND (m.owner_id IN ({placeholders}) OR m.privacy = 'PUBLIC')"
            v_params.extend(allowed_owners)
            
        v_query = f"""
            SELECT m.id, m.content_full, v.distance
            FROM memories_vec v JOIN memories m ON v.rowid = m.id
            WHERE v.embedding MATCH ? AND v.k = ? AND m.status = 'ACTIVE' {filter_clause}
            ORDER BY v.distance ASC
        """
        
        duplicates = []
        try:
            with self.db.trace_query("SELECT", "memories", v_query, tuple(v_params)) as span:
                cursor.execute(v_query, v_params)
                rows = cursor.fetchall()
            for row in rows:
                m_id, content, dist = row
                # Convert distance to similarity (1.0 is exact match)
                similarity = 1.0 / (1.0 + dist)
                if similarity >= threshold:
                    duplicates.append({
                        "id": m_id, 
                        "content_full": content, 
                        "similarity": similarity
                    })
        except Exception as e:
            logger.error(f"Duplicate search failed: {e}")
            
        return duplicates
