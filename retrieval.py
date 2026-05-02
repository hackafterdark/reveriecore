import math
import json
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from .database import DatabaseManager
from .graph_query import GraphQueryService
from .config import load_reverie_config
from .telemetry import get_tracer
from .schemas import (
    MemoryType, RelationType, RetrievalIntent,
    RetrievalContext, RetrievalHandler,
    RetrievalConfig, MaintenanceConfig, MesaConfig,
    AnchoringConfig, VectorConfig, GraphExpansionConfig, IntentClassifierConfig,
    IntentConfig, ScoringConfig, DecayConfig, BudgetConfig, PruningConfig,
    RewriterConfig, RerankConfig, PipelineConfig
)

# Handlers (Top-level imports now safe due to schema leaf refactor)
from .reranking import RerankerHandler
from .rewriting import QueryRewriterHandler
from .pruning import PruningHandler

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)

# --- Handler Registry ---

HANDLER_REGISTRY = {
    "intent_classifier": None, # Set below after class definitions
    "anchoring": None,
    "vector": None,
    "graph_expansion": None,
    "intent": None,
    "scoring": None,
    "budget": None,
    "rewriter": QueryRewriterHandler,
    "rerank": RerankerHandler,
    "pruning": PruningHandler
}

# --- Discovery Stages ---

class IntentClassifierDiscovery(RetrievalHandler):
    """Stage A0: Intent Classification (Zero-Shot) to guide edge filtering."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        if not retriever.enrichment:
            return

        span = trace.get_current_span()
        cfg = self.config # IntentClassifierConfig
        labels = list(cfg.mappings.keys())
        
        scores = retriever.enrichment._zero_shot_classify(
            context.query_text, 
            labels, 
            hypothesis_template="The user intent is {}.",
            strategy=cfg.intent_strategy
        )
        
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]
        
        span.set_attribute("retrieval.intent", best_intent)
        span.set_attribute("retrieval.intent_confidence", float(confidence))
        
        context.metadata["intent"] = best_intent
        context.metadata["intent_confidence"] = confidence
        
        if confidence > cfg.confidence_threshold:
            context.metadata["allowed_edges"] = cfg.mappings.get(best_intent, [])
            span.set_attribute("retrieval.allowed_edges", context.metadata["allowed_edges"])

class AnchoringDiscovery(RetrievalHandler):
    """Stage A: Semantic Anchoring (Graph-First)"""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        cfg = self.config # AnchoringConfig
        context.is_fresh = any(k in context.query_text.lower() for k in cfg.clean_slate_keywords)
        
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
        cfg = self.config # VectorConfig
        if not context.is_fresh and len(context.candidates) >= cfg.fallback_threshold:
            return

        cursor = retriever.db.get_cursor()
        candidate_limit = context.limit * cfg.candidate_multiplier
        
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
                
                with tracer.start_as_current_span("reverie.retrieval.precision_log") as p_span:
                    p_span.set_attribute("retrieval.score", float(similarity))
                    p_span.set_attribute("memory.id", m_id)
                    p_span.set_attribute("memory.content_snippet", (c_a or c_f)[:200])

                if similarity < cfg.precision_gate:
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

        cfg = self.config # GraphExpansionConfig
        if not cfg:
            logger.warning("GraphExpansionDiscovery: No configuration provided. Skipping.")
            return

        seed_ids = [cid for cid, c in sorted(context.candidates.items(), key=lambda x: x[1].get("similarity", x[1]["importance"]/10.0), reverse=True)[:cfg.seed_limit]]
        
        if not seed_ids:
            return

        gravity = retriever._calculate_gravity(context.query_text, list(context.candidates.values())[:5])
        
        allowed_edges = context.metadata.get("allowed_edges")
        linked_results = retriever.graph.get_related_memories(
            seed_ids, 
            anchor_entities=context.anchors, 
            gravity=gravity, 
            depth=1,
            allowed_edges=allowed_edges
        )
        depth = 1
        
        count = len(linked_results)
        avg_signal = sum(linked_results.values()) / count if count > 0 else 0.0
        
        if count < cfg.seed_limit or avg_signal < cfg.min_signal:
            linked_results = retriever.graph.get_related_memories(
                seed_ids, 
                anchor_entities=context.anchors, 
                gravity=gravity, 
                depth=2,
                allowed_edges=allowed_edges
            )
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
                    "discovery_boost": linked_results.get(m_id, cfg.discovery_boost),
                    "source": "graph", "type": m_type, "metadata": meta, "guid": guid
                }
                
        context.metrics["graph_expansion"] = {"depth": depth, "found": len(new_ids), "signal": avg_signal}

# --- Ranking Stages ---

class IntentRanker(RetrievalHandler):
    """Detects intent and sets weights."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        query_lower = context.query_text.lower()
        cfg = self.config # IntentConfig
        
        model_intent = context.metadata.get("intent")
        model_confidence = context.metadata.get("intent_confidence", 0.0)
        
        is_fact_model = model_intent in [
            "troubleshooting and root cause analysis", 
            "step-by-step instructions and prerequisites"
        ] and model_confidence > 0.3
        
        manual_sw = context.config.get("similarity_weight")
        manual_iw = context.config.get("importance_weight")
        manual_dw = context.config.get("decay_weight")
        
        if manual_sw is not None or manual_iw is not None or manual_dw is not None:
            context.intent = "Manual Override"
            base_w = cfg.weights["exploration"]
            context.weights = {
                "similarity": manual_sw if manual_sw is not None else base_w.similarity,
                "importance": manual_iw if manual_iw is not None else base_w.importance,
                "decay": manual_dw if manual_dw is not None else base_w.decay
            }
        elif is_fact_model or any(m in query_lower for m in cfg.fact_markers):
            context.intent = "Fact-Seeking"
            w = cfg.weights["fact_seeking"]
            context.weights = {"similarity": w.similarity, "importance": w.importance, "decay": w.decay}
        else:
            context.intent = "Exploration"
            w = cfg.weights["exploration"]
            context.weights = {"similarity": w.similarity, "importance": w.importance, "decay": w.decay}
            
        context.metrics["intent"] = context.intent

class ScoringRanker(RetrievalHandler):
    """Calculates final combined score for all candidates."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        sw = context.weights["similarity"]
        iw = context.weights["importance"]
        dw = context.weights["decay"]
        cfg = self.config # ScoringConfig
        
        for cid, c in context.candidates.items():
            default_sim = cfg.default_similarities.get(c["source"], cfg.default_similarities.get("other", 0.4))
            sim = c.get("similarity", default_sim)
            decay = 1.0 if context.is_fresh else retriever._calculate_decay(c["learned_at"], c["importance"], c["expires_at"])
            imp = min(c["importance"] / 10.0, 1.0)
            
            boost = 0.0
            if c["source"] == "anchor": boost = cfg.anchor_boost
            elif c["source"] == "graph": boost = c.get("discovery_boost", 0.5) * cfg.graph_boost_multiplier
            
            c["score"] = (sim * sw) + (imp * iw) + (decay * dw) + boost

class BudgetHandler(RetrievalHandler):
    """Selects results and formats output strings."""
    def process(self, context: RetrievalContext, retriever: 'Retriever') -> None:
        cfg = self.config # BudgetConfig
        relevance_floor = cfg.relevance_floor

        sorted_candidates = sorted(context.candidates.values(), key=lambda x: x["score"], reverse=True)
        strategy = context.config.get("strategy", "balanced")
        
        candidate_ids_for_summary = [
            c["id"] for c in sorted_candidates[:context.limit] 
            if c["type"] == "OBSERVATION" or c["source"] in ["graph", "anchor"]
        ]
        all_summaries = retriever.graph.get_neighbors_summaries(candidate_ids_for_summary)
        
        for c in sorted_candidates:
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
                label = "Incidental"
                if c["importance"] >= cfg.labels.get("critical", 8.0): label = "Critical"
                elif c["importance"] >= cfg.labels.get("relevant", 4.0): label = "Relevant"
                
                date_str = "Unknown"
                if c["learned_at"]:
                    date_str = c["learned_at"].split("T")[0] if "T" in c["learned_at"] else c["learned_at"].split(" ")[0]
                
                location = "N/A"
                if c.get("metadata"):
                    try:
                        meta_dict = json.loads(c["metadata"]) if isinstance(c["metadata"], str) else c["metadata"]
                        location = meta_dict.get("location") or meta_dict.get("geolocation") or "N/A"
                    except: pass

                guid = c.get("guid") or f"mem_{c['id']}"
                header_title = f"### MEMORY ID: {guid}" if context.include_ids else "### MEMORY"
                header = (
                    f"{header_title}\n"
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

# Finalize Registry
HANDLER_REGISTRY.update({
    "intent_classifier": IntentClassifierDiscovery,
    "anchoring": AnchoringDiscovery,
    "vector": VectorDiscovery,
    "graph_expansion": GraphExpansionDiscovery,
    "intent": IntentRanker,
    "scoring": ScoringRanker,
    "budget": BudgetHandler
})

# --- Core Engine ---

class Retriever:
    """RAG Engine: Handles vector search, importance-based ranking, and graph traversal."""
    
    def __init__(self, db: DatabaseManager, enrichment: Any = None):
        self.db = db
        self.graph = GraphQueryService(db)
        self.enrichment = enrichment
        raw_cfg = load_reverie_config()
        self.config = RetrievalConfig.from_dict(raw_cfg)
        self.discovery_pipeline: List[RetrievalHandler] = []
        self.ranking_pipeline: List[RetrievalHandler] = []
        self.budget_pipeline: List[RetrievalHandler] = []
        
        self._setup_pipelines()

    def _get_handler_config(self, name: str) -> Optional[BaseModel]:
        """Unified mapping for handler-specific sub-configurations."""
        cfg = self.config
        
        if name == "anchoring": return cfg.discovery.anchoring
        if name == "vector": return cfg.discovery.vector
        if name == "graph_expansion": return cfg.discovery.graph_expansion
        if name == "intent_classifier": return cfg.discovery.intent_classifier
        
        if name == "intent": return cfg.ranking.intent
        if name == "scoring": return cfg.ranking.scoring
        
        if name == "rewriter": return cfg.rewriter
        if name == "pruning": return cfg.pruning
        if name == "budget": return cfg.budget
        if name == "rerank": return cfg.rerank
        
        return None

    def _setup_pipelines(self):
        """Initializes pipelines from config or defaults using unified mapping."""
        cfg = self.config
        
        for name in cfg.pipeline.discovery:
            if name in HANDLER_REGISTRY:
                h_cls = HANDLER_REGISTRY[name]
                h_cfg = self._get_handler_config(name)
                self.register_handler(h_cls(config=h_cfg), "discovery")
            
        for name in cfg.pipeline.ranking:
            if name in HANDLER_REGISTRY:
                h_cls = HANDLER_REGISTRY[name]
                h_cfg = self._get_handler_config(name)
                self.register_handler(h_cls(config=h_cfg), "ranking")
            
        for name in cfg.pipeline.budget:
            if name in HANDLER_REGISTRY:
                h_cls = HANDLER_REGISTRY[name]
                h_cfg = self._get_handler_config(name)
                self.register_handler(h_cls(config=h_cfg), "budget")

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
        cfg = self.config.ranking.decay
        if not expires_at or importance >= 8.0:
            return 1.0
        
        try:
            learned_at = datetime.fromisoformat(learned_at_str.replace("Z", "+00:00"))
            now = datetime.utcnow()
            age_hours = (now - learned_at).total_seconds() / 3600.0
            
            decay = math.pow(0.5, age_hours / cfg.half_life_hours)
            return max(cfg.min_decay, decay)
        except Exception as e:
            logger.debug(f"Decay calculation failed: {e}")
            return 1.0

    def _calculate_gravity(self, query: str, primary_candidates: List[Dict[str, Any]]) -> float:
        if not self.enrichment:
            return 1.0
            
        intent_scores = self.enrichment.classify_intent(query)
        total_score = sum(intent_scores.values())
        if total_score > 0:
            for k in intent_scores:
                intent_scores[k] /= total_score
        
        base_gravity = (
            (intent_scores.get('retrieving specific facts or entities', 0) * 1.0) +
            (intent_scores.get('synthesizing related information', 0) * 0.8) +
            (intent_scores.get('exploring open-ended possibilities', 0) * 0.5)
        )
        
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
               limit: Optional[int] = None, 
               token_budget: Optional[int] = None,
               strategy: str = "balanced",
               similarity_weight: Optional[float] = None, 
               importance_weight: Optional[float] = None,
               decay_weight: Optional[float] = None,
               allowed_owners: List[str] = None,
               include_archived: bool = False,
               include_ids: bool = True,
               env: Optional[Any] = None) -> List[Dict[str, Any]]:
        limit = limit if limit is not None else self.config.discovery.default_limit
        token_budget = token_budget if token_budget is not None else self.config.budget.default_token_budget

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
            context.include_ids = include_ids

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
                span.set_attribute("retrieval.score", float(avg_score))

            logger.info(f"Retrieved {len(context.results)} memories ({context.consumed_tokens}/{token_budget} tokens). Intent: {context.intent}, Metrics: {context.metrics}")
            
            if context.results:
                self.db.update_access_timestamp([r["id"] for r in context.results])
                
            return context.results

    def find_duplicates(self, query_vector: List[float], threshold: float = 0.95, 
                        allowed_owners: List[str] = None) -> List[Dict[str, Any]]:
        cursor = self.db.get_cursor()
        import sqlite_vec
        
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
