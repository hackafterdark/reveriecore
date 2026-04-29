import logging
import time
from typing import List, Dict, Any, Optional
from .config import load_reverie_config
from .retrieval_base import RetrievalHandler, RetrievalContext
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    Ranker, RerankRequest = None, None
from opentelemetry import trace
from .telemetry import get_tracer

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)

class RerankerHandler(RetrievalHandler):
    """Stage D: Cross-Encoder Reranking using FlashRank"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # 1. Resolve Configuration
        self.config = config or load_reverie_config()
        cfg = self.config
        
        self.model_name = cfg.get("reranking_model") or kwargs.get("model_name") or "ms-marco-MiniLM-L-12-v2"

        # Defensive Check: Ensure we have strings
        if not isinstance(self.model_name, str):
            self.model_name = "ms-marco-MiniLM-L-12-v2"

        self._ranker = None

    @property
    def ranker(self):
        """Lazy load the FlashRank ranker."""
        if self._ranker is None:
            try:
                from flashrank import Ranker
                logger.info(f"ReverieCore: Initializing Reranker model ({self.model_name})...")
                self._ranker = Ranker(model_name=self.model_name)
                logger.info("ReverieCore: Reranker model initialized successfully.")
            except ImportError:
                # We handle this in process() to avoid repeated errors
                return None
            except Exception as e:
                logger.error(f"Failed to initialize FlashRank: {e}")
                return None
        return self._ranker

    def is_available(self) -> bool:
        """Check if FlashRank is installed and ready."""
        try:
            import flashrank
            return True
        except ImportError:
            return False

    def process(self, context: RetrievalContext, retriever: Any) -> None:
        """Rerank candidates using FlashRank."""
        if not context.candidates or len(context.candidates) <= 1:
            return

        # Prepare candidates for FlashRank
        # Format: [{"id": cid, "text": "..."}, ...]
        passages = []
        for cid, c in context.candidates.items():
            # Use content_full for maximum precision during reranking
            passages.append({
                "id": cid, 
                "text": c.get("content_full", "")
            })
        
        with tracer.start_as_current_span("reverie.retrieval.handler.RerankerHandler") as span:
            span.set_attribute("retrieval.handler", self.__class__.__name__)
            span.set_attribute("rag.retrieval.rerank_candidate_count", len(passages))
            
            # Skip if no query text (e.g. vector-only search from some tests)
            if not context.query_text:
                logger.debug("No query text provided. Skipping rerank.")
                return

            start_time = time.perf_counter()
            try:
                # 1. Check for library availability
                if not self.is_available():
                    logger.debug("FlashRank not found. Skipping rerank.")
                    return

                # 2. Lazy load ranker
                ranker = self.ranker
                if ranker is None:
                    return

                # 3. Execute Rerank using RerankRequest
                from flashrank import RerankRequest
                req = RerankRequest(query=context.query_text, passages=passages)
                results = ranker.rerank(req)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("rag.retrieval.rerank_latency", float(latency_ms))

                
                # Update scores and source
                for r in results:
                    cid = r["id"]
                    if cid in context.candidates:
                        # Override the score with the high-precision reranker score
                        context.candidates[cid]["score"] = r["score"]
                        context.candidates[cid]["source"] = "reranked"
                        
                logger.debug(f"Reranked {len(results)} candidates in {latency_ms:.2f}ms")
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
