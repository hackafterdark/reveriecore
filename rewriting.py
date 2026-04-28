import logging
import time
from typing import Any, Optional
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from .telemetry import get_tracer
from .retrieval_base import RetrievalContext, RetrievalHandler

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)

class QueryRewriterHandler(RetrievalHandler):
    """
    Stage A.2: Generative Query expansion.
    Rewrites short technical queries into detailed search terms using a local CPU-bound model.
    """
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self.generator = None
        self._initialized = False
        
    def _lazy_init(self, config: dict):
        if self._initialized:
            return
            
        self._initialized = True
        rewriter_cfg = config.get("settings", {}).get("rewriter", {})
        if not rewriter_cfg.get("enabled", True):
            return

        self.model_path = rewriter_cfg.get("model_path", "models/phi-3-mini-q4.gguf")
        self.max_words = rewriter_cfg.get("max_words", 10)
        threads = rewriter_cfg.get("threads", 2)
        
        import os
        if not os.path.exists(self.model_path):
            logger.warning(f"QueryRewriter model file NOT found at {self.model_path}. Expansion will be disabled.")
            self.skip_reason = "model_file_missing"
            return

        try:
            import llama_cpp
            logger.info(f"Initializing QueryRewriter with model: {self.model_path} (Threshold: {self.max_words} words)")
            self.generator = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=512,
                n_threads=threads,
                n_gpu_layers=0, # Force CPU
                verbose=False
            )
            logger.info("QueryRewriter initialized successfully.")
        except ImportError:
            logger.warning("llama-cpp-python is not installed. Query expansion disabled.")
            self.skip_reason = "library_missing"
            self.generator = None
        except Exception as e:
            logger.warning(f"Failed to initialize QueryRewriter model at {self.model_path}: {e}")
            self.skip_reason = "load_failed"
            self.generator = None

    def process(self, context: RetrievalContext, retriever: Any) -> None:
        from . import rewriting
        with rewriting.tracer.start_as_current_span("reverie.retrieval.handler.QueryRewriterHandler") as span:
            # Lazy init to handle model loading after config is available
            self._lazy_init(retriever.config)
            
            original_query = context.query_text
            word_count = len(original_query.split())
            span.set_attribute("rag.retrieval.word_count", word_count)

            # Resiliency Guard: Skip if model failed to load
            if not self.generator:
                span.set_attribute("rag.retrieval.skip_reason", getattr(self, "skip_reason", "model_not_loaded"))
                return

            # Skip if query is already detailed
            if word_count > self.max_words:
                span.set_attribute("rag.retrieval.skip_reason", "query_already_detailed")
                span.set_attribute("rag.retrieval.max_words_threshold", self.max_words)
                return
                
            # Skip if clean slate requested
            if context.is_fresh:
                span.set_attribute("rag.retrieval.skip_reason", "fresh_context_requested")
                return

            start_time = time.time()
            try:
                prompt = f"Rewrite this technical search query for optimal RAG retrieval. Output ONLY the query, no filler.\n\nQuery: {original_query}\nExpanded Query:"
                
                output = self.generator(
                    prompt,
                    max_tokens=64,
                    temperature=0.0,
                    stop=["\n"],
                    echo=False
                )
                
                rewritten_text = output["choices"][0]["text"].strip()
                
                # Calculate "useful" metrics
                original_tokens = len(original_query.split())
                rewritten_tokens = len(rewritten_text.split())
                token_diff = rewritten_tokens - original_tokens
                is_rewritten = bool(rewritten_text) and rewritten_text != original_query

                if is_rewritten:
                    latency = (time.time() - start_time) * 1000
                    logger.info(f"Query rewritten: '{original_query}' -> '{rewritten_text}' ({latency:.2f}ms)")
                    
                    context.query_text = rewritten_text
                    
                    # Update vector for subsequent vector search
                    if retriever.enrichment:
                        context.query_vector = retriever.enrichment.generate_embedding(rewritten_text)
                    
                    span.set_attribute("rag.retrieval.rewrite_latency", latency)
                else:
                    span.set_attribute("rag.retrieval.rewrite_skipped", "No change or empty output")
                
                # --- OTel Attribution ---
                span.set_attribute("rag.retrieval.original_query", original_query)
                span.set_attribute("rag.retrieval.rewritten_query", rewritten_text or original_query)
                span.set_attribute("rag.retrieval.token_delta", token_diff)
                span.set_attribute("rag.retrieval.is_rewritten", is_rewritten)
                    
            except Exception as e:
                logger.error(f"Query rewriting failed: {e}")
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)
