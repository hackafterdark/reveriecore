import logging
import os
import json
import re
import urllib.request
import traceback
import threading
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .schemas import MemoryType, RelationType
from .config import load_reverie_config
from abc import ABC, abstractmethod
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from .telemetry import get_tracer

tracer = get_tracer(__name__)

logger = logging.getLogger(__name__)

class EnrichmentContext:
    """Mutable state container for the enrichment (ingestion) pipeline."""
    def __init__(self, text: str, env: Optional[Any] = None, metadata: Dict[str, Any] = None):
        self.text = text
        self.env = env # EnvironmentalContext
        self.metadata = metadata or {}
        
        # Intermediate/Final Outputs
        self.memory_type: MemoryType = MemoryType.CONVERSATION
        self.importance_score: float = 2.0
        self.profile: str = ""
        self.embedding: List[float] = []
        self.entities: List[Dict] = []
        self.relations: List[Dict] = []
        
        self.metrics: Dict[str, Any] = {}
        self.expires_at: Optional[str] = None
        self.token_count_full: int = 0
        self.token_count_abstract: int = 0

class EnrichmentHandler(ABC):
    """Abstract base class for all enrichment pipeline handlers."""
    @abstractmethod
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        pass

# --- Importance Handlers ---

class HeuristicImportance(EnrichmentHandler):
    """Tier 1: Fast, rule-based scoring (Errors, Deadlines)."""
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        text_lower = context.text.lower()
        is_important = False
        
        # 1. Error & Failure Patterns
        if any(term in text_lower for term in ["error", "exception", "traceback", "failed", "crash", "broken", "bug"]):
            is_important = True
        # 2. Project/Temporal Urgency
        elif any(term in text_lower for term in ["deadline", "critical", "urgent", "asap", "priority", "important"]):
            is_important = True
        # 3. Security/Identity
        elif any(term in text_lower for term in ["password", "secret", "api_key", "token", "auth", "credentials"]):
            is_important = True
        # 4. Structural markers
        elif "```" in text_lower or "def " in text_lower or "class " in text_lower or "import " in text_lower:
            is_important = True
            
        if is_important:
            context.importance_score = 9.5
            context.metrics["importance_source"] = "heuristics"
            context.metrics["stage_complete"] = True
            
            # Heuristic Type Override
            if any(kw in text_lower for kw in ["error", "exception", "traceback"]):
                context.memory_type = MemoryType.RUNTIME_ERROR
            elif any(kw in text_lower for kw in ["todo", "task", "goal"]):
                context.memory_type = MemoryType.TASK

class ModelImportance(EnrichmentHandler):
    """Tier 2: Local semantic weight using mDeBERTa-v3/BART."""
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        if context.importance_score > 5.0: # Skip if heuristics already flagged it
            return
            
        labels = ["critical", "important", "minor", "trivial"]
        scores = service._zero_shot_classify(context.text, labels, "This information is {}.")
        
        # Weighted average shifted to 0-10 scale
        raw_score = (scores["critical"] * 10.0) + (scores["important"] * 7.0) + (scores["minor"] * 3.0) + (scores["trivial"] * 1.0)
        context.importance_score = max(0.0, min(10.0, raw_score))
        context.metrics["importance_source"] = "model"

class SoulImportance(EnrichmentHandler):
    """Tier 3: Identity-relative scoring via remote LLM."""
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        if not service.soul_prompt or not service.llm_client.check_connectivity():
            return

        try:
            prompt = f"""
You are an expert operating under these principles: {service.soul_prompt}.
Assess the importance of this information on a scale of 0-10.
- Information critical to your goals and role gets 9-10.
- Information that is merely 'nice to know' gets 4-6.
- Incidental or conversational noise gets 0-2.

Output ONLY the JSON: {{"importance": float, "confidence": float}}
"""
            res = service.llm_client.call([
                {"role": "system", "content": "You are a professional importance scoring utility."},
                {"role": "user", "content": prompt + f"\n\nInformation: {context.text[:2000]}"}
            ], json_mode=True, telemetry_metadata={"reverie.handler": "SoulImportance"})
            
            if res and "importance" in res:
                conf = res.get("confidence", 0.9)
                context.importance_score = res["importance"]
                context.metrics["importance_source"] = "soul"
                context.metrics["importance_confidence"] = conf
                
                # If very confident, we can mark this stage as "resolved" 
                # (to be used by orchestrator for early exit)
                if conf >= 0.9:
                    context.metrics["stage_complete"] = True
        except Exception as e:
            logger.debug(f"Soul scoring failed: {e}")

# --- Analysis Handlers ---

class TypeClassifier(EnrichmentHandler):
    """Zero-shot classification for MemoryType."""
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        # 1. Heuristic Overrides
        text_lower = context.text.lower()
        if any(kw in text_lower for kw in ["error", "exception", "traceback"]):
            context.memory_type = MemoryType.RUNTIME_ERROR
            return
        if any(kw in text_lower for kw in ["todo", "task", "goal"]):
            context.memory_type = MemoryType.TASK
            return
            
        # 2. Model Classification
        mapping = {
            "observation, fact, status": MemoryType.OBSERVATION,
            "source code, programming, snippet, code": MemoryType.CODE_SNIPPET,
            "user preference, personalization": MemoryType.USER_PREFERENCE,
            "learning, discovery, insight": MemoryType.LEARNING_EVENT,
            "expired task, overdue": MemoryType.EXPIRED_TASK,
            "conversation, dialogue, chat": MemoryType.CONVERSATION
        }
        
        scores = service._zero_shot_classify(context.text, list(mapping.keys()), "This information is {}.")
        best_label = max(scores, key=scores.get)
        context.memory_type = mapping[best_label]
        context.metrics["classification_confidence"] = scores[best_label]

# --- Profiling Handlers ---

class SemanticProfiler(EnrichmentHandler):
    """Generates a 1-2 sentence 'gist' of the memory."""
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        context.profile = service.generate_semantic_profile(context.text)

class TextEmbedder(EnrichmentHandler):
    """Generates a 384-dim vector for the semantic profile."""
    def process(self, context: EnrichmentContext, service: 'EnrichmentService') -> None:
        # Embed the profile for cleaner signal, or fallback to full text
        source = context.profile or context.text
        context.embedding = service.generate_embedding(source)

class ConfigLoader:
    """Helper to parse ~/.hermes/config.yaml without external dependencies like PyYAML."""
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        from hermes_constants import get_hermes_home
        try:
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                logger.debug(f"Config not found at {config_path}")
                return {}
                
            content = config_path.read_text()
            return ConfigLoader.parse_yaml_minimal(content)
        except Exception as e:
            logger.warning(f"Failed to load Hermes config: {e}")
            return {}

    @staticmethod
    def parse_yaml_minimal(content: str) -> Dict[str, Any]:
        """Extremely simple YAML parser for standard Hermes sections."""
        config = {"providers": [], "model": {}}
        current_provider = {}
        
        lines = content.splitlines()
        in_custom_providers = False
        in_model_section = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
                
            # 1. Handle root sections
            if not line.startswith(" ") and not line.startswith("-"):
                in_custom_providers = stripped.startswith("custom_providers:")
                in_model_section = stripped.startswith("model:")
                if in_custom_providers or in_model_section:
                    continue

            # 2. Handle model: section (active config)
            if in_model_section:
                match = re.match(r"^(\w+):\s*(.*)$", stripped)
                if match:
                    key, value = match.groups()
                    config["model"][key] = value.split("#")[0].strip().strip("'").strip('"')
                continue

            # 3. Handle custom_providers: section (catalog)
            if in_custom_providers:
                if stripped.startswith("- "):
                    if current_provider:
                        config["providers"].append(current_provider)
                    current_provider = {}
                    stripped = stripped[2:].strip()
                
                match = re.match(r"^(\w+):\s*(.*)$", stripped)
                if match:
                    key, value = match.groups()
                    current_provider[key] = value.split("#")[0].strip().strip("'").strip('"')
            
        if current_provider:
            config["providers"].append(current_provider)
            
        return config

class InternalLLMClient:
    """OpenAI-compatible client using urllib for zero-dependency execution."""
    
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name

    def is_connected(self) -> bool:
        """Fast-fail check to see if the LLM provider is reachable (2s timeout)."""
        url = f"{self.base_url}/models"
        req = urllib.request.Request(url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
            
        try:
            with urllib.request.urlopen(req, timeout=2.0) as _:
                return True
        except Exception as e:
            logger.warning(f"LLM Provider Connectivity Check FAILED for {self.base_url}: {e}")
            return False

    def check_connectivity(self) -> bool:
        """Alias for is_connected for backward compatibility."""
        return self.is_connected()



    def call(self, messages: List[Dict[str, str]], json_mode: bool = True, telemetry_metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        with tracer.start_as_current_span("reverie.llm.call") as span:
            span.set_attribute("gen_ai.system", self.base_url)
            span.set_attribute("gen_ai.request.model", self.model_name)
            span.set_attribute("gen_ai.operation.name", "chat")
            
            if telemetry_metadata:
                for k, v in telemetry_metadata.items():
                    span.set_attribute(k, v)

            url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        
        try:
            # Slightly longer timeout for actual inference
            with urllib.request.urlopen(req, timeout=45) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                
                usage = result.get("usage", {})
                if usage:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.get("prompt_tokens", 0))
                    span.set_attribute("gen_ai.usage.output_tokens", usage.get("completion_tokens", 0))
                
                return json.loads(content) if json_mode else content
        except Exception as e:
            logger.error(f"InternalLLMClient.call FAILED: {e}")
            span.set_status(StatusCode.ERROR)
            span.record_exception(e)
            return None

class EnrichmentService:
    """The Intelligence Layer: Handles embeddings, BART classification, and profiling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # 1. Resolve Configuration
        self.config = config or load_reverie_config()
        cfg = self.config
        
        self.embedding_model_name = cfg.get("embedding_model") or kwargs.get("embedding_model_name") or "all-MiniLM-L6-v2"
        self.summarization_model_name = cfg.get("summarization_model") or kwargs.get("summarization_model_name") or "sshleifer/distilbart-cnn-12-6"
        self.classifier_model_name = cfg.get("classifier_model") or kwargs.get("classifier_model_name") or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

        # Defensive Check: Ensure we have strings, not dicts from positional mismatch
        if not isinstance(self.embedding_model_name, str):
            self.embedding_model_name = "all-MiniLM-L6-v2"
        if not isinstance(self.summarization_model_name, str):
            self.summarization_model_name = "sshleifer/distilbart-cnn-12-6"
        if not isinstance(self.classifier_model_name, str):
            self.classifier_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

        # Models initialized as None (Lazy-Loading)
        self.embedding_model = None
        self.summarizer = None
        self.summarizer_tokenizer = None
        
        self.classifier_model = None
        self.classifier_tokenizer = None
        
        # Concurrent access control
        self._init_lock = threading.Lock()

        # LLM Client for Graph Extraction
        h_cfg = ConfigLoader.load_config()
        
        # Priority Logic for Provider Selection:
        # 1. Use root-level 'model' section (Hermes default)
        # 2. Use 'custom_providers' catalog
        # 3. Fallback to localhost
        
        model_cfg = h_cfg.get("model", {})
        providers = h_cfg.get("providers", [])
        
        # Determine base_url
        base_url = model_cfg.get("base_url")
        if not base_url and providers:
            base_url = providers[0].get("base_url")
        if not base_url:
            base_url = "http://localhost:11434/v1"
            
        # Determine model_name
        model_name = model_cfg.get("default") or model_cfg.get("model")
        if not model_name and providers:
            model_name = providers[0].get("model")
        if not model_name:
            model_name = "gemma2:2b"
            
        # Determine api_key
        api_key = model_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or "sk-reverie-internal"
        
        self.llm_client = InternalLLMClient(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name
        )
        
        # Handler Registry for Enrichment
        self.HANDLER_REGISTRY = {
            "heuristics": HeuristicImportance,
            "model_importance": ModelImportance,
            "soul_importance": SoulImportance,
            "classifier": TypeClassifier,
            "profiler": SemanticProfiler,
            "embedder": TextEmbedder
        }
        
        # Pipeline Configuration (from config)
        self.analysis_pipeline: List[EnrichmentHandler] = []
        pipeline_cfg = self.config.get("enrichment_pipeline", {})
        
        # Load analysis stage (Importance & Classification)
        for h_name in pipeline_cfg.get("analysis", ["heuristics", "classifier", "model_importance", "soul_importance"]):
            if h_name in self.HANDLER_REGISTRY:
                self.analysis_pipeline.append(self.HANDLER_REGISTRY[h_name]())
                
        # Load profiling stage (Summary & Embedding)
        self.profiling_pipeline: List[EnrichmentHandler] = []
        for h_name in pipeline_cfg.get("profiling", ["profiler", "embedder"]):
            if h_name in self.HANDLER_REGISTRY:
                self.profiling_pipeline.append(self.HANDLER_REGISTRY[h_name]())
        
        # Telemetry
        self.telemetry = {"success": 0, "failure": 0}
        
        # Identity / Soul Property
        self.soul_prompt = self._load_soul_prompt()
        
        if self.soul_prompt:
            logger.info("Soul-Aware Importance Scoring enabled.")

        # Eagerly load and warmup models for startup
        self.initialize()

    def initialize(self):
        """Eagerly load and warmup all models during startup."""
        with tracer.start_as_current_span("reverie.enrichment.initialize") as span:
            logger.info("Initializing EnrichmentService models...")
            self._load_models()
            self.warmup()
            logger.info("EnrichmentService initialization complete.")

    def _load_models(self):
        """Thread-safe eager loader for all model backends."""
        with self._init_lock:
            if self.embedding_model is None:
                logger.info(f"Loading embedding model: {self.embedding_model_name}...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")

            if self.summarizer is None:
                logger.info(f"Loading summarization model: {self.summarization_model_name}...")
                self.summarizer_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
                self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(
                    self.summarization_model_name,
                    low_cpu_mem_usage=False
                ).to("cpu")

            if self.classifier_model is None:
                logger.info(f"Loading zero-shot classifier: {self.classifier_model_name}...")
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_name, use_fast=False)
                self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                    self.classifier_model_name,
                    low_cpu_mem_usage=False
                ).to("cpu")
                logger.info("mDeBERTa Classifier loaded successfully (Entailment-Logic).")

    def warmup(self):
        """Perform dummy inference to force PyTorch graph compilation and memory allocation."""
        with tracer.start_as_current_span("reverie.enrichment.warmup") as span:
            logger.info("Warming up models...")
            
            # 1. Warmup Embedding
            if self.embedding_model:
                logger.info(f"Warming up Embedding model ({self.embedding_model_name})...")
                self.embedding_model.encode(["warmup"], show_progress_bar=False)
                logger.info(f"Model {self.embedding_model_name} warmed and ready.")
            
            # 2. Warmup Summarizer
            if self.summarizer and self.summarizer_tokenizer:
                logger.info(f"Warming up Summarization model ({self.summarization_model_name})...")
                inputs = self.summarizer_tokenizer("warmup text for graph compilation", return_tensors="pt")
                self.summarizer.generate(inputs["input_ids"], max_length=5)
                logger.info(f"Model {self.summarization_model_name} warmed and ready.")
                
            # 3. Warmup Classifier
            if self.classifier_model and self.classifier_tokenizer:
                logger.info(f"Warming up Classifier model ({self.classifier_model_name})...")
                self._zero_shot_classify("warmup text for classification", ["fact", "noise"])
                logger.info(f"Model {self.classifier_model_name} warmed and ready.")

    def _ensure_loaded(self, models: List[str]):
        """Fallback lazy loader (now largely redundant due to eager initialization)."""
        # Quick check if everything is already there
        if "embedding" in models and self.embedding_model is None:
            self._load_models()
        elif "summarizer" in models and self.summarizer is None:
            self._load_models()
        elif "classifier" in models and self.classifier_model is None:
            self._load_models()

    def generate_embedding(self, text: str) -> List[float]:
        try:
            self._ensure_loaded(["embedding"])
            return self.embedding_model.encode([text], show_progress_bar=False)[0].tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * 384 

    def count_tokens(self, text: str) -> int:
        """Counts tokens using the summarizer tokenizer as a proxy."""
        if not text:
            return 0
        try:
            self._ensure_loaded(["summarizer"])
            inputs = self.summarizer_tokenizer(text, return_tensors="pt", truncation=False)
            return inputs["input_ids"].shape[1]
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to rough estimation (4 chars per token)
            return (len(text) // 4) + 1

    def generate_semantic_profile(self, text: str) -> str:
        if len(text.split()) < 30: 
            return text
        try:
            self._ensure_loaded(["summarizer"])
            inputs = self.summarizer_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            outputs = self.summarizer.generate(
                inputs["input_ids"], 
                max_length=150, 
                min_length=10, 
                num_beams=2, 
                early_stopping=True
            )
            summary = self.summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Semantic profiling failed: {e}")
            return text

    def synthesize_memories(self, memories: Dict[int, str], entity_name: str) -> str:
        """Uses LLM to synthesize multiple fragmented memories into one high-quality 'Observation Anchor'."""
        try:
            if not self.llm_client.check_connectivity():
                # Fallback: simple join
                return "\n".join([f"Memory {mid}: {txt}" for mid, txt in memories.items()])[:3000]

            prompt = (
                f"You are a memory consolidation service for a Knowledge Graph. "
                f"Below are several fragmented experiences and memories related to the entity '{entity_name}'. "
                "Your goal is to synthesize them into one single high-level OBSERVATION ANCHOR. "
                "\n\nGUIDELINES:\n"
                "1. Focus on PATTERNS and WISDOM. Do not just list the events; explain the underlying technical behavior or trend.\n"
                "2. Be comprehensive but high-level. Keep the gritty details accessible by referencing the 'Source IDs'.\n"
                "3. Use a tone of a generalist summarizing for a specialist.\n"
                "4. Structure the output clearly with a summary followed by a 'Linked Nuance' section listing the IDs."
            )
            
            context = "\n---\n".join([f"ID {mid}: {txt}" for mid, txt in memories.items()])
            
            summary = self.llm_client.call([
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Memories to consolidate:\n{context}"}
            ], telemetry_metadata={"reverie.operation": "Synthesis"})
            
            return summary if isinstance(summary, str) else str(summary)
        except Exception as e:
            logger.error(f"Memory synthesis failed: {e}")
            return "\n".join(memories)

    def _zero_shot_classify(self, text: str, labels: List[str], hypothesis_template: str = "This example is {}.") -> Dict[str, float]:
        """Manual implementation of zero-shot classification for MNLI-trained models (mDeBERTa/BART)."""
        self._ensure_loaded(["classifier"])
        
        scores = {}
        for label in labels:
            hypothesis = hypothesis_template.format(label)
            
            # AutoTokenizer handles the specific formatting for the model
            inputs = self.classifier_tokenizer(text, hypothesis, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                logits = self.classifier_model(**inputs).logits
            
            # mDeBERTa/BART MNLI Label Mapping:
            # Index 0: entailment, Index 1: neutral, Index 2: contradiction (DeBERTaV3-MNLI-XNLI)
            # wait, actually mDeBERTa-v3-base-mnli-xnli uses:
            # 0: entailment, 1: neutral, 2: contradiction
            # Let's verify and use the softmax on entailment vs contradiction.
            probs = F.softmax(logits, dim=1)
            scores[label] = probs[0, 0].item() # Entailment is at index 0 for this model
            
        return scores

    def classify_intent(self, query: str) -> Dict[str, float]:
        """Classifies the retrieval intent using zero-shot classification."""
        labels = [
            'retrieving specific facts or entities', # Precision
            'synthesizing related information',    # Synthesis
            'exploring open-ended possibilities'    # Exploration
        ]
        return self._zero_shot_classify(query, labels, "The user intent is {}.")

    def set_soul(self, prompt: str):
        """Updates the agent's identity/personality context for scoring."""
        self.soul_prompt = prompt
        logger.info("EnrichmentService soul updated.")

    def _load_soul_prompt(self) -> Optional[str]:
        """Loads personality prompt from SOUL.md in Hermes home."""
        try:
            from hermes_constants import get_hermes_home
            soul_path = get_hermes_home() / "SOUL.md"
            if soul_path.exists():
                content = soul_path.read_text().strip()
                if content:
                    return content
        except Exception as e:
            logger.debug(f"Failed to load soul prompt: {e}")
        return None

    def enrich(self, text: str, env: Optional[Any] = None) -> EnrichmentContext:
        """Composable Pipeline Orchestrator for ingestion."""
        with tracer.start_as_current_span("reverie.enrichment") as span:
            context = EnrichmentContext(text, env=env)
            
            # 1. Analysis Stage (Classification & Importance)
            for handler in self.analysis_pipeline:
                with tracer.start_as_current_span(f"reverie.enrichment.handler.{handler.__class__.__name__}") as h_span:
                    try:
                        handler.process(context, self)
                        h_span.set_attribute("handler.name", handler.__class__.__name__)
                        h_span.set_attribute("importance_score", context.importance_score)
                    except Exception as e:
                        h_span.set_status(StatusCode.ERROR)
                        h_span.record_exception(e)
                        logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                        
                # Early exit if a handler (like Heuristics or Soul) is highly confident
                if context.metrics.get("stage_complete"):
                    span.set_attribute("reverie.enrichment.early_exit", True)
                    break
                
            # 2. Profiling Stage (Summary & Embedding)
            for handler in self.profiling_pipeline:
                with tracer.start_as_current_span(f"reverie.enrichment.handler.{handler.__class__.__name__}") as h_span:
                    try:
                        handler.process(context, self)
                        h_span.set_attribute("handler.name", handler.__class__.__name__)
                    except Exception as e:
                        h_span.set_status(StatusCode.ERROR)
                        h_span.record_exception(e)
                        logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                
            # Suggested Expiration
            if context.importance_score < 5.0:
                from datetime import datetime, timedelta
                context.expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()
                
            # Token Counts
            context.token_count_full = self.count_tokens(context.text)
            context.token_count_abstract = self.count_tokens(context.profile)
            
            span.set_attribute("reverie.importance", context.importance_score)
            span.set_attribute("reverie.memory_type", context.memory_type.value)
            
            return context

    def calculate_importance(self, text: str) -> Dict[str, Any]:
        """Backward compatibility for legacy ingestion calls."""
        ctx = self.enrich(text)
        return {"score": ctx.importance_score, "expires_at": ctx.expires_at}

    def _get_expiration(self, importance: float) -> Optional[str]:
        """Suggests an expiration for low-importance memories."""
        if importance < 5.0:
            from datetime import datetime, timedelta
            return (datetime.utcnow() + timedelta(days=7)).isoformat()
        return None

    def is_structurally_important(self, text: str) -> bool:
        """DEPRECATED: Use HeuristicImportance instead."""
        ctx = EnrichmentContext(text)
        HeuristicImportance().process(ctx, self)
        return ctx.importance_score > 2.0

    def calculate_importance_with_soul(self, text: str, soul_prompt: str) -> Dict[str, Any]:
        """DEPRECATED: Use SoulImportance instead."""
        ctx = EnrichmentContext(text)
        self.set_soul(soul_prompt)
        SoulImportance().process(ctx, self)
        return {"score": ctx.importance_score}
    def classify_type(self, text: str) -> MemoryType:
        """Backward compatibility for legacy classification calls."""
        return self.enrich(text).memory_type

    def extract_query_anchors(self, query: str) -> List[str]:
        """Lighter LLM pass to extract technical entities from a user query."""
        try:
            # Quick check: if the query is very short and non-technical, skip LLM
            if len(query.split()) < 3 and not any(c in query for c in [".", "(", "/", "\\"]):
                return []

            if not self.llm_client.check_connectivity():
                return []

            prompt = (
                "Extract technical entities (Files, Tools, Classes, Repos) from this query. "
                "Return as a JSON list: {\"anchors\": [\"name1\", \"name2\"]}. "
                "If no technical entities, return an empty list."
            )
            
            resp = self.llm_client.call([
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ], telemetry_metadata={"reverie.operation": "QueryAnchoring"})
            
            return resp.get("anchors", [])
        except Exception as e:
            logger.debug(f"Query anchor extraction failed: {e}")
            return []

    def extract_graph_data(self, text: str, memory_id: int, db_manager: Any):
        """
        Two-Pass Extraction Pipeline:
        1. Extract & Resolve Entities.
        2. Extract & Validate Triples using ID references.
        """
        with tracer.start_as_current_span("reverie.graph.extraction") as span:
            span.set_attribute("memory_id", memory_id)
            try:
                # Fail-fast connectivity check
                if not self.llm_client.check_connectivity():
                    logger.warning(f"Extraction skipped for memory {memory_id}: LLM Provider unreachable.")
                    self.telemetry["failure"] += 1
                    span.set_attribute("extraction.skipped", "connectivity")
                    return

                # Pass 1: Extract Entities
                entity_data = self.llm_client.call([
                    {"role": "system", "content": "Extract technical entities (Files, Functions, API Endpoints, Tools). Return JSON: {\"entities\": [{\"name\": \"...\", \"type\": \"...\", \"description\": \"...\"}]}"},
                    {"role": "user", "content": text}
                ], telemetry_metadata={"reverie.graph.stage": "EntityExtraction"})
                
                if not entity_data or "entities" not in entity_data:
                    logger.debug(f"No entities extracted for memory {memory_id}")
                    self.telemetry["failure"] += 1
                    return

                # Idempotency Safeguard: Purge old triples for this memory_id
                db_manager.purge_relations(memory_id)

                # Canonicalize & Store Entities
                entity_map = {} # name -> id
                cursor = db_manager.get_cursor()
                for ent in entity_data["entities"]:
                    name = ent.get("name", "").strip()
                    if not name: continue
                    
                    label = ent.get("type", "UNKNOWN").upper()
                    desc = ent.get("description", "")
                    
                    # Idempotent Insert (UPSERT pattern) with GUID generation
                    new_guid = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO entities (name, label, description, guid) 
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET 
                            label=excluded.label, 
                            description=COALESCE(excluded.description, description),
                            guid=COALESCE(entities.guid, excluded.guid)
                    """, (name, label, desc, new_guid))
                    
                    cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
                    entity_map[name] = cursor.fetchone()[0]
                    
                    # Restore MENTIONS link (Memory -> Entity)
                    cursor.execute("""
                        INSERT INTO memory_relations (source_id, source_type, target_id, target_type, relation_type, evidence_memory_id)
                        VALUES (?, 'MEMORY', ?, 'ENTITY', 'MENTIONS', ?)
                    """, (memory_id, entity_map[name], memory_id))

                db_manager.commit()

                # Pass 2: Extract Triples using Entity names
                # We ask for triples between identified entities
                triple_prompt = f"Entities identified: {list(entity_map.keys())}. \n"
                triple_prompt += f"Relationships allowed: {[t.value for t in RelationType]}. \n"
                triple_prompt += f"Extract triples from text: {text}. \n"
                triple_prompt += "Return JSON: {\"triples\": [{\"source\": \"name\", \"predicate\": \"TYPE\", \"target\": \"name\", \"confidence\": 0.9}]}"
                
                triple_data = self.llm_client.call([
                    {"role": "system", "content": "Extract relationships between technical entities. Use the provided list of entity names and allowed predicates."},
                    {"role": "user", "content": triple_prompt}
                ], telemetry_metadata={"reverie.graph.stage": "TripleExtraction"})

                if not triple_data or "triples" not in triple_data:
                    logger.debug(f"No triples extracted for memory {memory_id}")
                    self.telemetry["success"] += 1 # Partial success (entities saved)
                    return


                # Store Validated Triples
                valid_predicates = {t.value for t in RelationType}
                success_triples = 0
                for t in triple_data["triples"]:
                    src_name = t.get("source")
                    tgt_name = t.get("target")
                    pred = t.get("predicate", "").upper()
                    conf = t.get("confidence", 1.0)
                    
                    if src_name in entity_map and tgt_name in entity_map and pred in valid_predicates:
                        cursor.execute("""
                            INSERT INTO memory_relations (
                                source_id, source_type, target_id, target_type, relation_type, confidence_score, evidence_memory_id
                            ) VALUES (?, 'ENTITY', ?, 'ENTITY', ?, ?, ?)
                        """, (entity_map[src_name], entity_map[tgt_name], pred, conf, memory_id))

                        success_triples += 1
                    else:
                        logger.debug(f"Rejected invalid triple for memory {memory_id}: {t}")

                db_manager.commit()
                self.telemetry["success"] += 1
                logger.info(f"Extraction turn complete for memory {memory_id}: {len(entity_map)} entities, {success_triples} triples. Total: {self.telemetry}")

            except Exception as e:
                self.telemetry["failure"] += 1
                logger.error(f"Graph extraction FAILED for memory {memory_id}: {e}\n{traceback.format_exc()}")
                span.set_status(StatusCode.ERROR)
                span.record_exception(e)