import logging
import os
import json
import re
import urllib.request
import traceback
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .schemas import MemoryType, AssociationType

logger = logging.getLogger(__name__)

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

    def check_connectivity(self) -> bool:
        """Fast-fail check to see if the LLM provider is reachable (2s timeout)."""
        url = f"{self.base_url}/models"
        # Using GET instead of HEAD because some providers (llama.cpp/llama-swap) 
        # may only implement GET for the models endpoint.
        req = urllib.request.Request(url, method="GET")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
            
        try:
            with urllib.request.urlopen(req, timeout=2.0) as _:
                return True
        except Exception as e:
            logger.warning(f"LLM Provider Connectivity Check FAILED for {self.base_url}: {e}")
            return False

    def call(self, messages: List[Dict[str, str]], json_mode: bool = True) -> Optional[Dict[str, Any]]:
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
                return json.loads(content) if json_mode else content
        except Exception as e:
            logger.error(f"InternalLLMClient.call FAILED: {e}")
            return None

class EnrichmentService:
    """The Intelligence Layer: Handles embeddings, BART classification, and profiling."""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", 
                 summarization_model_name: str = "sshleifer/distilbart-cnn-12-6",
                 classifier_model_name: str = "facebook/bart-large-mnli"):
        self.embedding_model_name = embedding_model_name
        self.summarization_model_name = summarization_model_name
        self.classifier_model_name = classifier_model_name
        
        # Models initialized as None (Lazy-Loading)
        self.embedding_model = None
        self.summarizer = None
        self.summarizer_tokenizer = None
        
        self.classifier_model = None
        self.classifier_tokenizer = None

        # LLM Client for Graph Extraction
        cfg = ConfigLoader.load_config()
        
        # Priority Logic for Provider Selection:
        # 1. Use root-level 'model' section (Hermes default)
        # 2. Use 'custom_providers' catalog
        # 3. Fallback to localhost
        
        model_cfg = cfg.get("model", {})
        providers = cfg.get("providers", [])
        
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
        
        # Telemetry
        self.telemetry = {"success": 0, "failure": 0}
        
        logger.info(f"EnrichmentService initialized. LLM Source: {self.llm_client.base_url}, Model: {self.llm_client.model_name}")

    def _ensure_loaded(self, models: List[str]):
        """Lazy loader for specific model backends."""
        if "embedding" in models and self.embedding_model is None:
            logger.debug(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        if "summarizer" in models and self.summarizer is None:
            logger.debug(f"Loading summarization model: {self.summarization_model_name}")
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
            self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(
                self.summarization_model_name,
                low_cpu_mem_usage=False
            ).to("cpu")

        if "classifier" in models and self.classifier_model is None:
            logger.debug(f"Loading zero-shot classifier: {self.classifier_model_name} (This may take a moment)")
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_name)
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                self.classifier_model_name,
                low_cpu_mem_usage=False
            ).to("cpu")
            logger.debug("BART Classifier loaded successfully.")

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
            return self.summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Profiling failed: {e}")
            return text[:200] + "..."

    def _zero_shot_classify(self, text: str, labels: List[str], hypothesis_template: str = "This example is {}.") -> Dict[str, float]:
        """Manual implementation of zero-shot classification for BART MNLI."""
        self._ensure_loaded(["classifier"])
        
        scores = {}
        for label in labels:
            hypothesis = hypothesis_template.format(label)
            
            # BART MNLI expects: [CLS] text [SEP] [SEP] hypothesis [SEP]
            # AutoTokenizer handles the specific formatting for the model
            inputs = self.classifier_tokenizer(text, hypothesis, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                logits = self.classifier_model(**inputs).logits
            
            # Index 0: contradiction, Index 1: neutral, Index 2: entailment
            # We care about the entailment vs contradiction relationship
            entail_contr_logits = logits[:, [0, 2]]
            probs = F.softmax(entail_contr_logits, dim=1)
            scores[label] = probs[0, 1].item()
            
        return scores

    def calculate_importance(self, text: str) -> float:
        """Uses BART to weigh the importance of a memory on a 1.0-5.0 scale."""
        try:
            # We measure entailment against 'important' and 'critical' vs 'minor'
            labels = ["critical", "important", "minor", "trivial"]
            scores = self._zero_shot_classify(text, labels, "This information is {}.")
            
            # Calculate a weighted score
            # Highest weight for critical/important
            raw_score = (scores["critical"] * 5.0) + (scores["important"] * 4.0) + (scores["minor"] * 2.0) + (scores["trivial"] * 1.0)
            
            # Normalize to 1.0-5.0 range
            # Note: BART scores are probabilities, so critical=1.0 would give 5.0
            return max(1.0, min(5.0, raw_score))
        except Exception as e:
            logger.warning(f"Importance scoring failed: {e}")
            return 1.0

    def classify_type(self, text: str) -> MemoryType:
        """Robust zero-shot classification using BART."""
        try:
            mapping = {
                "error, exception, crash": MemoryType.RUNTIME_ERROR,
                "source code, programming, snippet": MemoryType.CODE_SNIPPET,
                "task, goal, action item": MemoryType.TASK,
                "user preference, personalization": MemoryType.USER_PREFERENCE,
                "learning, discovery, insight": MemoryType.LEARNING_EVENT,
                "observation, fact, status": MemoryType.OBSERVATION,
                "expired task, overdue": MemoryType.EXPIRED_TASK,
                "conversation, dialogue, chat": MemoryType.CONVERSATION
            }
            
            labels = list(mapping.keys())
            scores = self._zero_shot_classify(text, labels, "This text is about {}.")
            
            # Pick the label with the highest entailment score
            best_label = max(scores, key=scores.get)
            return mapping[best_label]
        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}. Falling back to CONVERSATION.")
            return MemoryType.CONVERSATION

    def extract_graph_data(self, text: str, memory_id: int, db_manager: Any):
        """
        Two-Pass Extraction Pipeline:
        1. Extract & Resolve Entities.
        2. Extract & Validate Triples using ID references.
        """
        try:
            # Fail-fast connectivity check
            if not self.llm_client.check_connectivity():
                logger.warning(f"Extraction skipped for memory {memory_id}: LLM Provider unreachable.")
                self.telemetry["failure"] += 1
                return

            # Pass 1: Extract Entities
            entity_data = self.llm_client.call([
                {"role": "system", "content": "Extract technical entities (Files, Functions, API Endpoints, Tools). Return JSON: {\"entities\": [{\"name\": \"...\", \"type\": \"...\", \"description\": \"...\"}]}"},
                {"role": "user", "content": text}
            ])
            
            if not entity_data or "entities" not in entity_data:
                logger.debug(f"No entities extracted for memory {memory_id}")
                self.telemetry["failure"] += 1
                return

            # Canonicalize & Store Entities
            entity_map = {} # name -> id
            cursor = db_manager.get_cursor()
            for ent in entity_data["entities"]:
                name = ent.get("name", "").strip()
                if not name: continue
                
                label = ent.get("type", "UNKNOWN").upper()
                desc = ent.get("description", "")
                
                # Idempotent Insert (UPSERT pattern)
                cursor.execute("""
                    INSERT INTO entities (name, label, description) 
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET 
                        label=excluded.label, 
                        description=COALESCE(excluded.description, description)
                """, (name, label, desc))
                
                cursor.execute("SELECT id FROM entities WHERE name = ?", (name,))
                entity_map[name] = cursor.fetchone()[0]

            # Pass 2: Extract Triples using Entity names
            # We ask for triples between identified entities
            triple_prompt = f"Entities identified: {list(entity_map.keys())}. \n"
            triple_prompt += f"Relationships allowed: {[t.value for t in AssociationType]}. \n"
            triple_prompt += f"Extract triples from text: {text}. \n"
            triple_prompt += "Return JSON: {\"triples\": [{\"source\": \"name\", \"predicate\": \"TYPE\", \"target\": \"name\", \"confidence\": 0.9}]}"
            
            triple_data = self.llm_client.call([
                {"role": "system", "content": "Extract relationships between technical entities. Use the provided list of entity names and allowed predicates."},
                {"role": "user", "content": triple_prompt}
            ])

            if not triple_data or "triples" not in triple_data:
                logger.debug(f"No triples extracted for memory {memory_id}")
                self.telemetry["success"] += 1 # Partial success (entities saved)
                return

            # Idempotency Safeguard: Purge old triples for this memory_id
            db_manager.purge_associations(memory_id)

            # Store Validated Triples
            valid_predicates = {t.value for t in AssociationType}
            success_triples = 0
            for t in triple_data["triples"]:
                src_name = t.get("source")
                tgt_name = t.get("target")
                pred = t.get("predicate", "").upper()
                conf = t.get("confidence", 1.0)
                
                if src_name in entity_map and tgt_name in entity_map and pred in valid_predicates:
                    cursor.execute("""
                        INSERT INTO memory_associations (
                            source_id, source_type, 
                            target_id, target_type, 
                            association_type, confidence_score, 
                            evidence_memory_id
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