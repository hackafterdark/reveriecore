import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from .schemas import MemoryType

logger = logging.getLogger(__name__)

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
        
        logger.info("EnrichmentService initialized (Lazy). BART is ready for zero-shot tasks.")

    def _ensure_loaded(self, models: List[str]):
        """Lazy loader for specific model backends."""
        if "embedding" in models and self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        if "summarizer" in models and self.summarizer is None:
            logger.info(f"Loading summarization model: {self.summarization_model_name}")
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
            self.summarizer = AutoModelForSeq2SeqLM.from_pretrained(
                self.summarization_model_name,
                low_cpu_mem_usage=False
            ).to("cpu")

        if "classifier" in models and self.classifier_model is None:
            logger.info(f"Loading zero-shot classifier: {self.classifier_model_name} (This may take a moment)")
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_name)
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                self.classifier_model_name,
                low_cpu_mem_usage=False
            ).to("cpu")
            logger.info("BART Classifier loaded successfully.")

    def generate_embedding(self, text: str) -> List[float]:
        try:
            self._ensure_loaded(["embedding"])
            return self.embedding_model.encode([text])[0].tolist()
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