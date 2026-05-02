from enum import Enum
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import math
import logging
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# --- Core Enums ---

class MemoryType(Enum):
    """Classification for how a memory is processed and retrieved."""
    CONVERSATION = "CONVERSATION"
    TASK = "TASK"
    OBSERVATION = "OBSERVATION"
    USER_PREFERENCE = "USER_PREFERENCE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    CODE_SNIPPET = "CODE_SNIPPET"
    LEARNING_EVENT = "LEARNING_EVENT"
    EXPIRED_TASK = "EXPIRED_TASK"
    ENTITY = "ENTITY"

class RetrievalIntent(Enum):
    """Categorical user intent to guide graph traversal."""
    CAUSAL = "CAUSAL"
    PROCEDURAL = "PROCEDURAL"
    DESCRIPTIVE = "DESCRIPTIVE"

class RelationType(Enum):
    """Categorical relationships between two memory nodes."""
    PRECEDES = "PRECEDES"
    SUPPORTS = "SUPPORTS"
    RELATED_TO = "RELATED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    FOLLOWS = "FOLLOWS"
    CAUSES = "CAUSES"
    CONTRADICTS = "CONTRADICTS"
    PREREQUISITE_FOR = "PREREQUISITE_FOR"
    IS_EXAMPLE_OF = "IS_EXAMPLE_OF"
    FIXES = "FIXES"
    PART_OF = "PART_OF"
    IS_A = "IS_A"
    MENTIONS = "MENTIONS"
    DEFINES = "DEFINES"
    DEFINED_IN = "DEFINED_IN"
    CHILD_OF = "CHILD_OF"
    SUPERSEDES = "SUPERSEDES"

# --- Context and Base Handler ---

class RetrievalContext:
    """Mutable state container for the retrieval pipeline."""
    def __init__(self, query_text: str, query_vector: List[float], limit: int, token_budget: int, config: Dict[str, Any] = None, env: Optional[Any] = None):
        self.query_text = query_text
        self.query_vector = query_vector
        self.limit = limit
        self.token_budget = token_budget
        self.config = config or {}
        self.env = env
        
        self.candidates: Dict[int, Dict[str, Any]] = {} # id -> memory_dict
        self.metrics: Dict[str, Any] = {} # telemetry
        self.weights: Dict[str, float] = {"similarity": 0.5, "importance": 0.3, "decay": 0.2}
        self.intent: str = "Exploration"
        self.consumed_tokens: int = 0
        self.results: List[Dict[str, Any]] = []
        self.is_fresh: bool = False
        self.anchors: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.include_ids: bool = True

    @property
    def remaining_budget(self) -> int:
        return self.token_budget - self.consumed_tokens

class RetrievalHandler(ABC):
    """Abstract base class for all retrieval pipeline handlers."""
    def __init__(self, config: Optional[Any] = None):
        self.config = config

    @abstractmethod
    def process(self, context: RetrievalContext, retriever: Any) -> None:
        pass

# --- Configuration Models ---

class AnchoringConfig(BaseModel):
    clean_slate_keywords: List[str] = Field(default_factory=lambda: ["clean slate", "new idea", "fresh start", "fresh project"])

class VectorConfig(BaseModel):
    precision_gate: float = Field(default=0.45, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=3, ge=1)
    fallback_threshold: int = Field(default=3, ge=0)

class GraphExpansionConfig(BaseModel):
    seed_limit: int = Field(default=3, ge=1)
    min_signal: float = Field(default=0.6, ge=0.0, le=1.0)
    discovery_boost: float = Field(default=0.5, ge=0.0, le=1.0)

class IntentClassifierConfig(BaseModel):
    mappings: Dict[str, List[str]] = Field(default_factory=lambda: {
        "troubleshooting and root cause analysis": ["CAUSES", "DEPENDS_ON", "SUPPORTS"],
        "step-by-step instructions and prerequisites": ["PRECEDES", "FOLLOWS", "PREREQUISITE_FOR"],
        "general definition and conceptual mapping": ["IS_A", "PART_OF", "DEFINES", "MENTIONS"]
    })
    intent_strategy: str = Field(default="binary")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

class DiscoveryConfig(BaseModel):
    default_limit: int = Field(default=5, ge=1)
    anchoring: AnchoringConfig = Field(default_factory=AnchoringConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    graph_expansion: GraphExpansionConfig = Field(default_factory=GraphExpansionConfig)
    intent_classifier: IntentClassifierConfig = Field(default_factory=IntentClassifierConfig)

class IntentWeights(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)
    importance: float = Field(..., ge=0.0, le=1.0)
    decay: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_sum(self) -> 'IntentWeights':
        total = self.similarity + self.importance + self.decay
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            logger.warning(f"Intent weights sum to {total}, not 1.0. Normalizing.")
            self.similarity /= total
            self.importance /= total
            self.decay /= total
        return self

class IntentConfig(BaseModel):
    fact_markers: List[str] = Field(default_factory=lambda: ["what is", "how ", "who ", "where ", "when ", "why ", "list ", "explain ", "identify"])
    weights: Dict[str, IntentWeights] = Field(default_factory=lambda: {
        "fact_seeking": IntentWeights(similarity=0.7, importance=0.1, decay=0.2),
        "exploration": IntentWeights(similarity=0.4, importance=0.4, decay=0.2)
    })

class ScoringConfig(BaseModel):
    anchor_boost: float = Field(default=0.2, ge=0.0, le=1.0)
    graph_boost_multiplier: float = Field(default=0.1, ge=0.0, le=1.0)
    default_similarities: Dict[str, float] = Field(default_factory=lambda: {"anchor": 0.6, "other": 0.4})

class DecayConfig(BaseModel):
    half_life_hours: float = Field(default=48.0, gt=0.0)
    min_decay: float = Field(default=0.1, ge=0.0, le=1.0)

class RankingConfig(BaseModel):
    intent: IntentConfig = Field(default_factory=IntentConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)

class BudgetConfig(BaseModel):
    relevance_floor: float = Field(default=0.2, ge=0.0, le=1.0)
    default_token_budget: int = Field(default=1000, ge=1)
    labels: Dict[str, float] = Field(default_factory=lambda: {"critical": 8.0, "relevant": 4.0})

class PruningConfig(BaseModel):
    top_n: int = Field(default=3, ge=1)
    relative_threshold: float = Field(default=0.0, ge=0.0, le=1.0) # Passive by default
    min_absolute_score: float = Field(default=0.3, ge=0.0, le=1.0)

class PipelineConfig(BaseModel):
    discovery: List[str] = Field(default_factory=lambda: ["intent_classifier", "anchoring", "vector"])
    ranking: List[str] = Field(default_factory=lambda: ["intent", "graph_expansion", "scoring", "rerank", "pruning"])
    budget: List[str] = Field(default_factory=lambda: ["budget"])

class RewriterConfig(BaseModel):
    model_path: str = Field(default="models/Phi-3-mini-4k-instruct-q4.gguf")
    device: str = Field(default="cpu")
    threads: int = Field(default=2, ge=1)
    max_words: int = Field(default=50, ge=1)

class RerankConfig(BaseModel):
    model_name: str = Field(default="ms-marco-MiniLM-L-12-v2")
    rerank_boost: float = Field(default=1.0, ge=0.0)

class RetrievalConfig(BaseModel):
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    rewriter: RewriterConfig = Field(default_factory=RewriterConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalConfig':
        """Safely build config with Pydantic validation."""
        r_data = data.get("retrieval", {})
        try:
            return cls(**r_data)
        except Exception as e:
            logger.warning(f"Invalid retrieval configuration in YAML: {e}. Using defaults.")
            return cls()

class MesaConfig(BaseModel):
    pipeline: List[str] = Field(default_factory=lambda: ["soft_prune", "consolidate", "deep_clean"])
    dry_run: bool = Field(default=False)
    interval_seconds: int = Field(default=3600, ge=60)
    centrality_threshold: int = Field(default=2, ge=0)
    retention_days: int = Field(default=14, ge=0)
    importance_cutoff: float = Field(default=4.0, ge=0.0, le=10.0)
    consolidation_threshold: int = Field(default=5, ge=2)
    purge_enabled: bool = Field(default=True)
    deep_clean_interval_days: int = Field(default=30, ge=1)
    archive_retention_days: int = Field(default=90, ge=1)

class MaintenanceConfig(BaseModel):
    mesa: MesaConfig = Field(default_factory=MesaConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaintenanceConfig':
        """Safely build maintenance config with Pydantic validation."""
        m_data = data.get("maintenance", {})
        try:
            return cls(**m_data)
        except Exception as e:
            logger.warning(f"Invalid maintenance config in YAML. Using defaults. Error: {e}")
            return cls()
