from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

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
