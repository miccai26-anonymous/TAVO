"""Base class for selection methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SelectionResult:
    """Result of a selection method."""
    selected: List[str]
    scores: Dict[str, float]
    method: str
    budget: int
    metadata: Dict


class SelectionMethod(ABC):
    name: str = "base"
    requires_embeddings: bool = False
    requires_gradients: bool = False
    requires_query_set: bool = False

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    @abstractmethod
    def select(
        self,
        pool_ids: List[str],
        budget: int,
        embeddings: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None,
        query_embeddings: Optional[np.ndarray] = None,
        query_gradients: Optional[np.ndarray] = None,
        **kwargs
    ) -> SelectionResult:
        pass

    def save_result(self, result: SelectionResult, output_path: Path) -> None:
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": result.method, "budget": result.budget,
            "selected": result.selected, "scores": result.scores,
            "metadata": result.metadata,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_result(cls, path: Path) -> SelectionResult:
        import json
        with open(path) as f:
            data = json.load(f)
        return SelectionResult(
            selected=data["selected"], scores=data["scores"],
            method=data["method"], budget=data["budget"],
            metadata=data.get("metadata", {}),
        )
