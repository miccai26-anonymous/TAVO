"""Base classes for meta-selection framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MetaResult:
    """Result from meta-optimizer."""
    best_weights: np.ndarray
    best_fitness: float
    methods: List[str]
    history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class MetaOptimizer(ABC):
    """
    Abstract base class for meta-selection optimizers.

    Meta-optimizers find optimal weights for combining multiple
    selection methods based on a fitness signal.
    """

    name: str = "base"

    def __init__(self, methods: List[str], seed: int = 42):
        """
        Args:
            methods: List of selection method names to combine
            seed: Random seed for reproducibility
        """
        self.methods = methods
        self.n_methods = len(methods)
        self.seed = seed
        self.history: List[Dict[str, Any]] = []
        np.random.seed(seed)

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the optimizer with configuration."""
        pass

    @abstractmethod
    def ask(self) -> np.ndarray:
        """
        Get next candidate weights to evaluate.

        Returns:
            weights: Array of shape (n_methods,) with non-negative weights
        """
        pass

    @abstractmethod
    def tell(self, weights: np.ndarray, fitness: float) -> None:
        """
        Report fitness for evaluated weights.

        Args:
            weights: The weights that were evaluated
            fitness: The fitness value (higher is better)
        """
        pass

    @abstractmethod
    def get_best(self) -> Tuple[np.ndarray, float]:
        """
        Return best weights and fitness found so far.

        Returns:
            (best_weights, best_fitness)
        """
        pass

    def save_state(self, path: Path) -> None:
        """Save optimizer state to file."""
        import json

        state = {
            "name": self.name,
            "methods": self.methods,
            "seed": self.seed,
            "history": self.history,
        }
        state.update(self._get_state())

        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    def load_state(self, path: Path) -> None:
        """Load optimizer state from file."""
        import json

        with open(path) as f:
            state = json.load(f)

        self.methods = state["methods"]
        self.n_methods = len(self.methods)
        self.seed = state["seed"]
        self.history = state["history"]
        self._set_state(state)

    def _get_state(self) -> Dict[str, Any]:
        """Get optimizer-specific state for saving. Override in subclasses."""
        return {}

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set optimizer-specific state after loading. Override in subclasses."""
        pass


class ScoreCombiner(ABC):
    """
    Abstract base class for combining selection method scores.
    """

    name: str = "base"

    @abstractmethod
    def combine(
        self,
        scores: Dict[str, Dict[str, float]],  # method -> (sample_id -> score)
        weights: np.ndarray,
        methods: List[str],
    ) -> Dict[str, float]:
        """
        Combine scores from multiple methods.

        Args:
            scores: Dict mapping method name to dict of sample scores
            weights: Weights for each method
            methods: List of method names (defines weight order)

        Returns:
            Combined scores for each sample
        """
        pass


class FitnessEvaluator(ABC):
    """
    Abstract base class for evaluating fitness of a selection.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        selected_cases: List[str],
        **kwargs
    ) -> float:
        """
        Evaluate fitness of a selection.

        Args:
            selected_cases: List of selected case IDs
            **kwargs: Additional arguments (e.g., training config)

        Returns:
            Fitness value (higher is better)
        """
        pass
