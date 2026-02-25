"""CMA-ES meta-optimizer for selection method weights."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cma
except ImportError:
    cma = None

from ..base import MetaOptimizer
from ..registry import register_optimizer


@register_optimizer("cmaes")
class CMAESOptimizer(MetaOptimizer):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

    Optimizes weights for combining selection methods using CMA-ES,
    a derivative-free optimization algorithm well-suited for
    continuous optimization problems.
    """

    name = "cmaes"

    def __init__(self, methods: List[str], seed: int = 42):
        if cma is None:
            raise ImportError("pycma not installed. Run: pip install cma")
        super().__init__(methods, seed)
        self.es: Optional[cma.CMAEvolutionStrategy] = None
        self.current_solutions: List[np.ndarray] = []
        self.current_idx: int = 0
        self.best_weights: Optional[np.ndarray] = None
        self.best_fitness: float = float("-inf")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize CMA-ES.

        Config options:
            sigma0: Initial step size (default: 0.3)
            popsize: Population size (default: 4 + 3*log(n))
            maxiter: Maximum iterations (default: 100)
            bounds: Parameter bounds (default: [0, 1])
        """
        sigma0 = config.get("sigma0", 0.3)
        popsize = config.get("popsize", None)
        maxiter = config.get("maxiter", 100)
        bounds = config.get("bounds", [0, 1])

        # Start from uniform weights
        x0 = np.ones(self.n_methods) / self.n_methods

        opts = {
            "seed": self.seed,
            "bounds": bounds,
            "maxiter": maxiter,
            "verbose": -9,  # Suppress output
        }
        if popsize is not None:
            opts["popsize"] = popsize

        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        self.current_solutions = []
        self.current_idx = 0

    def ask(self) -> np.ndarray:
        """Get next candidate weights."""
        if not self.current_solutions or self.current_idx >= len(self.current_solutions):
            # Generate new population
            self.current_solutions = self.es.ask()
            self.current_idx = 0

        weights = np.array(self.current_solutions[self.current_idx])
        self.current_idx += 1

        # Ensure non-negative and normalize
        weights = np.maximum(weights, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.n_methods) / self.n_methods

        return weights

    def tell(self, weights: np.ndarray, fitness: float) -> None:
        """Report fitness (CMA-ES minimizes, so negate)."""
        self.history.append({
            "weights": weights.tolist(),
            "fitness": fitness,
            "generation": self.es.countiter if self.es else 0,
        })

        # Track best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_weights = weights.copy()

        # If we've evaluated all solutions in population, tell CMA-ES
        if self.current_idx >= len(self.current_solutions):
            # Get fitnesses for this generation (negate for minimization)
            n_pop = len(self.current_solutions)
            recent = self.history[-n_pop:]
            fitnesses = [-h["fitness"] for h in recent]
            self.es.tell(self.current_solutions, fitnesses)

    def get_best(self) -> Tuple[np.ndarray, float]:
        """Return best weights and fitness."""
        if self.best_weights is None:
            return np.ones(self.n_methods) / self.n_methods, float("-inf")
        return self.best_weights.copy(), self.best_fitness

    def _get_state(self) -> Dict[str, Any]:
        """Get CMA-ES specific state."""
        return {
            "best_weights": self.best_weights.tolist() if self.best_weights is not None else None,
            "best_fitness": self.best_fitness,
            "generation": self.es.countiter if self.es else 0,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Restore CMA-ES specific state."""
        if state.get("best_weights"):
            self.best_weights = np.array(state["best_weights"])
        self.best_fitness = state.get("best_fitness", float("-inf"))

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self.es.countiter if self.es else 0

    @property
    def stop(self) -> bool:
        """Check if optimization should stop."""
        return self.es.stop() if self.es else True
