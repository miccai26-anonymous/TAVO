"""
Bayesian Optimization meta-optimizer for selection method weights.

Uses BoTorch (GP + q-EI) to efficiently search the weight space.
Sobol quasi-random warm-start for the first generation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False

from ..base import MetaOptimizer
from ..registry import register_optimizer


@register_optimizer("bayesian")
class BayesianOptimizer(MetaOptimizer):
    """
    Bayesian Optimization using GP surrogate + q-Expected Improvement.

    Each generation:
    1. Fit GP to all observed (weights, fitness) pairs
    2. Optimize q-EI to get a batch of candidates
    3. Evaluate each candidate at full fidelity
    4. Update observations

    First generation uses Sobol quasi-random sampling.
    """

    name = "bayesian"

    def __init__(self, methods: List[str], seed: int = 42):
        if not HAS_BOTORCH:
            raise ImportError("botorch required: pip install botorch")
        super().__init__(methods, seed)
        self.train_X: Optional[torch.Tensor] = None  # (n_obs, n_methods)
        self.train_Y: Optional[torch.Tensor] = None  # (n_obs, 1)
        self.best_weights: Optional[np.ndarray] = None
        self.best_fitness: float = float("-inf")
        self.batch_size: int = 8
        self.bounds = torch.tensor([[0.0] * self.n_methods, [1.0] * self.n_methods],
                                   dtype=torch.float64)
        self.generation_count: int = 0
        self.device = torch.device("cpu")

    def initialize(self, config: Dict[str, Any]) -> None:
        self.batch_size = config.get("batch_size", 8)
        self.generation_count = 0
        self.train_X = None
        self.train_Y = None
        torch.manual_seed(self.seed)

        print(f"Bayesian Optimizer initialized:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Dimensions: {self.n_methods}")

    def ask_batch(self) -> List[np.ndarray]:
        """Get a batch of candidate weight vectors."""
        self.generation_count += 1

        if self.train_X is None or len(self.train_X) < 2 * self.n_methods:
            # Warm-start: Sobol quasi-random
            sobol = torch.quasirandom.SobolEngine(dimension=self.n_methods, scramble=True,
                                                   seed=self.seed + self.generation_count)
            raw = sobol.draw(self.batch_size).to(dtype=torch.float64)
            candidates = raw.numpy()
        else:
            # Fit GP and optimize acquisition
            candidates = self._bo_candidates()

        # Normalize weights to sum to 1
        result = []
        for w in candidates:
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(self.n_methods) / self.n_methods
            result.append(w)

        return result

    def _bo_candidates(self) -> np.ndarray:
        """Use GP + q-EI to propose candidates."""
        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = self.train_Y.max().item()
        acqf = qExpectedImprovement(model=model, best_f=best_f)

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=256,
        )

        return candidates.detach().numpy()

    def tell_batch(self, weights_list: List[np.ndarray], fitnesses: List[float]) -> None:
        """Report a batch of (weights, fitness) results."""
        new_X = torch.tensor(np.array(weights_list), dtype=torch.float64)
        new_Y = torch.tensor([[f] for f in fitnesses], dtype=torch.float64)

        if self.train_X is None:
            self.train_X = new_X
            self.train_Y = new_Y
        else:
            self.train_X = torch.cat([self.train_X, new_X], dim=0)
            self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)

        for w, f in zip(weights_list, fitnesses):
            self.history.append({
                "weights": w.tolist(),
                "fitness": f,
                "generation": self.generation_count,
            })
            if f > self.best_fitness:
                self.best_fitness = f
                self.best_weights = np.array(w).copy()

    def ask(self) -> np.ndarray:
        """Single candidate — use ask_batch for BO."""
        return self.ask_batch()[0]

    def tell(self, weights: np.ndarray, fitness: float) -> None:
        """Single result — use tell_batch for BO."""
        self.tell_batch([weights], [fitness])

    def get_best(self) -> Tuple[np.ndarray, float]:
        if self.best_weights is None:
            return np.ones(self.n_methods) / self.n_methods, float("-inf")
        return self.best_weights.copy(), self.best_fitness

    def _get_state(self) -> Dict[str, Any]:
        return {
            "best_weights": self.best_weights.tolist() if self.best_weights is not None else None,
            "best_fitness": self.best_fitness,
            "generation": self.generation_count,
            "train_X": self.train_X.tolist() if self.train_X is not None else None,
            "train_Y": self.train_Y.tolist() if self.train_Y is not None else None,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        if state.get("best_weights"):
            self.best_weights = np.array(state["best_weights"])
        self.best_fitness = state.get("best_fitness", float("-inf"))
        self.generation_count = state.get("generation", 0)
        if state.get("train_X"):
            self.train_X = torch.tensor(state["train_X"], dtype=torch.float64)
            self.train_Y = torch.tensor(state["train_Y"], dtype=torch.float64)

    @property
    def stop(self) -> bool:
        return False
