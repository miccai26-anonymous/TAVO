"""
Bayesian Optimization + Successive Halving meta-optimizer.

Each generation:
1. GP + q-EI proposes a batch of candidates (or Sobol for warm-start)
2. Successive Halving filters the batch with graduated step budgets
3. Only final-stage survivors feed back into the GP

This combines BO's sample efficiency with SH's cheap filtering.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False

from ..base import MetaOptimizer
from ..registry import register_optimizer


@register_optimizer("bo_sh")
class BOSHOptimizer(MetaOptimizer):
    """
    Bayesian Optimization with Successive Halving for fitness evaluation.

    Instead of evaluating every candidate at full fidelity:
    - BO proposes a batch of candidates
    - SH runs them through graduated step budgets
    - Only survivors (final stage) update the GP
    - Eliminated candidates are logged but don't pollute the GP
    """

    name = "bo_sh"

    def __init__(self, methods: List[str], seed: int = 42):
        if not HAS_BOTORCH:
            raise ImportError("botorch required: pip install botorch")
        super().__init__(methods, seed)
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None
        self.best_weights: Optional[np.ndarray] = None
        self.best_fitness: float = float("-inf")
        self.R: int = 500
        self.eta: int = 3
        self.n_candidates: int = 50
        self.bounds = torch.tensor([[0.0] * self.n_methods, [1.0] * self.n_methods],
                                   dtype=torch.float64)
        self.generation_count: int = 0
        self.schedule: List[Tuple[int, int]] = []
        self.device = torch.device("cpu")

    def initialize(self, config: Dict[str, Any]) -> None:
        self.R = config.get("R", 500)
        self.eta = config.get("eta", 3)
        self.n_candidates = config.get("n_candidates", 50)
        self.generation_count = 0
        self.train_X = None
        self.train_Y = None
        torch.manual_seed(self.seed)

        self.schedule = self._compute_schedule()

        print(f"BO-SH Optimizer initialized:")
        print(f"  R={self.R}, eta={self.eta}, n_candidates={self.n_candidates}")
        print(f"  Schedule: {self.schedule}")
        total = sum(n * r for n, r in self.schedule)
        naive = self.n_candidates * self.R
        print(f"  Steps per gen: {total} (vs naive {naive}, {naive / total:.1f}x speedup)")

    def _compute_schedule(self) -> List[Tuple[int, int]]:
        """Compute (n_configs, n_steps) for each SH round."""
        n = self.n_candidates
        s = int(math.log(n, self.eta))
        r_min = max(1, int(self.R / (self.eta ** s)))

        schedule = []
        for i in range(s + 1):
            n_i = max(1, int(n / (self.eta ** i)))
            r_i = min(self.R, int(r_min * (self.eta ** i)))
            schedule.append((n_i, r_i))

        return schedule

    def _propose_candidates(self) -> np.ndarray:
        """Propose n_candidates weight vectors."""
        if self.train_X is None or len(self.train_X) < 2 * self.n_methods:
            # Sobol warm-start
            sobol = torch.quasirandom.SobolEngine(
                dimension=self.n_methods, scramble=True,
                seed=self.seed + self.generation_count
            )
            raw = sobol.draw(self.n_candidates).to(dtype=torch.float64)
            return raw.numpy()
        else:
            return self._bo_candidates()

    def _bo_candidates(self) -> np.ndarray:
        """Use GP + q-EI to propose candidates."""
        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        best_f = self.train_Y.max().item()
        acqf = qExpectedImprovement(model=model, best_f=best_f)

        # BoTorch q-EI with large batch: optimize in chunks if needed
        q = min(self.n_candidates, 20)  # q-EI batch limit
        all_candidates = []

        while len(all_candidates) < self.n_candidates:
            remaining = self.n_candidates - len(all_candidates)
            batch_q = min(q, remaining)

            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self.bounds,
                q=batch_q,
                num_restarts=10,
                raw_samples=256,
            )
            all_candidates.extend(candidates.detach().numpy())

        return np.array(all_candidates[:self.n_candidates])

    def run_generation(
        self,
        fitness_fn: Callable[[np.ndarray, int], float],
    ) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        Run one BO generation with successive halving.

        Args:
            fitness_fn: Function(weights, n_steps) -> fitness

        Returns:
            (best_weights, best_fitness, history)
        """
        self.generation_count += 1

        # Propose candidates
        raw_candidates = self._propose_candidates()

        # Normalize weights
        candidates = []
        for w in raw_candidates:
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(self.n_methods) / self.n_methods
            candidates.append(w)

        # Track all evaluations
        history = []
        active_indices = list(range(len(candidates)))

        # Run successive halving
        for round_idx, (n_keep, n_steps) in enumerate(self.schedule):
            if len(active_indices) == 0:
                break

            print(f"    Round {round_idx}: {len(active_indices)} candidates @ {n_steps} steps")

            # Evaluate active candidates
            round_results = []
            for orig_idx in active_indices:
                w = candidates[orig_idx]
                fitness = fitness_fn(w, n_steps)
                round_results.append((orig_idx, fitness))

                history.append({
                    "weights": w.tolist(),
                    "fitness": fitness,
                    "stage": round_idx,
                    "steps": n_steps,
                    "candidate_idx": orig_idx,
                })

            # Sort by fitness descending
            round_results.sort(key=lambda x: x[1], reverse=True)

            # Determine survivors
            if round_idx < len(self.schedule) - 1:
                next_n = self.schedule[round_idx + 1][0]
                survivors = set(idx for idx, _ in round_results[:next_n])
            else:
                survivors = set(idx for idx, _ in round_results)

            active_indices = [idx for idx in active_indices if idx in survivors]

        # Feed only final-stage survivors to the GP
        final_stage = len(self.schedule) - 1
        survivor_entries = [h for h in history if h["stage"] == final_stage]

        if survivor_entries:
            new_X = torch.tensor(
                [h["weights"] for h in survivor_entries], dtype=torch.float64
            )
            new_Y = torch.tensor(
                [[h["fitness"]] for h in survivor_entries], dtype=torch.float64
            )

            if self.train_X is None:
                self.train_X = new_X
                self.train_Y = new_Y
            else:
                self.train_X = torch.cat([self.train_X, new_X], dim=0)
                self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)

        # Track best
        for h in survivor_entries:
            if h["fitness"] > self.best_fitness:
                self.best_fitness = h["fitness"]
                self.best_weights = np.array(h["weights"]).copy()

        # Return best from this generation
        if history:
            best = max(history, key=lambda h: h["fitness"])
            return np.array(best["weights"]), best["fitness"], history
        else:
            return np.ones(self.n_methods) / self.n_methods, float("-inf"), history

    def get_best(self) -> Tuple[np.ndarray, float]:
        if self.best_weights is None:
            return np.ones(self.n_methods) / self.n_methods, float("-inf")
        return self.best_weights.copy(), self.best_fitness

    def ask(self) -> np.ndarray:
        raise NotImplementedError("Use run_generation() for BO-SH")

    def tell(self, weights: np.ndarray, fitness: float) -> None:
        raise NotImplementedError("Use run_generation() for BO-SH")

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
