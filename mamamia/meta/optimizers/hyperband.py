"""
Successive Halving accelerated CMA-ES for meta-selection.

Each CMA-ES generation:
1. Generate N candidates
2. Run successive halving (not full training for all)
3. Report results to CMA-ES

Example with N=50, R=100, eta=3:
  Round 0: 50 candidates @ 4 steps
  Round 1: 16 candidates @ 11 steps
  Round 2: 5 candidates @ 33 steps
  Round 3: 1 candidate @ 100 steps
  Total: ~641 steps vs naive 5000 (7.8x speedup)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

try:
    import cma
except ImportError:
    cma = None

from ..base import MetaOptimizer
from ..registry import register_optimizer


@dataclass
class SHCandidate:
    """Candidate being evaluated in successive halving."""
    weights: np.ndarray
    fitness: float = float("-inf")
    stage: int = 0
    steps_used: int = 0


@register_optimizer("successive_halving")
class SuccessiveHalvingCMAES(MetaOptimizer):
    """
    CMA-ES with successive halving for fitness evaluation.

    Instead of running full training for every candidate:
    - Start all candidates with few steps
    - Eliminate bottom (1 - 1/eta) fraction
    - Give survivors more steps
    - Repeat until one remains
    """

    name = "successive_halving"

    def __init__(self, methods: List[str], seed: int = 42):
        if cma is None:
            raise ImportError("pycma required: pip install cma")
        super().__init__(methods, seed)
        self.es = None
        self.R = 100  # Max steps
        self.eta = 3  # Keep 1/eta each round
        self.n_candidates = 50  # Candidates per generation
        self.stage_penalty = 50.0  # Penalty for early elimination
        self.best_weights: Optional[np.ndarray] = None
        self.best_fitness: float = float("-inf")

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize optimizer.

        Config:
            R: Max training steps (default: 100)
            eta: Elimination factor, keep 1/eta (default: 3)
            n_candidates: Candidates per generation (default: 50)
            sigma0: CMA-ES step size (default: 0.3)
            stage_penalty: Penalty per stage missed (default: 50)
        """
        self.R = config.get("R", 100)
        self.eta = config.get("eta", 3)
        self.n_candidates = config.get("n_candidates", 50)
        self.stage_penalty = config.get("stage_penalty", 50.0)
        sigma0 = config.get("sigma0", 0.3)

        # Compute successive halving schedule
        self.schedule = self._compute_schedule()

        # Initialize CMA-ES
        x0 = np.ones(self.n_methods) / self.n_methods
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "seed": self.seed,
            "bounds": [0, 1],
            "popsize": self.n_candidates,
            "verbose": -9,
        })

        print(f"Successive Halving CMA-ES initialized:")
        print(f"  R={self.R}, eta={self.eta}, n_candidates={self.n_candidates}")
        print(f"  Schedule: {self.schedule}")
        total = sum(n * r for n, r in self.schedule)
        naive = self.n_candidates * self.R
        print(f"  Steps per gen: {total} (vs naive {naive}, {naive/total:.1f}x speedup)")

    def _compute_schedule(self) -> List[Tuple[int, int]]:
        """
        Compute (n_configs, n_steps) for each round.

        Returns list of (n, r) tuples where:
        - n = number of candidates this round
        - r = steps to run this round
        """
        n = self.n_candidates
        s = int(math.log(n, self.eta))  # Number of elimination rounds
        r_min = max(1, int(self.R / (self.eta ** s)))

        schedule = []
        for i in range(s + 1):
            n_i = max(1, int(n / (self.eta ** i)))
            r_i = min(self.R, int(r_min * (self.eta ** i)))
            schedule.append((n_i, r_i))

        return schedule

    def run_generation(
        self,
        fitness_fn: Callable[[np.ndarray, int], float],
    ) -> Tuple[np.ndarray, float, List[Dict]]:
        """
        Run one CMA-ES generation with successive halving.

        Args:
            fitness_fn: Function(weights, n_steps) -> fitness

        Returns:
            (best_weights, best_fitness, history)
        """
        # Get candidates from CMA-ES
        solutions = self.es.ask()
        n_pop = len(solutions)

        # Initialize all candidates with their original index
        candidates = []
        for idx, sol in enumerate(solutions):
            w = np.array(sol)
            w = np.maximum(w, 0)
            w = w / w.sum() if w.sum() > 0 else np.ones(self.n_methods) / self.n_methods
            candidates.append(SHCandidate(weights=w))

        # Track final results for each original candidate (by index)
        final_results = [None] * n_pop  # Will store (fitness, stage, weights) for each
        active_indices = list(range(n_pop))  # Track which candidates are still active

        history = []

        # Run successive halving
        for round_idx, (n_keep, n_steps) in enumerate(self.schedule):
            if len(active_indices) == 0:
                break

            print(f"    Round {round_idx}: {len(active_indices)} candidates @ {n_steps} steps")

            # Evaluate each active candidate
            round_results = []
            for i, orig_idx in enumerate(active_indices):
                cand = candidates[orig_idx]
                cand.fitness = fitness_fn(cand.weights, n_steps)
                cand.stage = round_idx
                cand.steps_used = n_steps
                round_results.append((orig_idx, cand.fitness))

                history.append({
                    "weights": cand.weights.tolist(),
                    "fitness": cand.fitness,
                    "stage": round_idx,
                    "steps": n_steps,
                    "candidate_idx": orig_idx,
                })

            # Sort by fitness (descending) and determine survivors
            round_results.sort(key=lambda x: x[1], reverse=True)

            if round_idx < len(self.schedule) - 1:
                next_n = self.schedule[round_idx + 1][0]
                survivors = set(idx for idx, _ in round_results[:next_n])
                eliminated = set(idx for idx, _ in round_results[next_n:])
            else:
                # Final round - everyone "survives" to get their final fitness
                survivors = set(idx for idx, _ in round_results)
                eliminated = set()

            # Record final results for eliminated candidates
            for orig_idx in eliminated:
                cand = candidates[orig_idx]
                final_results[orig_idx] = (cand.fitness, cand.stage, cand.weights.copy())

            # Record final results for survivors in last round
            if round_idx == len(self.schedule) - 1:
                for orig_idx in survivors:
                    cand = candidates[orig_idx]
                    final_results[orig_idx] = (cand.fitness, cand.stage, cand.weights.copy())

            # Update active indices for next round
            active_indices = [idx for idx in active_indices if idx in survivors]

        # Report to CMA-ES: exactly n_pop fitnesses, one per original candidate
        max_stage = len(self.schedule) - 1
        cmaes_fitnesses = []

        for orig_idx, sol in enumerate(solutions):
            if final_results[orig_idx] is not None:
                fitness, stage, weights = final_results[orig_idx]
                # Scale fitness by how far they made it (penalty for early elimination)
                stages_missed = max_stage - stage
                adjusted = fitness - self.stage_penalty * stages_missed
            else:
                # Shouldn't happen, but fallback
                adjusted = float("-inf")

            cmaes_fitnesses.append(-adjusted)  # CMA-ES minimizes

            # Track global best (only from final stage)
            if final_results[orig_idx] is not None:
                fitness, stage, weights = final_results[orig_idx]
                if fitness > self.best_fitness and stage == max_stage:
                    self.best_fitness = fitness
                    self.best_weights = weights.copy()

        self.es.tell(solutions, cmaes_fitnesses)

        # Return best from this generation
        best = max(history, key=lambda h: h["fitness"])
        return np.array(best["weights"]), best["fitness"], history

    def get_best(self) -> Tuple[np.ndarray, float]:
        if self.best_weights is None:
            return np.ones(self.n_methods) / self.n_methods, float("-inf")
        return self.best_weights.copy(), self.best_fitness

    def ask(self) -> np.ndarray:
        """Not used directly - use run_generation instead."""
        raise NotImplementedError("Use run_generation() for successive halving")

    def tell(self, weights: np.ndarray, fitness: float) -> None:
        """Not used directly - use run_generation instead."""
        raise NotImplementedError("Use run_generation() for successive halving")

    def _get_state(self) -> Dict[str, Any]:
        return {
            "best_weights": self.best_weights.tolist() if self.best_weights is not None else None,
            "best_fitness": self.best_fitness,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        if state.get("best_weights"):
            self.best_weights = np.array(state["best_weights"])
        self.best_fitness = state.get("best_fitness", float("-inf"))

    @property
    def stop(self) -> bool:
        return self.es.stop() if self.es else True
