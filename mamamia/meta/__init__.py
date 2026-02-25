"""Meta-selection framework for optimizing selection method combinations."""

from .base import MetaOptimizer, ScoreCombiner, FitnessEvaluator, MetaResult
from .registry import (
    register_optimizer,
    register_combiner,
    register_fitness,
    get_optimizer,
    get_combiner,
    get_fitness,
    list_optimizers,
    list_combiners,
    list_fitness,
)

# Import implementations to register them
from .optimizers.cmaes import CMAESOptimizer
from .optimizers.hyperband import SuccessiveHalvingCMAES
from .optimizers.bayesian import BayesianOptimizer
from .optimizers.bo_sh import BOSHOptimizer
from .combiners.weighted import WeightedCombiner
from .fitness.median_val import MedianValFitnessEvaluator
from .fitness.best_val import BestValFitnessEvaluator
from .fitness.final_val import FinalValFitnessEvaluator

__all__ = [
    "MetaOptimizer",
    "ScoreCombiner",
    "FitnessEvaluator",
    "MetaResult",
    "register_optimizer",
    "register_combiner",
    "register_fitness",
    "get_optimizer",
    "get_combiner",
    "get_fitness",
    "list_optimizers",
    "list_combiners",
    "list_fitness",
    "CMAESOptimizer",
    "WeightedCombiner",
    "MedianValFitnessEvaluator",
]
