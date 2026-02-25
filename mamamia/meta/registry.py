"""Registry for meta-selection components."""

from typing import Dict, Type

from .base import MetaOptimizer, ScoreCombiner, FitnessEvaluator


_OPTIMIZERS: Dict[str, Type[MetaOptimizer]] = {}
_COMBINERS: Dict[str, Type[ScoreCombiner]] = {}
_FITNESS: Dict[str, Type[FitnessEvaluator]] = {}


def register_optimizer(name: str):
    """Decorator to register a meta-optimizer."""
    def decorator(cls: Type[MetaOptimizer]):
        _OPTIMIZERS[name] = cls
        cls.name = name
        return cls
    return decorator


def register_combiner(name: str):
    """Decorator to register a score combiner."""
    def decorator(cls: Type[ScoreCombiner]):
        _COMBINERS[name] = cls
        cls.name = name
        return cls
    return decorator


def register_fitness(name: str):
    """Decorator to register a fitness evaluator."""
    def decorator(cls: Type[FitnessEvaluator]):
        _FITNESS[name] = cls
        cls.name = name
        return cls
    return decorator


def get_optimizer(name: str) -> Type[MetaOptimizer]:
    """Get optimizer class by name."""
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(_OPTIMIZERS.keys())}")
    return _OPTIMIZERS[name]


def get_combiner(name: str) -> Type[ScoreCombiner]:
    """Get combiner class by name."""
    if name not in _COMBINERS:
        raise ValueError(f"Unknown combiner: {name}. Available: {list(_COMBINERS.keys())}")
    return _COMBINERS[name]


def get_fitness(name: str) -> Type[FitnessEvaluator]:
    """Get fitness evaluator class by name."""
    if name not in _FITNESS:
        raise ValueError(f"Unknown fitness: {name}. Available: {list(_FITNESS.keys())}")
    return _FITNESS[name]


def list_optimizers() -> list:
    """List available optimizers."""
    return list(_OPTIMIZERS.keys())


def list_combiners() -> list:
    """List available combiners."""
    return list(_COMBINERS.keys())


def list_fitness() -> list:
    """List available fitness evaluators."""
    return list(_FITNESS.keys())
