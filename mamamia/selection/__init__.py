"""Data selection methods."""

from .base import SelectionMethod, SelectionResult

SELECTION_METHODS = {}


def get_selection_method(name: str) -> type:
    if name not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection method: {name}. "
                        f"Available: {list(SELECTION_METHODS.keys())}")
    return SELECTION_METHODS[name]


def register_method(name: str):
    def decorator(cls):
        SELECTION_METHODS[name] = cls
        return cls
    return decorator


from .random import RandomSelection
from .simple import KCenterSelection, HerdingSelection, KMeansSelection, DiversitySelection
from .rds import RDSSelection
from .less import LESSSelection
from .gradmatch import GradMatchSelection
from .craig_proxy import CRAIGProxySelection
from .craig import CRAIGSelection
from .orient import OrientSelection

__all__ = [
    "SelectionMethod", "SelectionResult", "SELECTION_METHODS",
    "get_selection_method", "register_method",
    "RandomSelection", "KCenterSelection", "HerdingSelection",
    "KMeansSelection", "DiversitySelection", "RDSSelection",
    "LESSSelection", "GradMatchSelection", "CRAIGProxySelection",
    "CRAIGSelection", "OrientSelection",
]
