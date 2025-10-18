"""
Inference engine registry.

Usage:
    from inference import get_engine, list_engines

    Engine = get_engine('baseline')
    engine = Engine(model)
    output = engine.generate(prompt, max_new_tokens=100)
"""

from typing import Dict, Type
from .base import InferenceEngine

# Registry
_ENGINES: Dict[str, Type[InferenceEngine]] = {}


def register(cls: Type[InferenceEngine]) -> Type[InferenceEngine]:
    """
    Register an inference engine.

    Usage:
        @register
        class MyEngine(InferenceEngine):
            name = "my_engine"
            ...
    """
    if not hasattr(cls, 'name') or cls.name == "base":
        raise ValueError(f"Engine {cls.__name__} must define a unique 'name' attribute")

    if cls.name in _ENGINES:
        raise ValueError(f"Engine '{cls.name}' already registered")

    _ENGINES[cls.name] = cls

    return cls


def get_engine(name: str) -> Type[InferenceEngine]:
    """Get engine class by name."""
    if name not in _ENGINES:
        available = ', '.join(_ENGINES.keys())
        raise ValueError(f"Unknown engine '{name}'. Available: {available}")
    return _ENGINES[name]


def list_engines() -> Dict[str, Type[InferenceEngine]]:
    """Get all registered engines."""
    return _ENGINES.copy()


# Explicitly import engines to register them
# (Add new imports here as you create new engines)
from . import baseline
from . import sampling
from . import kv_cache
from . import fp8
from . import fp8_static

__all__ = ['InferenceEngine', 'register', 'get_engine', 'list_engines']
