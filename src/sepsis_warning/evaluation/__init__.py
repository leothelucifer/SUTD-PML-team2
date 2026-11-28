"""Frozen evaluation public API."""

__all__ = ["run_frozen_evaluation", "AlreadyEvaluated", "EVALUATED_TAG"]


def __getattr__(name):
    if name in __all__:
        from . import frozen
        return getattr(frozen, name)
    raise AttributeError(name)
