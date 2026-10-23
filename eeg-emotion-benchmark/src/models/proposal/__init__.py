"""Paper-faithful implementations of the five thesis-proposal models.

Each module documents what was reproduced from its paper and every deviation.
"""

from . import cit_emotionnet, clisa, dadpc, dah_net, stgate  # noqa: F401

__all__ = ["dadpc", "cit_emotionnet", "stgate", "dah_net", "clisa"]
