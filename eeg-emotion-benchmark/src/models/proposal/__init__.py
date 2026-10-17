"""Paper-faithful implementations of the five thesis-proposal models.

Each module documents what was reproduced from its paper and every deviation.
The earlier prototypes in `src/models/` (dadpc.py, cit_emotionnet.py, ...) belong to
the pre-proposal DREAMER model zoo driven by `src/benchmark_si.py`.
"""

from . import cit_emotionnet, clisa, dadpc, dah_net, stgate  # noqa: F401

__all__ = ["dadpc", "cit_emotionnet", "stgate", "dah_net", "clisa"]
