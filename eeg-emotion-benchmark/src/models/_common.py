"""
_common.py
==========
Shared plumbing so every `src/models/proposal/<algo>.py` file exposes an
identical `build(cfg)` factory without copy-pasting the same 10 lines.

`build(cfg)` is the single "pass parameters -> get a ready model" surface
(`build_model({..., "meta": meta})`).
`cfg` carries:
    * `cfg["meta"]["input_shape"]` : the loader-derived input dims for this fold
    * `cfg["n_classes"]`           : number of target classes (default 2)
    * any architecture hyper-parameter, overriding the per-model default
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple


def input_shape(cfg: Dict, default: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return `cfg["meta"]["input_shape"]` as a tuple, or `default` if absent."""
    return tuple(cfg.get("meta", {}).get("input_shape", default))


def make_builder(
    cls: Callable,
    *,
    default_shape: Tuple[int, ...],
    shape_kwargs: Callable[[Tuple[int, ...]], Dict],
    params: Dict,
) -> Callable[[Dict], object]:
    """Build a `build(cfg) -> cls(...)` factory.

    Parameters
    ----------
    cls
        The `nn.Module` subclass to instantiate.
    default_shape
        Input shape assumed when `cfg` carries no `meta["input_shape"]`.
    shape_kwargs
        Maps the resolved input shape to the constructor kwargs that describe the
        input dimensions, e.g. ``lambda s: {"n_channels": s[0], "n_samples": s[1]}``.
    params
        ``{constructor_kwarg: default}``. Each entry is overridable via ``cfg[key]``.
    """

    def build(cfg: Dict):
        shape = input_shape(cfg, default_shape)
        kw = {k: cfg.get(k, v) for k, v in params.items()}
        kw.update(shape_kwargs(shape))
        kw.setdefault("n_classes", cfg.get("n_classes", 2))
        return cls(**kw)

    return build
