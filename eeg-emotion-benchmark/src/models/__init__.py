"""Model zoo for the DREAMER emotion-recognition benchmark.

Each module defines one architecture (plus, where relevant, its custom loss) and a
`build(cfg)` factory with the signature expected by
`evaluation.run_subject_dependent` / `run_loso` (`build_model({..., "meta": meta})`).
"""

from . import (  # noqa: F401
    eegnet,
    dgcnn,
    siam_gcan,
    cit_emotionnet,
    shap_diagnostic,
    cnn_transformer_reduced,
    dadpc,
    dual_attention,
)

__all__ = [
    "eegnet",
    "dgcnn",
    "siam_gcan",
    "cit_emotionnet",
    "shap_diagnostic",
    "cnn_transformer_reduced",
    "dadpc",
    "dual_attention",
]
