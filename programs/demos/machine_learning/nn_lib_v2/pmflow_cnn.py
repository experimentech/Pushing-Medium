"""
Back-compat shim for legacy imports.

Historically some notebooks/code imported `pmflow_cnn.PMFlowCNN`.
In v0.2.0, CNN models live under `pmflow_bnn.baselines.CNNBaseline`.

This module provides a stable alias so older code keeps working:

    from pmflow_cnn import PMFlowCNN

Prefer new imports in fresh code:

    from pmflow_bnn.baselines import CNNBaseline
"""

try:
    # Preferred when imported as a package module
    from .pmflow_bnn.baselines import PMFlowCNN, CNNBaseline
except Exception:  # Fallback when imported top-level with sys.path pointing here
    from pmflow_bnn.baselines import PMFlowCNN, CNNBaseline  # type: ignore

__all__ = ["PMFlowCNN", "CNNBaseline"]
