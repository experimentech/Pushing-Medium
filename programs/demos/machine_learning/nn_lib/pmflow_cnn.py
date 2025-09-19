"""
Back-compat shim for legacy imports in nn_lib (pre-v2).

Older notebooks used:

    from pmflow_cnn import PMFlowCNN

CNN baselines now live under `pmflow_bnn.baselines`.
This shim keeps those imports working.
"""

try:
    from .pmflow_bnn.baselines import CNNBaseline as PMFlowCNN, CNNBaseline
except Exception:
    from pmflow_bnn.baselines import CNNBaseline as PMFlowCNN, CNNBaseline  # type: ignore

__all__ = ["PMFlowCNN", "CNNBaseline"]
