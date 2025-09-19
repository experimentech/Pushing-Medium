from .pmflow import PMField, LateralEI, pm_local_plasticity
from .bnn import PMBNN, PMBNNAlwaysPlastic
from .baselines import CNNBaseline, GRUBaseline
try:
    # Optional MLP baseline if present (some branches miss it)
    from .baselines import MLPBaseline  # type: ignore
except Exception:  # pragma: no cover - optional
    MLPBaseline = None  # type: ignore
from .utils import count_params, set_seed
from .version import __version__
from .factory import get_model

__all__ = [
    "PMField", "LateralEI", "pm_local_plasticity",
    "PMBNN", "PMBNNAlwaysPlastic",
    "CNNBaseline", "GRUBaseline",
    # Optional
    *( ["MLPBaseline"] if MLPBaseline is not None else [] ),
    "count_params", "set_seed",
    "get_model"
]

