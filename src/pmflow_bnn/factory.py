from .bnn import PMBNN, PMBNNAlwaysPlastic
from .baselines import CNNBaseline, GRUBaseline


def get_model(name, **kwargs):
    """Factory to create models by name."""
    name = name.lower()
    if name == "pmflow_bnn":
        return PMBNN(**kwargs)
    elif name == "pmflow_bnn_plastic":
        return PMBNNAlwaysPlastic(**kwargs)
    elif name == "cnn":
        return CNNBaseline(**kwargs)
    elif name == "gru":
        return GRUBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
