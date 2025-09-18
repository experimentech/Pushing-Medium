from .bnn import PMBNN, PMBNNAlwaysPlastic
from .baselines import CNNBaseline, GRUBaseline

def get_model(name, **kwargs):
    """
    Factory to create models by name.

    Args:
        name (str): One of:
            - "pmflow_bnn"            : Train-then-freeze PMFlow-BNN
            - "pmflow_bnn_plastic"    : Always-plastic PMFlow-BNN
            - "cnn"                   : CNN baseline
            - "gru"                   : GRU baseline
        **kwargs: Passed through to the model constructor.

    Returns:
        torch.nn.Module: The requested model instance.
    """
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

