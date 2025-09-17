# pmflow_bnn

Small neural components and models used in the Pushing‑Medium demos.

- `PMField`, `LateralEI`, `pm_local_plasticity` — medium dynamics building blocks
- `PMBNN`, `PMBNNAlwaysPlastic` — analogue BNNs for toy vision tasks
- `CNNBaseline`, `GRUBaseline` — simple baselines for comparison

## Install (editable)

From the repo root:

```bash
pip install -e programs/demos/nn_lib
```

## Quick usage

```python
import torch
from pmflow_bnn import get_model

m = get_model("pmflow_bnn_plastic", d_latent=8, channels=64, pm_steps=4)
x = torch.randn(8, 1, 28, 28)
logits, _ = m(x, T=3)
print(logits.shape)  # (8, 10)
```

## Tests

```bash
pytest programs/demos/nn_lib -q
```
