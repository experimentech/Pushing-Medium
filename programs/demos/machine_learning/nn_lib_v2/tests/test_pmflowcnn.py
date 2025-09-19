import sys
import time
from pathlib import Path

import torch

# Ensure local package path for tests
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from pmflow_bnn import PMFlowCNN  # noqa: E402


def test_forward_shape_cpu():
    device = torch.device("cpu")
    model = PMFlowCNN(n_classes=10, temporal_parallel=True, chunk_size=8).to(device).eval()
    x = torch.randn(5, 1, 28, 28, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (5, 10)


def test_throughput_monotonic_cpu():
    device = torch.device("cpu")
    model = PMFlowCNN(n_classes=10, temporal_parallel=True, chunk_size=16).to(device).eval()
    batch_sizes = [8, 16, 32]
    throughputs = []
    for bs in batch_sizes:
        xb = torch.randn(bs, 1, 28, 28, device=device)
        with torch.no_grad():
            for _ in range(2):
                _ = model(xb)
        t0 = time.time(); iters = 6
        with torch.no_grad():
            for _ in range(iters):
                _ = model(xb)
        dt = (time.time() - t0) / iters
        throughputs.append(bs / dt)

    assert throughputs[1] >= 0.6 * throughputs[0]
    assert throughputs[2] >= 0.6 * throughputs[1]
