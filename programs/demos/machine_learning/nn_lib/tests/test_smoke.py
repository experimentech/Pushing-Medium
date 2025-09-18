import sys, os
import torch

# Ensure the editable install path works during CI or local run
# If not installed, add the package path manually
pkg_path = os.path.join(os.path.dirname(__file__), '..')
if pkg_path not in sys.path:
    sys.path.insert(0, os.path.abspath(pkg_path))

import pmflow_bnn as P

def test_import_and_factory():
    assert hasattr(P, '__version__')
    for name in ['pmflow_bnn', 'pmflow_bnn_plastic', 'cnn', 'gru']:
        m = P.get_model(name)
        assert m is not None


def test_forward_shapes():
    bnn = P.get_model('pmflow_bnn_plastic', d_latent=8, channels=32, pm_steps=2)
    x = torch.randn(4, 1, 28, 28)
    logits, (z, h) = bnn(x, T=2)
    assert logits.shape == (4, 10)
    assert z.shape[0] == 4
    assert h.shape[0] == 4
