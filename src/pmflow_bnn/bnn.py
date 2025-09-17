import torch
import torch.nn as nn
from .pmflow import PMField, LateralEI


class PMBNN(nn.Module):
    """Train-then-freeze PMFlow-BNN"""

    def __init__(self, d_latent=8, channels=64, pm_steps=4, n_centers=48, n_classes=10):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.Tanh(),
            nn.Linear(256, d_latent),
        )
        self.pm = PMField(d_latent=d_latent, n_centers=n_centers, steps=pm_steps)
        self.ei = LateralEI(gain=0.06)
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, n_classes)

    def step(self, z, h):
        z = self.pm(z)
        h = 0.90 * h + 0.10 * torch.tanh(self.proj(z))
        h = h + self.ei(z, h)
        logits = self.readout(h)
        return z, h, logits

    def forward(self, x, T=5):
        B = x.size(0)
        z = self.enc(x)
        h = torch.zeros(B, self.readout.in_features, device=x.device)
        logits = None
        for _ in range(T):
            z, h, logits = self.step(z, h)
        return logits, (z, h)


class PMBNNAlwaysPlastic(nn.Module):
    """Always-adapting PMFlow-BNN (plasticity applied outside forward)."""

    def __init__(self, d_latent=8, channels=64, pm_steps=4, n_centers=48, n_classes=10, plastic=True):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.Tanh(),
            nn.Linear(256, d_latent),
        )
        self.pm = PMField(d_latent=d_latent, n_centers=n_centers, steps=pm_steps)
        self.ei = LateralEI(gain=0.06)
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, n_classes)
        self.plastic = plastic

    def step(self, z, h):
        z = self.pm(z)
        h = 0.90 * h + 0.10 * torch.tanh(self.proj(z))
        h = h + self.ei(z, h)
        logits = self.readout(h)
        return z, h, logits

    def forward(self, x, T=5):
        B = x.size(0)
        z = self.enc(x)
        h = torch.zeros(B, self.readout.in_features, device=x.device)
        logits = None
        for _ in range(T):
            z, h, logits = self.step(z, h)
        return logits, (z, h)
