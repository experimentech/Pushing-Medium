import torch
import torch.nn as nn


class PMField(nn.Module):
    def __init__(self, d_latent=8, n_centers=32, steps=4, dt=0.12, beta=1.0, clamp=3.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent) * 0.7)
        self.mus = nn.Parameter(torch.ones(n_centers) * 0.4)
        self.steps = steps
        self.dt = dt
        self.beta = beta
        self.clamp = clamp

    def grad_ln_n(self, z):
        eps = 1e-4
        B, _ = z.shape
        n = torch.ones(B, device=z.device)
        g = torch.zeros_like(z)
        for c, mu in zip(self.centers, self.mus):
            rvec = z - c
            r2 = (rvec * rvec).sum(1) + eps
            r = torch.sqrt(r2)
            n = n + mu / r
            denom = r.unsqueeze(1) * r2.unsqueeze(1)
            g = g + (-mu) * rvec / denom
        return g / n.unsqueeze(1)

    def forward(self, z):
        for _ in range(self.steps):
            z = z + self.dt * self.beta * self.grad_ln_n(z)
            z = torch.clamp(z, -self.clamp, self.clamp)
        return z


class LateralEI(nn.Module):
    def __init__(self, sigma_e=0.6, sigma_i=1.2, k_e=0.8, k_i=1.0, gain=0.05):
        super().__init__()
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.k_e = k_e
        self.k_i = k_i
        self.gain = gain

    def forward(self, z, h):
        with torch.no_grad():
            dist2 = torch.cdist(z, z).pow(2)
            Ke = self.k_e * torch.exp(-dist2 / (2 * self.sigma_e ** 2))
            Ki = self.k_i * torch.exp(-dist2 / (2 * self.sigma_i ** 2))
            K = Ke - Ki
            K = K / (K.sum(1, keepdim=True) + 1e-6)
        return self.gain * (K @ h)


@torch.no_grad()
def pm_local_plasticity(pmfield: PMField, z_batch, h_batch, mu_lr=1e-3, c_lr=1e-3):
    s2 = 0.8 ** 2
    C = pmfield.centers
    dist2 = torch.cdist(C, z_batch).pow(2)
    W = torch.exp(-dist2 / (2 * s2))
    hpow = (h_batch * h_batch).sum(1, keepdim=True).T
    drive = (W * hpow).mean(1)
    pmfield.mus.add_(mu_lr * (drive - 0.1 * pmfield.mus))
    denom = W.sum(1, keepdim=True) + 1e-6
    target = (W @ z_batch) / denom
    pmfield.centers.add_(c_lr * (target - C))
