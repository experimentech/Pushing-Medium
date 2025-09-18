import torch
import torch.nn as nn
import torch.nn.functional as F

class PMField(nn.Module):
    def __init__(self, d_latent=8, n_centers=16, steps=4, dt=0.15, beta=1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent)*0.7)
        self.mus = nn.Parameter(torch.ones(n_centers)*0.5)
        self.steps, self.dt, self.beta = steps, dt, beta

    def grad_ln_n(self, z):
        # z: [B,d]; returns g: [B,d]
        eps = 1e-4
        n = torch.ones(z.size(0), device=z.device)
        g = torch.zeros_like(z)
        for c, mu in zip(self.centers, self.mus):
            rvec = z - c
            r = torch.sqrt((rvec*rvec).sum(dim=1) + eps)
            n = n + mu / r
            g += (-mu) * rvec / (r.pow(3).unsqueeze(1))
        return g / n.unsqueeze(1)

    def forward(self, z):
        for _ in range(self.steps):
            z = torch.clamp(z + self.dt * self.beta * self.grad_ln_n(z), -3.0, 3.0)
        return z

class LateralEI(nn.Module):
    def __init__(self, d_latent, sigma_e=0.6, sigma_i=1.2, k_e=0.8, k_i=1.0):
        super().__init__()
        self.sigma_e, self.sigma_i = sigma_e, sigma_i
        self.k_e, self.k_i = k_e, k_i

    def forward(self, z, h):
        # z: [B,d], h: [B,C]; approximate local EI with Gaussian affinities
        # (for simplicity use batchwise affinities; for large B use k-NN)
        with torch.no_grad():
            dist2 = torch.cdist(z, z).pow(2)  # [B,B]
            Ke = self.k_e * torch.exp(-dist2/(2*self.sigma_e**2))
            Ki = self.k_i * torch.exp(-dist2/(2*self.sigma_i**2))
            K = Ke - Ki
        return K @ h  # [B,B]@[B,C] -> [B,C]

class PMBNNCell(nn.Module):
    def __init__(self, d_in, d_latent=8, d_out=10, channels=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 128), nn.Tanh(), nn.Linear(128, d_latent))
        self.pm = PMField(d_latent=d_latent)
        self.ei = LateralEI(d_latent=d_latent)
        self.readout = nn.Linear(channels, d_out)
        self.proj = nn.Linear(d_latent, channels)  # lift to channels

    def step(self, x, h, z):
        # encode to latent coords, advect, lateral EI, leak, nonlinearity
        z = self.pm(z)  # move coordinates along flow (tissue drift)
        h = 0.9*h + 0.1*F.tanh(self.proj(z))  # leak + drive from latent coords
        h = h + 0.05*self.ei(z, h)            # lateral E-I competition
        y = self.readout(h)
        return h, z, y

    def forward(self, x, T=5):
        B = x.size(0)
        z = self.enc(x.view(B, -1))
        h = torch.zeros(B, self.readout.in_features, device=x.device)
        for _ in range(T):
            h, z, y = self.step(x, h, z)
        return y

