import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# 1) Data: two interleaving spirals
# -----------------------------
def make_spirals(n_per_class=400, noise=0.08, turns=1.75, seed=0):
    rng = np.random.RandomState(seed)
    n = n_per_class
    t = np.linspace(0.0, 1.0, n)
    ang = turns * 2*np.pi * t
    r = t
    x1 = np.stack([r*np.cos(ang), r*np.sin(ang)], axis=1) + noise*rng.randn(n,2)
    x2 = np.stack([r*np.cos(ang+np.pi), r*np.sin(ang+np.pi)], axis=1) + noise*rng.randn(n,2)
    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=0)
    # scale to ~[-1,1]
    X /= np.max(np.linalg.norm(X, axis=1))
    return X.astype(np.float32), y

X, y = make_spirals(n_per_class=500, noise=0.06, turns=2.2, seed=42)
X_train = torch.from_numpy(X)
y_train = torch.from_numpy(y)

# -----------------------------
# 2) PM flow layer in latent space (2D)
# -----------------------------
class PMFlow(nn.Module):
    def __init__(self, centers, mus, steps=6, dt=0.2, advect=0.0):
        super().__init__()
        # Store as parameters so we can "swap" at inference by assigning .data
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32), requires_grad=False)
        self.mus     = nn.Parameter(torch.tensor(mus, dtype=torch.float32), requires_grad=False)
        self.steps = steps
        self.dt = dt
        self.advect = advect  # optional uniform drift term

    @torch.no_grad()
    def set_flow(self, centers, mus, steps=None, dt=None, advect=None):
        self.centers.data = torch.tensor(centers, dtype=torch.float32, device=self.centers.device)
        self.mus.data     = torch.tensor(mus, dtype=torch.float32, device=self.mus.device)
        if steps is not None: self.steps = int(steps)
        if dt    is not None: self.dt = float(dt)
        if advect is not None: self.advect = float(advect)

    def forward(self, z):
        # z: [B,2]
        # n(z) = 1 + sum(mu / r); grad ln n = (grad n)/n
        def n_and_grad(z):
            # z: [B,2]
            B = z.shape[0]
            n = torch.ones(B, device=z.device)
            grad = torch.zeros_like(z)
            eps = 1e-4
            for c, mu in zip(self.centers, self.mus):
                rvec = z - c  # [B,2]
                r = torch.sqrt((rvec*rvec).sum(dim=1) + eps)  # [B]
                n = n + mu / r
                grad = grad + (-mu) * rvec / (r**3).unsqueeze(1)
            grad_ln_n = grad / n.unsqueeze(1)
            return n, grad_ln_n

        # Euler integrate a short PM ODE in latent space
        z_out = z
        for _ in range(self.steps):
            nloc, g = n_and_grad(z_out)
            # Optional: project gradient perpendicular to feature direction (skip; latent has no "ray dir").
            # Use a mild advection (constant drift) to keep things dynamic if desired.
            z_out = z_out + self.dt * (g + self.advect * torch.tensor([0.0, 1.0], device=z.device))
            # Optional stabilization
            z_out = torch.clamp(z_out, -3.0, 3.0)
        return z_out

# -----------------------------
# 3) Model: encoder -> PM flow -> head
# -----------------------------
class PMClassifier(nn.Module):
    def __init__(self, flow_cfg=None, use_flow=True):
        super().__init__()
        self.use_flow = use_flow
        self.enc = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2)   # latent 2D
        )
        if use_flow:
            centers = flow_cfg.get("centers", [[-0.6, 0.0], [0.6, 0.0]])
            mus     = flow_cfg.get("mus",     [0.35, 0.35])
            steps   = flow_cfg.get("steps",   6)
            dt      = flow_cfg.get("dt",      0.2)
            advect  = flow_cfg.get("advect",  0.0)
            self.flow = PMFlow(centers, mus, steps=steps, dt=dt, advect=advect)
        else:
            self.flow = None
        self.head = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        z = self.enc(x)
        if self.flow is not None:
            z = self.flow(z)
        logits = self.head(z)
        return logits

# -----------------------------
# 4) Train one model
# -----------------------------
def train_model(model, X, y, epochs=200, lr=5e-3, device="cpu"):
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for ep in tqdm(range(1, epochs+1), desc="Training", unit="ep"):
        model.train()
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean().item()
            hist.append((ep, loss.item(), acc))
    return hist

# -----------------------------
# 5) Eval + visualize decision boundaries
# -----------------------------
@torch.no_grad()
def eval_acc(model, X, y, device="cpu"):
    model.eval().to(device)
    X = X.to(device)
    y = y.to(device)
    logits = model(X)
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def plot_decision(model, fname, title, grid=400, device="cpu"):
    model.eval().to(device)
    xs = np.linspace(-1.2, 1.2, grid)
    ys = np.linspace(-1.2, 1.2, grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    pts_t = torch.from_numpy(pts).to(device)
    logits = model(pts_t)
    probs = logits.softmax(dim=1)[:,1].reshape(grid, grid).cpu().numpy()
    plt.figure(figsize=(5.2,5))
    plt.contourf(XX, YY, probs, levels=25, cmap="coolwarm", alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, s=6, cmap="coolwarm", edgecolors='k', linewidths=0.2)
    plt.title(title)
    plt.xlim(-1.2,1.2); plt.ylim(-1.2,1.2); plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

# -----------------------------
# 6) Run: train with flow, evaluate, then swap flow and re-evaluate
# -----------------------------
device = "cpu"

flow_train_cfg = dict(
    centers=[[-0.5, 0.0], [0.5, 0.0]],
    mus=[0.45, 0.45],
    steps=6,
    dt=0.2,
    advect=0.0
)
model_pm = PMClassifier(flow_cfg=flow_train_cfg, use_flow=True)
hist = train_model(model_pm, X_train, y_train, epochs=200, lr=5e-3, device=device)
acc_train = eval_acc(model_pm, X_train, y_train, device=device)
print(f"Train accuracy (PM flow): {acc_train:.4f}")
plot_decision(model_pm, "decision_pm_trained.png", "Trained with PM flow", grid=400, device=device)

# Inference: swap the flow (no weight changes)
flow_swap_cfg = dict(
    centers=[[-0.2,  0.4], [0.7, -0.1], [-0.7, -0.1]],
    mus=[0.35, 0.30, 0.30],
    steps=6,
    dt=0.25,
    advect=0.05
)
model_pm.flow.set_flow(**flow_swap_cfg)
acc_swapped = eval_acc(model_pm, X_train, y_train, device=device)
print(f"Inference accuracy after flow swap (weights frozen): {acc_swapped:.4f}")
plot_decision(model_pm, "decision_pm_swapped.png", "Inference with swapped PM flow", grid=400, device=device)

# Optional: baseline without flow for comparison
model_base = PMClassifier(use_flow=False)
hist_b = train_model(model_base, X_train, y_train, epochs=200, lr=5e-3, device=device)
acc_base = eval_acc(model_base, X_train, y_train, device=device)
print(f"Train accuracy (no flow): {acc_base:.4f}")
plot_decision(model_base, "decision_noflow_trained.png", "Trained without flow", grid=400, device=device)

print("Saved: decision_pm_trained.png, decision_pm_swapped.png, decision_noflow_trained.png")

