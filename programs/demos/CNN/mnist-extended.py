import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np

# -----------------------------
# PM Flow block
# -----------------------------
class PMFlow(nn.Module):
    def __init__(self, latent_dim=16, centers=None, mus=None, steps=3, dt=0.1):
        super().__init__()
        if centers is None:
            centers = torch.randn(4, latent_dim) * 0.5
        if mus is None:
            mus = torch.ones(len(centers)) * 0.5
        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.mus = nn.Parameter(torch.tensor(mus, dtype=torch.float32))
        self.steps = steps
        self.dt = dt

    def forward(self, z):
        for _ in range(self.steps):
            n = torch.ones(z.size(0), device=z.device)
            grad = torch.zeros_like(z)
            for c, mu in zip(self.centers, self.mus):
                rvec = z - c
                r = torch.norm(rvec, dim=1) + 1e-4
                n = n + mu / r
                grad = grad + (-mu) * rvec / (r.unsqueeze(1)**3)
            grad_ln_n = grad / n.unsqueeze(1)
            z = z + self.dt * grad_ln_n
        return z

    @torch.no_grad()
    def set_flow(self, centers, mus, steps=None, dt=None):
        self.centers.copy_(torch.tensor(centers, dtype=torch.float32, device=self.centers.device))
        self.mus.copy_(torch.tensor(mus, dtype=torch.float32, device=self.mus.device))
        if steps is not None: self.steps = steps
        if dt is not None: self.dt = dt

# -----------------------------
# Model
# -----------------------------
class PMNet(nn.Module):
    def __init__(self, use_flow=True, latent_dim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.flow = PMFlow(latent_dim=latent_dim) if use_flow else None
        self.head = nn.Linear(latent_dim, 10)

    def forward(self, x):
        z = self.enc(x)
        if self.flow is not None:
            z = self.flow(z)
        return self.head(z)

    def encode_latent(self, x, apply_flow=True):
        z = self.enc(x)
        if apply_flow and self.flow is not None:
            z = self.flow(z)
        return z

# -----------------------------
# Training / evaluation
# -----------------------------
def train_epoch(model, opt, loader, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct/total

# -----------------------------
# t-SNE plotting
# -----------------------------
@torch.no_grad()
def plot_tsne(model, loader, device, title, fname, apply_flow=True):
    model.eval()
    all_z, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        z = model.encode_latent(x, apply_flow=apply_flow)
        all_z.append(z.cpu().numpy())
        all_y.append(y.numpy())
    all_z = np.concatenate(all_z, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=30)
    z2d = tsne.fit_transform(all_z)
    plt.figure(figsize=(6,6))
    scatter = plt.scatter(z2d[:,0], z2d[:,1], c=all_y, cmap='tab10', s=5, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Train baseline
    model_base = PMNet(use_flow=False).to(device)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
    for ep in range(5):
        train_epoch(model_base, opt_base, train_loader, device)
    base_acc = eval_model(model_base, test_loader, device)
    print(f"Baseline test acc: {base_acc:.4f}")

    # Train PMFlow
    model_pm = PMNet(use_flow=True).to(device)
    opt_pm = torch.optim.Adam(model_pm.parameters(), lr=1e-3)
    for ep in range(5):
        train_epoch(model_pm, opt_pm, train_loader, device)
    pm_acc = eval_model(model_pm, test_loader, device)
    print(f"PMFlow test acc: {pm_acc:.4f}")

    # Plot t-SNEs
    plot_tsne(model_base, test_loader, device, "Baseline: No flow", "tsne_baseline.png", apply_flow=False)
    plot_tsne(model_pm, test_loader, device, "Trained with PM flow", "tsne_pm_trained.png", apply_flow=True)

    # Swap flow params
    new_centers = model_pm.flow.centers + 0.5*torch.randn_like(model_pm.flow.centers)
    new_mus = model_pm.flow.mus * (0.8 + 0.4*torch.rand_like(model_pm.flow.mus))
    model_pm.flow.set_flow(new_centers, new_mus)
    swapped_acc = eval_model(model_pm, test_loader, device)
    print(f"Test acc after flow swap: {swapped_acc:.4f}")
    plot_tsne(model_pm, test_loader, device, "Inference with swapped PM flow", "tsne_pm_swapped.png", apply_flow=True)

if __name__ == "__main__":
    main()

