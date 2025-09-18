import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- PM Flow block ---
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

# --- Model ---
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

# --- Training / evaluation ---
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

# --- Main ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    model_base = PMNet(use_flow=False).to(device)
    model_pm = PMNet(use_flow=True).to(device)
    opt_base = torch.optim.Adam(model_base.parameters(), lr=1e-3)
    opt_pm = torch.optim.Adam(model_pm.parameters(), lr=1e-3)

    epochs = 10
    base_train_acc, base_test_acc = [], []
    pm_train_acc, pm_test_acc = [], []

    for ep in range(1, epochs+1):
        _, tr_acc_b = train_epoch(model_base, opt_base, train_loader, device)
        te_acc_b = eval_model(model_base, test_loader, device)
        base_train_acc.append(tr_acc_b)
        base_test_acc.append(te_acc_b)

        _, tr_acc_p = train_epoch(model_pm, opt_pm, train_loader, device)
        te_acc_p = eval_model(model_pm, test_loader, device)
        pm_train_acc.append(tr_acc_p)
        pm_test_acc.append(te_acc_p)

        print(f"Epoch {ep}: Base train={tr_acc_b:.4f} test={te_acc_b:.4f} | PM train={tr_acc_p:.4f} test={te_acc_p:.4f}")

    # --- Improved visualisation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), sharey=True)

    # Subplot 1: Training accuracy
    ax1.plot(range(1, epochs+1), base_train_acc, 'bo--', label='Base Train')
    ax1.plot(range(1, epochs+1), pm_train_acc, 'r^--', label='PM Train')
    ax1.set_title("Training Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Test accuracy
    ax2.plot(range(1, epochs+1), base_test_acc, 'bo-', label='Base Test')
    ax2.plot(range(1, epochs+1), pm_test_acc, 'r^-', label='PM Test')
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("MNIST: Baseline vs PMFlow")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("mnist_pmflow_accuracy_split.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()

