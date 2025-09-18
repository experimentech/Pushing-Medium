import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# PM Flow block
# -----------------------------
class PMFlow(nn.Module):
    def __init__(self, latent_dim=64, centers=None, mus=None, steps=3, dt=0.1):
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

# -----------------------------
# MLP for 28×28 grayscale
# -----------------------------
class MLPNet(nn.Module):
    def __init__(self, use_flow=True, latent_dim=16, in_ch=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*in_ch, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.flow = PMFlow(latent_dim=latent_dim) if use_flow else None
        self.head = nn.Linear(latent_dim, 10)

    def forward(self, x):
        z = self.enc(x)
        if self.flow is not None:
            z = self.flow(z)
        return self.head(z)

# -----------------------------
# CNN for 32×32 RGB
# -----------------------------
class CNNNet(nn.Module):
    def __init__(self, use_flow=True, latent_dim=64, in_ch=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # CIFAR-10: 32x32 -> after 2x pooling: 8x8x128 = 8192
        self.fc1 = nn.Linear(8192, latent_dim)
        self.flow = PMFlow(latent_dim=latent_dim) if use_flow else None
        self.fc2 = nn.Linear(latent_dim, 10)

    def forward(self, x):
        z = self.conv(x)
        z = self.fc1(z)
        if self.flow is not None:
            z = self.flow(z)
        return self.fc2(z)

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
# Main
# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Dataset selection
    # -------------------------
    # --- MNIST (28×28 grayscale) ---
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    model_class = MLPNet
    latent_dim = 16
    in_ch = 1

    # --- Fashion-MNIST (28×28 grayscale) ---
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    # test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    # model_class = MLPNet
    # latent_dim = 16
    # in_ch = 1

    # --- CIFAR-10 (32×32 RGB) ---
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    # model_class = CNNNet
    # latent_dim = 64
    # in_ch = 3

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    # Models
    model_base = model_class(use_flow=False, latent_dim=latent_dim, in_ch=in_ch).to(device)
    model_pm = model_class(use_flow=True, latent_dim=latent_dim, in_ch=in_ch).to(device)

    models = [
        ("Base", model_base, 'bo-', 'Base Train', 'bs-', 'Base Test'),
        ("PMFlow", model_pm, 'r^-', 'PM Train', 'rv-', 'PM Test'),
    ]

    epochs = 10
    results = {name: {"train": [], "test": []} for name, *_ in models}

    for ep in range(1, epochs+1):
        for name, model, train_style, train_label, test_style, test_label in models:
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            if ep == 1:
                model.opt = opt
            else:
                opt = model.opt
            _, tr_acc = train_epoch(model, opt, train_loader, device)
            te_acc = eval_model(model, test_loader, device)
            results[name]["train"].append(tr_acc)
            results[name]["test"].append(te_acc)
        print(f"Epoch {ep} complete.")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), sharey=True)
    for name, model, train_style, train_label, test_style, test_label in models:
        ax1.plot(range(1, epochs+1), results[name]["train"], train_style, label=train_label)
        ax2.plot(range(1, epochs+1), results[name]["test"], test_style, label=test_label)

    ax1.set_title("Training Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle("Baseline vs PMFlow")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("pmflow_accuracy.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()

