import math, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# -----------------------------
# Data: MNIST with noise & occlusion
# -----------------------------
class RandomOcclusion(object):
    def __init__(self, p=0.3, max_frac=0.3):
        self.p = p; self.max_frac = max_frac
    def __call__(self, x):
        if random.random() > self.p: return x
        c, h, w = x.shape
        fh, fw = int(h * random.uniform(0.1, self.max_frac)), int(w * random.uniform(0.1, self.max_frac))
        y0, x0 = random.randint(0, h - fh), random.randint(0, w - fw)
        x[:, y0:y0+fh, x0:x0+fw] = 0.0
        return x

def make_loaders(batch=128):
    num_workers = int(os.getenv("NUM_WORKERS", "2"))
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t + 0.2*torch.randn_like(t)),
        RandomOcclusion(p=0.25, max_frac=0.25),
        transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0))
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])
    data_dir = os.getenv("DATA_DIR", "./data")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=True))

# -----------------------------
# PMFlow dynamics
# -----------------------------
class PMField(nn.Module):
    def __init__(self, d_latent=8, n_centers=32, steps=4, dt=0.12, beta=1.0, clamp=3.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent)*0.7)
        self.mus     = nn.Parameter(torch.ones(n_centers)*0.4)
        self.steps, self.dt, self.beta, self.clamp = steps, dt, beta, clamp

    def grad_ln_n(self, z):
        eps = 1e-4
        B, _ = z.shape
        n = torch.ones(B, device=z.device)
        g = torch.zeros_like(z)
        for c, mu in zip(self.centers, self.mus):
            rvec = z - c
            r2 = (rvec*rvec).sum(1) + eps
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
        self.sigma_e, self.sigma_i = sigma_e, sigma_i
        self.k_e, self.k_i = k_e, k_i
        self.gain = gain
    def forward(self, z, h):
        with torch.no_grad():
            dist2 = torch.cdist(z, z).pow(2)
            Ke = self.k_e * torch.exp(-dist2/(2*self.sigma_e**2))
            Ki = self.k_i * torch.exp(-dist2/(2*self.sigma_i**2))
            K = Ke - Ki
            K = K / (K.sum(1, keepdim=True) + 1e-6)
        return self.gain * (K @ h)

# -----------------------------
# Local plasticity
# -----------------------------
@torch.no_grad()
def pm_local_plasticity(pmfield: PMField, z_batch, h_batch, mu_lr=1e-3, c_lr=1e-3):
    s2 = 0.8**2
    C = pmfield.centers
    dist2 = torch.cdist(C, z_batch).pow(2)
    W = torch.exp(-dist2 / (2*s2))
    hpow = (h_batch*h_batch).sum(1, keepdim=True).T
    drive = (W * hpow).mean(1)
    pmfield.mus.add_(mu_lr * (drive - 0.1*pmfield.mus))
    denom = W.sum(1, keepdim=True) + 1e-6
    target = (W @ z_batch) / denom
    pmfield.centers.add_(c_lr * (target - C))

# -----------------------------
# Always-plastic PMFlow-BNN
# -----------------------------
class PMBNNAlwaysPlastic(nn.Module):
    def __init__(self, d_latent=8, channels=64, pm_steps=4, plastic=True):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.Tanh(),
            nn.Linear(256, d_latent)
        )
        self.pm = PMField(d_latent=d_latent, n_centers=48, steps=pm_steps)
        self.ei = LateralEI(gain=0.06)
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, 10)
        self.plastic = plastic

    def step(self, z, h):
        z = self.pm(z)
        h = 0.90*h + 0.10*torch.tanh(self.proj(z))
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
            # Plasticity is applied outside the forward pass (in the training loop)
            # to avoid in-place parameter updates before autograd backward.
        return logits, (z, h)

# -----------------------------
# Baselines
# -----------------------------
class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

class GRUBaseline(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 10)
    def forward(self, x):
        seq = x.squeeze(1)
        out, _ = self.gru(seq)
        return self.fc(out[:, -1, :])

# -----------------------------
# Training loops with progress
# -----------------------------
def train_epoch(model, opt, loader, device, kind="bnn", T=5, epoch_idx=1, total_epochs=1):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(enumerate(loader, 1), total=len(loader),
                desc=f"[Epoch {epoch_idx}/{total_epochs}] {kind.upper()} train", ncols=100)
    for _, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        z_h = None
        if kind == "bnn":
            logits, z_h = model(x, T=T)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        # Apply local plasticity AFTER optimizer step to keep autograd happy
        if kind == "bnn" and getattr(model, "plastic", False) and z_h is not None:
            z, h = z_h
            pm_local_plasticity(model.pm, z.detach(), h.detach())
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_model(model, loader, device, kind="bnn", T=5):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if kind == "bnn":
            logits, _ = model(x, T=T)
        else:
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct/total

@torch.no_grad()
def robustness_eval(models, loader, device, noise_std=0.5, occ_p=0.5, occ_frac=0.4, T=5):
    res = {}
    for name, kind, m in models:
        m.eval()
    total = 0
    correct = {name:0 for name,_,_ in models}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xn = torch.clamp(x + noise_std*torch.randn_like(x), 0, 1)
        if random.random() < occ_p:
            c, h, w = xn.shape[1:]
            fh, fw = int(h*occ_frac), int(w*occ_frac)
            y0, x0 = random.randint(0, h-fh), random.randint(0, w-fw)
            xn[:,:,y0:y0+fh, x0:x0+fw] = 0.0
        for name, kind, m in models:
            if kind == "bnn":
                logits, _ = m(xn, T=T)
            else:
                logits = m(xn)
            correct[name] += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    for name in correct:
        res[name] = correct[name]/total
    return res

@torch.no_grad()
def attractor_memory_probe(bnn, x, device, T_on=3, T_off=15):
    bnn.eval()
    x = x.to(device)
    logits_on, (z, h) = bnn(x, T=T_on)
    traj_energy = []
    z_cur, h_cur = z.clone(), h.clone()
    logits_seq = []
    for _ in range(T_off):
        z_cur, h_cur, logits = bnn.step(z_cur, h_cur)
        logits_seq.append(logits)
        e = (h_cur.pow(2).sum(1).sqrt().mean()).item()
        traj_energy.append(e)
    logits_final = logits_seq[-1]
    return logits_on, logits_final, traj_energy

# -----------------------------
# Main orchestration
# -----------------------------
def main():
    headless = os.getenv("HEADLESS", "0") == "1"
    if headless:
        # Use non-interactive backend if requested
        import matplotlib
        matplotlib.use("Agg", force=True)
    print("[INFO] Starting PMFlow-BNN demo...")
    set_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = make_loaders(batch=128)

    bnn = PMBNNAlwaysPlastic(d_latent=8, channels=64, pm_steps=4, plastic=True).to(device)
    cnn = CNNBaseline().to(device)
    gru = GRUBaseline(hidden=128).to(device)

    opt_bnn = torch.optim.Adam(bnn.parameters(), lr=1e-3)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    opt_gru = torch.optim.Adam(gru.parameters(), lr=1e-3)

    epochs = int(os.getenv("EPOCHS", "5"))
    logs = { "BNN": {"train":[], "test":[]},
             "CNN": {"train":[], "test":[]},
             "GRU": {"train":[], "test":[]} }

    print(f"Params — BNN: {count_params(bnn):,} | CNN: {count_params(cnn):,} | GRU: {count_params(gru):,}")

    for ep in range(1, epochs+1):
        tr_bnn_loss, tr_bnn_acc = train_epoch(bnn, opt_bnn, train_loader, device, kind="bnn", T=5, epoch_idx=ep, total_epochs=epochs)
        tr_cnn_loss, tr_cnn_acc = train_epoch(cnn, opt_cnn, train_loader, device, kind="cnn", epoch_idx=ep, total_epochs=epochs)
        tr_gru_loss, tr_gru_acc = train_epoch(gru, opt_gru, train_loader, device, kind="gru", epoch_idx=ep, total_epochs=epochs)

        te_bnn_acc = eval_model(bnn, test_loader, device, kind="bnn", T=5)
        te_cnn_acc = eval_model(cnn, test_loader, device, kind="cnn")
        te_gru_acc = eval_model(gru, test_loader, device, kind="gru")

        logs["BNN"]["train"].append(tr_bnn_acc); logs["BNN"]["test"].append(te_bnn_acc)
        logs["CNN"]["train"].append(tr_cnn_acc); logs["CNN"]["test"].append(te_cnn_acc)
        logs["GRU"]["train"].append(tr_gru_acc); logs["GRU"]["test"].append(te_gru_acc)

        print(f"Epoch {ep:02d} | "
              f"BNN tr {tr_bnn_acc:.3f} te {te_bnn_acc:.3f} | "
              f"CNN tr {tr_cnn_acc:.3f} te {te_cnn_acc:.3f} | "
              f"GRU tr {tr_gru_acc:.3f} te {te_gru_acc:.3f}")

    # Plot accuracy curves
    epochs_axis = list(range(1, epochs+1))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4), sharey=True)
    ax1.plot(epochs_axis, logs["BNN"]["train"], 'r^--', label='BNN Train')
    ax1.plot(epochs_axis, logs["CNN"]["train"], 'bo--', label='CNN Train')
    ax1.plot(epochs_axis, logs["GRU"]["train"], 'gs--', label='GRU Train')
    ax1.set_title("Training Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.grid(True); ax1.legend()
    ax2.plot(epochs_axis, logs["BNN"]["test"], 'r^-', label='BNN Test')
    ax2.plot(epochs_axis, logs["CNN"]["test"], 'bo-', label='CNN Test')
    ax2.plot(epochs_axis, logs["GRU"]["test"], 'gs-', label='GRU Test')
    ax2.set_title("Test Accuracy"); ax2.set_xlabel("Epoch"); ax2.grid(True); ax2.legend()
    plt.suptitle("Noisy MNIST: Always-Plastic PMFlow-BNN vs Baselines")
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig("acc_curves.png", dpi=150); plt.close()

    # Robustness eval
    models = [("BNN","bnn",bnn), ("CNN","cnn",cnn), ("GRU","gru",gru)]
    robust = robustness_eval(models, test_loader, device, noise_std=0.7, occ_p=0.7, occ_frac=0.45, T=6)
    print("Robustness (heavy noise+occlusion) accuracy:", robust)

    # Attractor probe
    x_sample, _ = next(iter(test_loader))
    x_sample = x_sample[:64]
    logits_on, logits_final, traj_energy = attractor_memory_probe(bnn, x_sample, device, T_on=4, T_off=20)
    conf_on = F.softmax(logits_on, dim=1).max(1).values.mean().item()
    conf_final = F.softmax(logits_final, dim=1).max(1).values.mean().item()
    print(f"Attractor probe — mean confidence: on={conf_on:.3f} -> final={conf_final:.3f}")
    plt.figure(figsize=(5,3.2))
    plt.plot(range(1, len(traj_energy)+1), traj_energy, 'k.-')
    plt.xlabel("Off-input steps")
    plt.ylabel("||h|| RMS")
    plt.title("BNN internal settling (attractor proxy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bnn_attractor_energy.png", dpi=150)
    if not headless:
        plt.show()  # comment out if running headless
    else:
        plt.close('all')
    print("[INFO] Finished PMFlow-BNN demo.")
    

if __name__ == "__main__":
    # Ensure the script actually runs when executed directly, and surface errors
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

