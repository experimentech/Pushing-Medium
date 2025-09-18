# pmflow_bnn_prototype.py
import math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

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
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t + 0.2*torch.randn_like(t)),  # Gaussian noise
        RandomOcclusion(p=0.25, max_frac=0.25),
        transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0))
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor()
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# -----------------------------
# PMFlow dynamics components
# -----------------------------
class PMField(nn.Module):
    def __init__(self, d_latent=8, n_centers=32, steps=4, dt=0.12, beta=1.0, clamp=3.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, d_latent)*0.7)
        self.mus     = nn.Parameter(torch.ones(n_centers)*0.4)
        self.steps, self.dt, self.beta, self.clamp = steps, dt, beta, clamp
        
    def grad_ln_n(self, z):
        eps = 1e-4
        B, d = z.shape
        n = torch.ones(B, device=z.device)
        g = torch.zeros_like(z)
        for c, mu in zip(self.centers, self.mus):
            rvec = z - c
            r2 = (rvec * rvec).sum(1) + eps  # [B]
            r = torch.sqrt(r2)               # [B]
            n = n + mu / r
            denom = r.unsqueeze(1) * r2.unsqueeze(1)  # [B,1]
            g = g + (-mu) * rvec / denom
        return g / n.unsqueeze(1)

    def forward(self, z):
        for _ in range(self.steps):
            z = z + self.dt * self.beta * self.grad_ln_n(z)
            z = torch.clamp(z, -self.clamp, self.clamp)
        return z

class LateralEI(nn.Module):
    # Batch-wise Gaussian affinity; for larger batches, use kNN approximation
    def __init__(self, sigma_e=0.6, sigma_i=1.2, k_e=0.8, k_i=1.0, gain=0.05):
        super().__init__()
        self.sigma_e, self.sigma_i = sigma_e, sigma_i
        self.k_e, self.k_i = k_e, k_i
        self.gain = gain
    def forward(self, z, h):
        with torch.no_grad():
            dist2 = torch.cdist(z, z).pow(2)  # [B,B]
            Ke = self.k_e * torch.exp(-dist2/(2*self.sigma_e**2))
            Ki = self.k_i * torch.exp(-dist2/(2*self.sigma_i**2))
            K = Ke - Ki
            K = K / (K.sum(1, keepdim=True) + 1e-6)  # normalize rows
        return self.gain * (K @ h)

# -----------------------------
# PMFlow-BNN model
# -----------------------------
class PMBNN(nn.Module):
    def __init__(self, d_latent=8, channels=64, pm_steps=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.Tanh(),
            nn.Linear(256, d_latent)
        )
        self.pm = PMField(d_latent=d_latent, n_centers=48, steps=pm_steps, dt=0.12, beta=1.0, clamp=3.0)
        self.ei = LateralEI(gain=0.06)
        self.proj = nn.Linear(d_latent, channels)
        self.readout = nn.Linear(channels, 10)

    def step(self, z, h):
        # Advect latent coords, update population activity with leak and EI
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
        return logits, (z, h)

# -----------------------------
# Local plasticity (unsupervised, interleaved)
# -----------------------------
@torch.no_grad()
def pm_local_plasticity(pmfield: PMField, z_batch, h_batch, I_strength=0.2, mu_lr=1e-3, c_lr=1e-3, homeo_target=0.2, homeo_lr=5e-4):
    # Hebbian-like: reinforce centers near high |h| and high |z| density
    B, d = z_batch.shape
    # Influence weights around centers
    # w_k = sum_b exp(-||z_b - c_k||^2 / (2*s^2)) * ||h_b||^2
    s2 = 0.8**2
    C = pmfield.centers  # [K,d]
    K = C.size(0)
    # Compute affinities
    dist2 = torch.cdist(C, z_batch).pow(2)  # [K,B]
    W = torch.exp(-dist2 / (2*s2))          # [K,B]
    hpow = (h_batch*h_batch).sum(1, keepdim=True).T  # [1,B]
    drive = (W * hpow).mean(1)  # [K]

    # Update mus (Hebbian with decay)
    pmfield.mus.add_(mu_lr * (drive - 0.1*pmfield.mus))

    # Update centers toward activity barycenters
    # c_k <- c_k + c_lr * sum_b W_kb * z_b / sum_b W_kb - c_k
    denom = W.sum(1, keepdim=True) + 1e-6  # [K,1]
    target = (W @ z_batch) / denom         # [K,d]
    pmfield.centers.add_(c_lr * (target - C))

    # Homeostasis on overall field strength (optional)
    # Scale mus to keep mean flow magnitude around target
    with torch.enable_grad():
        # Approximate flow magnitude on batch
        pmfield.zero_grad(set_to_none=True)
        g = pmfield.grad_ln_n(z_batch.detach())
        mag = g.norm(dim=1).mean().detach()
    scale = (homeo_target / (mag + 1e-6))
    pmfield.mus.mul_(1.0 + homeo_lr * (scale - 1.0))

# -----------------------------
# Baselines
# -----------------------------
class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

class GRUBaseline(nn.Module):
    # Treat each row (28) as timestep, features=28
    def __init__(self, hidden=128, layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 10)
    def forward(self, x):
        B = x.size(0)
        seq = x.squeeze(1)  # [B,28,28]
        out, h = self.gru(seq)
        return self.fc(out[:, -1, :])

# -----------------------------
# Training and evaluation
# -----------------------------
def train_epoch_bnn(model, opt, loader, device, T=5, plasticity_every=2):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits, (z, h) = model(x, T=T)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        # Interleave local unsupervised plasticity (no labels)
        if step % plasticity_every == 0:
            with torch.no_grad():
                # Run one extra step to get z,h reflecting recent weights
                z, h, _ = model.step(model.enc(x), torch.zeros(x.size(0), model.readout.in_features, device=device))
                pm_local_plasticity(model.pm, z, h)

        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_model(model, loader, device, kind="logits", T=5):
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
        # Apply heavy corruption at inference
        xn = torch.clamp(x + noise_std*torch.randn_like(x), 0, 1)
        # Occlusion
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
def attractor_memory_probe(bnn: PMBNN, x, device, T_on=3, T_off=15):
    # Phase 1: run with input to get a state
    bnn.eval()
    x = x.to(device)
    logits_on, (z, h) = bnn(x, T=T_on)

    # Phase 2: freeze input; iterate internal dynamics only
    # We emulate “no new drive” by not re-encoding the image; just step latent/activity
    traj_energy = []
    z_cur, h_cur = z.clone(), h.clone()
    logits_seq = []
    for t in range(T_off):
        z_cur, h_cur, logits = bnn.step(z_cur, h_cur)
        logits_seq.append(logits)
        # simple energy proxy: change magnitude
        e = (h_cur.pow(2).sum(1).sqrt().mean()).item()
        traj_energy.append(e)

    logits_final = logits_seq[-1]
    return logits_on, logits_final, traj_energy

# -----------------------------
# Orchestration
# -----------------------------
def main():
    set_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = make_loaders(batch=128)

    # Instantiate models
    bnn = PMBNN(d_latent=8, channels=64, pm_steps=4).to(device)
    cnn = CNNBaseline().to(device)
    gru = GRUBaseline(hidden=128).to(device)

    opt_bnn = torch.optim.Adam(bnn.parameters(), lr=1e-3)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    opt_gru = torch.optim.Adam(gru.parameters(), lr=1e-3)

    epochs = 5
    logs = {
        "BNN": {"train":[], "test":[]},
        "CNN": {"train":[], "test":[]},
        "GRU": {"train":[], "test":[]},
    }

    print(f"Params — BNN: {count_params(bnn):,} | CNN: {count_params(cnn):,} | GRU: {count_params(gru):,}")

    for ep in range(1, epochs+1):
        # Train
        tr_bnn_loss, tr_bnn_acc = train_epoch_bnn(bnn, opt_bnn, train_loader, device, T=5, plasticity_every=2)

        # Simple one-pass train for baselines (no plasticity)
        def train_epoch(model, opt):
            model.train()
            total, correct, loss_sum = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item()*x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
            return loss_sum/total, correct/total

        tr_cnn_loss, tr_cnn_acc = train_epoch(cnn, opt_cnn)
        tr_gru_loss, tr_gru_acc = train_epoch(gru, opt_gru)

        # Eval
        te_bnn_acc = eval_model(bnn, test_loader, device, kind="bnn", T=5)
        te_cnn_acc = eval_model(cnn, test_loader, device)
        te_gru_acc = eval_model(gru, test_loader, device)

        logs["BNN"]["train"].append(tr_bnn_acc); logs["BNN"]["test"].append(te_bnn_acc)
        logs["CNN"]["train"].append(tr_cnn_acc); logs["CNN"]["test"].append(te_cnn_acc)
        logs["GRU"]["train"].append(tr_gru_acc); logs["GRU"]["test"].append(te_gru_acc)

        print(f"Epoch {ep:02d} | "
              f"BNN tr {tr_bnn_acc:.3f} te {te_bnn_acc:.3f} | "
              f"CNN tr {tr_cnn_acc:.3f} te {te_cnn_acc:.3f} | "
              f"GRU tr {tr_gru_acc:.3f} te {te_gru_acc:.3f}")

    # Plot accuracy curves (distinct markers per model)
    epochs_axis = list(range(1, epochs+1))
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4), sharey=True)
    # Train
    ax1.plot(epochs_axis, logs["BNN"]["train"], 'r^--', label='BNN Train')
    ax1.plot(epochs_axis, logs["CNN"]["train"], 'bo--', label='CNN Train')
    ax1.plot(epochs_axis, logs["GRU"]["train"], 'gs--', label='GRU Train')
    ax1.set_title("Training Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.grid(True); ax1.legend()
    # Test
    ax2.plot(epochs_axis, logs["BNN"]["test"], 'r^-', label='BNN Test')
    ax2.plot(epochs_axis, logs["CNN"]["test"], 'bo-', label='CNN Test')
    ax2.plot(epochs_axis, logs["GRU"]["test"], 'gs-', label='GRU Test')
    ax2.set_title("Test Accuracy"); ax2.set_xlabel("Epoch"); ax2.grid(True); ax2.legend()
    plt.suptitle("Noisy MNIST: PMFlow-BNN vs Baselines")
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig("acc_curves.png", dpi=150); plt.close()

    # Robustness eval (heavy corruption)
    models = [("BNN","bnn",bnn), ("CNN","cnn",cnn), ("GRU","gru",gru)]
    robust = robustness_eval(models, test_loader, device, noise_std=0.7, occ_p=0.7, occ_frac=0.45, T=6)
    print("Robustness (heavy noise+occlusion) accuracy:", robust)

    # Attractor memory probe on a small batch
    x_sample, y_sample = next(iter(test_loader))
    x_sample = x_sample[:64]
    logits_on, logits_final, traj_energy = attractor_memory_probe(bnn, x_sample, device, T_on=4, T_off=20)
    conf_on = F.softmax(logits_on, dim=1).max(1).values.mean().item()
    conf_final = F.softmax(logits_final, dim=1).max(1).values.mean().item()
    print(f"Attractor probe — mean confidence: on={conf_on:.3f} -> final={conf_final:.3f}")

    # Plot attractor trajectory energy proxy
    plt.figure(figsize=(5,3.2))
    plt.plot(range(1, len(traj_energy)+1), traj_energy, 'k.-')
    plt.xlabel("Off-input steps"); plt.ylabel("||h|| RMS")
    plt.title("BNN internal settling (attractor proxy)")
    plt.grid(True); plt.tight_layout(); plt.savefig("bnn_attractor_energy.png", dpi=150); plt.close()

    # Save a quick summary
    with open("summary.txt","w") as f:
        f.write("Final results\n")
        for name in ["BNN","CNN","GRU"]:
            f.write(f"{name} test acc: {logs[name]['test'][-1]:.4f}\n")
        f.write("Robustness acc (heavy noise/occlusion):\n")
        for k,v in robust.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"Attractor probe mean confidence on->final: {conf_on:.4f} -> {conf_final:.4f}\n")

if __name__ == "__main__":
    main()

