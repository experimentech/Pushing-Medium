import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
# Ensure we can import the local package without installation
try:
    from pmflow_bnn import (
        get_model, PMBNNAlwaysPlastic, CNNBaseline, GRUBaseline,
        set_seed, count_params
    )
except ImportError:
    import sys
    # Try adding the package dir near this demo
    sys.path.insert(0, os.path.dirname(__file__))
    # Also try the conventional src/ layout at repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    from pmflow_bnn import (
        get_model, PMBNNAlwaysPlastic, CNNBaseline, GRUBaseline,
        set_seed, count_params
    )

# --- Data with noise & occlusion ---
class RandomOcclusion(object):
    def __init__(self, p=0.3, max_frac=0.3):
        self.p = p; self.max_frac = max_frac
    def __call__(self, x):
        import random
        if random.random() > self.p: return x
        c, h, w = x.shape
        fh, fw = int(h * random.uniform(0.1, self.max_frac)), int(w * random.uniform(0.1, self.max_frac))
        y0, x0 = random.randint(0, h - fh), random.randint(0, w - fw)
        x[:, y0:y0+fh, x0:x0+fw] = 0.0
        return x

def make_loaders(batch=128):
    num_workers = int(os.getenv("NUM_WORKERS", "2"))
    data_dir = os.getenv("DATA_DIR", "./data")
    train_limit = int(os.getenv("TRAIN_LIMIT", "0"))
    test_limit = int(os.getenv("TEST_LIMIT", "0"))
    test_batch = int(os.getenv("TEST_BATCH_SIZE", "512"))
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t + 0.2*torch.randn_like(t)),
        RandomOcclusion(p=0.25, max_frac=0.25),
        transforms.Lambda(lambda t: torch.clamp(t, 0.0, 1.0))
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=test_tf)
    if train_limit > 0:
        train_ds = Subset(train_ds, range(min(train_limit, len(train_ds))))
    if test_limit > 0:
        test_ds = Subset(test_ds, range(min(test_limit, len(test_ds))))
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds, batch_size=test_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    )

# --- Training helpers ---
def train_epoch(model, opt, loader, device, kind="bnn", T=5):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"{kind.upper()} train", ncols=100)
    for x, y in pbar:
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
        # Apply local plasticity AFTER optimizer step
        if kind == "bnn" and hasattr(model, "plastic") and model.plastic and z_h is not None:
            try:
                from pmflow_bnn import pm_local_plasticity
            except Exception:
                from pmflow_bnn.pmflow import pm_local_plasticity
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

# --- Main ---
def main():
    print("[INFO] Starting nn_lib demo_train...")
    headless = os.getenv("HEADLESS", "0") == "1"
    do_plot = os.getenv("PLOT", "0") == "1"
    if headless:
        matplotlib.use("Agg", force=True)
    set_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    train_loader, test_loader = make_loaders(batch=batch_size)

    bnn = PMBNNAlwaysPlastic(d_latent=8, channels=64, pm_steps=4, plastic=True).to(device)
    cnn = CNNBaseline().to(device)
    gru = GRUBaseline(hidden=128).to(device)

    opt_bnn = torch.optim.Adam(bnn.parameters(), lr=1e-3)
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    opt_gru = torch.optim.Adam(gru.parameters(), lr=1e-3)

    print(f"Params — BNN: {count_params(bnn):,} | CNN: {count_params(cnn):,} | GRU: {count_params(gru):,}")

    epochs = int(os.getenv("EPOCHS", "3"))
    acc_log = {"BNN": [], "CNN": [], "GRU": []}
    for ep in range(1, epochs+1):
        train_epoch(bnn, opt_bnn, train_loader, device, kind="bnn")
        train_epoch(cnn, opt_cnn, train_loader, device, kind="cnn")
        train_epoch(gru, opt_gru, train_loader, device, kind="gru")

        te_bnn = eval_model(bnn, test_loader, device, kind="bnn")
        te_cnn = eval_model(cnn, test_loader, device, kind="cnn")
        te_gru = eval_model(gru, test_loader, device, kind="gru")
        acc_log["BNN"].append(te_bnn)
        acc_log["CNN"].append(te_cnn)
        acc_log["GRU"].append(te_gru)

        print(f"Epoch {ep} — Test Acc: BNN={te_bnn:.3f} | CNN={te_cnn:.3f} | GRU={te_gru:.3f}")
    if do_plot:
        xs = list(range(1, epochs+1))
        plt.figure(figsize=(7.5, 4))
        plt.plot(xs, acc_log["BNN"], 'r^-', label='BNN')
        plt.plot(xs, acc_log["CNN"], 'bo-', label='CNN')
        plt.plot(xs, acc_log["GRU"], 'gs-', label='GRU')
        plt.xlabel('Epoch'); plt.ylabel('Test Accuracy'); plt.title('nn_lib demo: Test Accuracy'); plt.grid(True); plt.legend()
        plt.tight_layout(); plt.savefig('nn_lib_acc_curves.png', dpi=140)
        if not headless:
            plt.show()
        else:
            plt.close('all')
    print("[INFO] Finished nn_lib demo_train.")

if __name__ == "__main__":
    main()

