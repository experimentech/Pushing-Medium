#!/usr/bin/env python3
import argparse
import math
import os
import random
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    print("This demo requires PyTorch. Please install torch.")
    raise

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Try to make pmflow_bnn importable (env → local subdir)
PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

def setup_pmflow_bnn(nn_lib_path_override: Optional[str] = None):
    global PMFLOW_IMPORT, get_model_v2, get_performance_config
    PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}
    # environment first
    try:
        from pmflow_bnn import get_model_v2, get_performance_config  # type: ignore
        PMFLOW_IMPORT['available'] = True
        PMFLOW_IMPORT['source'] = 'environment'
        try:
            from pmflow_bnn.version import __version__  # type: ignore
            PMFLOW_IMPORT['version'] = __version__
        except Exception:
            PMFLOW_IMPORT['version'] = 'Development'
        return
    except Exception:
        pass
    # local fallback (nn_lib_v2 in repo)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", ".."))
    default_nn_lib_path = os.path.join(repo_root, "programs", "demos", "machine_learning", "nn_lib_v2")
    nn_lib_path = nn_lib_path_override or default_nn_lib_path
    if os.path.exists(nn_lib_path) and nn_lib_path not in sys.path:
        sys.path.insert(0, nn_lib_path)
    try:
        from pmflow_bnn import get_model_v2, get_performance_config  # type: ignore
        PMFLOW_IMPORT['available'] = True
        PMFLOW_IMPORT['source'] = 'local'
        try:
            from pmflow_bnn.version import __version__  # type: ignore
            PMFLOW_IMPORT['version'] = __version__
        except Exception:
            PMFLOW_IMPORT['version'] = 'Development'
    except Exception:
        PMFLOW_IMPORT['available'] = False
        PMFLOW_IMPORT['source'] = 'none'

# ----------------------------
# Utilities
# ----------------------------

def tokenize(s: str) -> List[str]:
    return s.strip().split()

class Vocab:
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        # ensure special tokens
        for tok in ["<pad>", "<unk>", "<bos>", "<eos>"]:
            self.add(tok)
    def add(self, w: str) -> int:
        if w not in self.stoi:
            self.stoi[w] = len(self.itos)
            self.itos.append(w)
        return self.stoi[w]
    def __len__(self):
        return len(self.itos)
    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in toks]
    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

# ----------------------------
# Plastic modules
# ----------------------------

class PlasticRNNCell(nn.Module):
    """A simple RNN cell (tanh) with additive plasticity on recurrent weights.
    y_t = tanh(W_ih x_t + W_hh h_{t-1} + b)
    Plasticity: W_hh += lr * e_t, where e_t is an eligibility trace updated with Hebbian-like outer products.
    """
    def __init__(self, input_size: int, hidden_size: int, lr: float = 1e-2, beta: float = 0.98):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size) * 0.05)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.05)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.lr = lr
        self.beta = beta
        # eligibility trace for W_hh
        self.register_buffer("elig", torch.zeros(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, input_size), h: (B, hidden_size)
        pre = x @ self.W_ih.T + h @ self.W_hh.T + self.b
        h_new = torch.tanh(pre)
        return h_new

    @torch.no_grad()
    def plastic_update(self, h_prev: torch.Tensor, h_new: torch.Tensor):
        # Update eligibility and recurrent weights using batch-averaged Hebbian term
        # elig = beta * elig + outer(h_new, h_prev)
        # W_hh += lr * elig
        hp = h_prev.mean(dim=0)  # (H)
        hn = h_new.mean(dim=0)   # (H)
        outer = torch.ger(hn, hp)  # (H,H)
        self.elig.mul_(self.beta).add_(outer)
        self.W_hh.add_(self.lr * self.elig)

class PlasticEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int, lr: float = 1e-2, beta: float = 0.98):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embed_dim) * 0.05)
        self.lr = lr
        self.beta = beta
        self.register_buffer("elig", torch.zeros_like(self.weight))
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]
    @torch.no_grad()
    def plastic_update(self, ids: torch.Tensor, grad_signal: torch.Tensor):
        # elig = beta * elig; elig[ids] += grad_signal; weight += lr * elig
        self.elig.mul_(self.beta)
        # scatter-add grad_signal to elig rows
        self.elig.index_add_(0, ids, grad_signal)
        self.weight.add_(self.lr * self.elig)

class PlasticLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, backend: Optional[nn.Module] = None,
                 backend_feat_dim: int = 0, lr: float = 1e-2, beta: float = 0.98):
        super().__init__()
        self.emb = PlasticEmbedding(vocab_size, embed_dim, lr=lr, beta=beta)
        self.rnn = PlasticRNNCell(embed_dim, hidden_size, lr=lr, beta=beta)
        self.backend = backend  # module with encode(x: Tensor[B,E]) -> Tensor[B,F]
        self.backend_feat_dim = backend_feat_dim
        in_dim = hidden_size + (backend_feat_dim if backend is not None else 0)
        self.out = nn.Linear(in_dim, vocab_size)

    def forward(self, ids: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ids: (B,), h: (B,H)
        x = self.emb(ids)  # (B,E)
        h_new = self.rnn(x, h)  # (B,H)
        if self.backend is not None:
            with torch.no_grad():
                f = self.backend.encode(x)  # (B,F)
            comb = torch.cat([h_new, f], dim=-1)
        else:
            comb = h_new
        logits = self.out(comb)  # (B,V)
        return logits, h_new, x

# ----------------------------
# Backends: BNN and CNN via nn_lib_v2
# ----------------------------

class ImageProjector(nn.Module):
    def __init__(self, in_dim: int, out_hw: int = 28):
        super().__init__()
        self.out_hw = out_hw
        self.fc = nn.Linear(in_dim, out_hw * out_hw)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = torch.tanh(self.fc(x))  # (B, 784)
        return img.view(x.size(0), 1, self.out_hw, self.out_hw)

class BNNBackend(nn.Module):
    def __init__(self, embed_dim: int, device: torch.device, n_feats: int = 64):
        super().__init__()
        self.device = device
        self.proj = ImageProjector(embed_dim).to(device)
        # Build BNN model using performance config but override n_classes as feature dim
        cfg = get_performance_config('cpu') if PMFLOW_IMPORT['available'] else {}
        cfg = {**cfg, 'n_classes': n_feats}
        model_type = cfg.pop('model_type', 'standard_v2')
        self.model = get_model_v2(model_type, **cfg).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.feature_dim = n_feats
    @torch.no_grad()
    def encode(self, x_embed: torch.Tensor) -> torch.Tensor:
        # x_embed: (B,E)
        img = self.proj(x_embed)  # (B,1,28,28)
        flat = img.view(img.size(0), -1)
        out = self.model(flat)
        if isinstance(out, tuple):
            out = out[0]
        return out  # (B, n_feats)

class CNNBackend(nn.Module):
    def __init__(self, embed_dim: int, device: torch.device, n_feats: int = 64):
        super().__init__()
        self.device = device
        self.proj = ImageProjector(embed_dim).to(device)
        try:
            from pmflow_bnn.baselines import CNNBaseline  # type: ignore
            self.model = CNNBaseline(n_classes=n_feats).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.ok = True
        except Exception:
            # Minimal local CNN fallback
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128*7*7, 256), nn.ReLU(),
                nn.Linear(256, n_feats)
            ).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.ok = True
        self.feature_dim = n_feats
    @torch.no_grad()
    def encode(self, x_embed: torch.Tensor) -> torch.Tensor:
        img = self.proj(x_embed)  # (B,1,28,28)
        return self.model(img)

# ----------------------------
# Sequence language model powered by BNN/CNN cores
# ----------------------------

class BNNSequenceCore(nn.Module):
    """Drives a PMFlow-BNN across a token sequence, maintaining its internal state h.
    We project token embeddings to 28x28, encode to latent z, then advance the BNN one step.
    """
    def __init__(self, embed_dim: int, device: torch.device, cfg_override: Optional[dict] = None):
        super().__init__()
        self.device = device
        self.project = ImageProjector(embed_dim).to(device)
        # Prefer always_plastic_v2 so the core adapts online
        cfg = get_performance_config('cpu') if PMFLOW_IMPORT['available'] else {}
        cfg = {**cfg, **(cfg_override or {})}
        model_type = cfg.pop('model_type', 'always_plastic_v2')
        self.bnn = get_model_v2(model_type, **cfg).to(device)
        self.bnn.train()  # enable plasticity for AlwaysPlastic
        # State dimension is the readout in_features
        self.state_dim = int(self.bnn.readout.in_features)
        self.register_buffer('h', torch.zeros(1, self.state_dim))
    def reset(self, batch_size: int = 1):
        self.h = torch.zeros(batch_size, self.state_dim, device=self.h.device)
    def step_embed(self, x_embed: torch.Tensor) -> torch.Tensor:
        # x_embed: (B,E)
        img = self.project(x_embed)
        flat = img.view(img.size(0), -1)
        # Encode to latent and advance one step
        z = self.bnn.enc(flat)
        # Use common step API if present; else fallback to one pipeline stage
        if hasattr(self.bnn, 'step'):
            _, h_new, _ = self.bnn.step(z, self.h.to(flat.device))
        else:
            # TemporalPipelineBNN fallback
            _, h_new, _ = self.bnn.pipeline_stage(z, self.h.to(flat.device), 0)
        self.h = h_new.detach()
        return self.h

class CNNMembraneCore(nn.Module):
    """CNN feature extractor with a leaky-membrane state to mimic BNN dynamics as a fallback."""
    def __init__(self, embed_dim: int, device: torch.device, n_feats: int = 128, channels: int = 96):
        super().__init__()
        self.device = device
        self.project = ImageProjector(embed_dim).to(device)
        try:
            from pmflow_bnn.baselines import CNNBaseline  # type: ignore
            self.cnn = CNNBaseline(n_classes=n_feats).to(device).eval()
        except Exception:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128*7*7, 256), nn.ReLU(),
                nn.Linear(256, n_feats)
            ).to(device).eval()
        for p in self.cnn.parameters():
            p.requires_grad_(False)
        self.map = nn.Linear(n_feats, channels).to(device)
        self.tau = 0.90
        self.gain = 0.10
        self.state_dim = channels
        self.register_buffer('h', torch.zeros(1, self.state_dim))
    def reset(self, batch_size: int = 1):
        self.h = torch.zeros(batch_size, self.state_dim, device=self.h.device)
    def step_embed(self, x_embed: torch.Tensor) -> torch.Tensor:
        img = self.project(x_embed)
        with torch.no_grad():
            f = self.cnn(img)
        h_proj = torch.tanh(self.map(f))
        self.h = self.tau * self.h + self.gain * h_proj
        return self.h

class SequenceBNNLanguageModel(nn.Module):
    """Language model that uses a BNN/CNN core for sequence memory and a trainable readout to vocab."""
    def __init__(self, vocab_size: int, embed_dim: int, core: nn.Module):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.05)
        self.core = core
        self.out = nn.Linear(self.core.state_dim, vocab_size)
    def reset(self, batch_size: int = 1):
        if hasattr(self.core, 'reset'):
            self.core.reset(batch_size)
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B,)
        x = self.emb(ids)
        h = self.core.step_embed(x)
        logits = self.out(h)
        return logits

# ----------------------------
# Data
# ----------------------------

def load_corpus(name: str) -> List[str]:
    if name == "tiny":
        text = """
        hello there how are you
        hello there what is up
        this is a tiny tiny corpus
        teach me your slang
        the model will adapt online
        words can change their meaning
        """
        return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    elif os.path.isfile(name):
        with open(name, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    else:
        print(f"Unknown corpus '{name}', using tiny.")
        return load_corpus("tiny")

# ----------------------------
# Training / Interactive Loop
# ----------------------------

def build_vocab(lines: List[str]) -> Vocab:
    V = Vocab()
    for ln in lines:
        for t in tokenize(ln):
            V.add(t)
    return V

def batchify(lines: List[str], V: Vocab) -> List[Tuple[List[int], List[int]]]:
    pairs = []
    for ln in lines:
        toks = ["<bos>"] + tokenize(ln) + ["<eos>"]
        ids = V.encode(toks)
        for i in range(len(ids) - 1):
            pairs.append(([ids[i]], [ids[i+1]]))  # next-token pairs
    return pairs

def _pca_2d(xs: np.ndarray) -> np.ndarray:
    # Center and compute top-2 principal components via SVD
    X = xs - xs.mean(axis=0, keepdims=True)
    # Use economy SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Project onto first 2 PCs
    comps = Vt[:2].T  # (D,2)
    return X @ comps  # (N,2)

def viz_embeddings(V: Vocab, E: torch.Tensor, title: str = "Embeddings"):
    if not HAVE_MPL:
        return
    xs = E.detach().cpu().numpy()
    words = V.itos[:min(200, len(V))]
    xs = xs[:len(words)]
    if xs.shape[1] >= 2 and len(xs) >= 2:
        pts = _pca_2d(xs)
    else:
        # Fallback: pad to 2D
        pad = np.zeros((xs.shape[0], 2))
        pad[:, :min(2, xs.shape[1])] = xs[:, :min(2, xs.shape[1])]
        pts = pad
    plt.clf()
    plt.scatter(pts[:,0], pts[:,1], s=20, c=np.arange(len(words)))
    for i, w in enumerate(words):
        plt.text(pts[i,0], pts[i,1], w, fontsize=7)
    plt.title(title)
    plt.pause(0.001)

def main():
    p = argparse.ArgumentParser(description="Plastic text demo: Teach Me Your Slang")
    p.add_argument("--corpus", default="tiny")
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--beta", type=float, default=0.98)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--viz", action="store_true")
    p.add_argument("--backend", choices=["auto","bnn","cnn","none"], default="auto",
                   help="Use nn_lib_v2 backend as feature extractor (BNN preferred, CNN fallback)")
    p.add_argument("--nn-lib-path", type=str, default=None)
    args = p.parse_args()

    lines = load_corpus(args.corpus)
    V = build_vocab(lines)
    pairs = batchify(lines, V)

    torch.manual_seed(0)
    # Setup backend
    setup_pmflow_bnn(args.nn_lib_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_module: Optional[nn.Module] = None
    backend_feat_dim = 0
    backend_choice = args.backend
    if backend_choice == "auto":
        if PMFLOW_IMPORT['available']:
            backend_choice = "bnn"
        else:
            backend_choice = "cnn"
    if backend_choice == "bnn" and PMFLOW_IMPORT['available']:
        backend_module = BNNBackend(args.embed_dim, device=device, n_feats=64)
        backend_feat_dim = backend_module.feature_dim
    elif backend_choice == "cnn":
        backend_module = CNNBackend(args.embed_dim, device=device, n_feats=64)
        backend_feat_dim = backend_module.feature_dim
    else:
        backend_choice = "none"
    print(f"Backend: {backend_choice} | pmflow: {PMFLOW_IMPORT['source']} {PMFLOW_IMPORT['version']}")

    # Build model: prefer BNN/CNN core; fallback to PlasticLanguageModel when backend=none
    if backend_choice == "none":
        model = PlasticLanguageModel(len(V), args.embed_dim, args.hidden,
                                     backend=backend_module, backend_feat_dim=backend_feat_dim,
                                     lr=args.lr, beta=args.beta).to(device)
        core_mode = "rnn+optional-backend"
    else:
        if backend_choice == "bnn":
            core = BNNSequenceCore(args.embed_dim, device=device)
        else:
            core = CNNMembraneCore(args.embed_dim, device=device)
        model = SequenceBNNLanguageModel(len(V), args.embed_dim, core).to(device)
        core_mode = f"sequence-core:{backend_choice}"

    if args.viz and HAVE_MPL:
        plt.ion()
    # Use appropriate embedding reference
    E = model.emb.weight if hasattr(model, 'emb') else model.emb.weight
    viz_embeddings(V, E, title="Init embeddings")

    # Warmup on tiny corpus pairs
    # Warmup: iterate simple next-token pairs
    if backend_choice == "none":
        h = torch.zeros(1, args.hidden, device=device)
    for step in range(min(args.steps, len(pairs))):
        x_id = torch.tensor(pairs[step][0], device=device)
        y_id = torch.tensor(pairs[step][1], device=device)
        if backend_choice == "none":
            logits, h_new, x_embed = model(x_id, h)
        else:
            logits = model(x_id)
        # loss and prediction
        loss = F.cross_entropy(logits, y_id)
        # pred = logits.argmax(dim=-1)
        # Plasticity update: compute a simple signed error signal for embeddings
        if backend_choice == "none":
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                oh = torch.zeros_like(probs)
                oh[0, y_id.item()] = 1.0
                err = (oh - probs)  # (1,V)
                combined_dim = model.out.in_features
                hidden_dim = model.rnn.hidden_size
                grad_combined = (model.out.weight.T @ err.T).T  # (1, combined_dim)
                grad_hidden = grad_combined[:, :hidden_dim]      # (1, H)
                grad_embed = (model.rnn.W_ih.T @ grad_hidden[0].unsqueeze(-1)).T  # (1,E)
                model.emb.plastic_update(x_id, grad_embed)
                model.rnn.plastic_update(h, h_new)
            h = h_new.detach()
        if (step+1) % 50 == 0 and args.viz and HAVE_MPL:
            E = model.emb.weight if hasattr(model, 'emb') else model.emb.weight
            viz_embeddings(V, E, title=f"Embeddings after {step+1} steps")

    print("\nInteractive mode — type text; ':add <word>' to add vocab; ':quit' to exit")
    if backend_choice == "none":
        h = torch.zeros(1, args.hidden, device=device)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print() ; break
        if not line:
            continue
        if line.startswith(":quit"):
            break
        if line.startswith(":add "):
            word = line.split(None, 1)[1].strip()
            if word:
                idx = V.add(word)
                # expand embedding matrix
                with torch.no_grad():
                    new_row = torch.randn(1, args.embed_dim) * 0.05
                    model.emb.weight.data = torch.cat([model.emb.weight.data, new_row], dim=0)
                    model.emb.elig = torch.zeros_like(model.emb.weight)
                    in_features = model.out.in_features
                    model.out.weight.data = torch.cat([model.out.weight.data, torch.randn(1, in_features) * 0.05], dim=0)
                    model.out.bias.data = torch.cat([model.out.bias.data, torch.zeros(1)], dim=0)
                print(f"Added '{word}' (id={idx})")
            continue

        toks = ["<bos>"] + tokenize(line)
        ids = V.encode(toks)
        for i in range(len(ids)):
            x_id = torch.tensor([ids[i]], device=device)
            if backend_choice == "none":
                logits, h_new, x_embed = model(x_id, h)
            else:
                logits = model(x_id)
            pred_id = int(logits.argmax(dim=-1).item())
            pred_tok = V.itos[pred_id] if 0 <= pred_id < len(V) else "<unk>"
            print(f"  [{i}] input='{V.itos[ids[i]]}'  →  pred='{pred_tok}'")
            # plasticity update using simple error signal toward the actual next token if available
            if i + 1 < len(ids):
                y_id = torch.tensor([ids[i+1]], device=device)
                if backend_choice == "none":
                    with torch.no_grad():
                        probs = F.softmax(logits, dim=-1)
                        oh = torch.zeros_like(probs)
                        oh[0, y_id.item()] = 1.0
                        err = (oh - probs)  # (1,V)
                        grad_combined = (model.out.weight.T @ err.T).T
                        hidden_dim = model.rnn.hidden_size
                        grad_hidden = grad_combined[:, :hidden_dim]
                        grad_embed = (model.rnn.W_ih.T @ grad_hidden[0].unsqueeze(-1)).T  # (1,E)
                        model.emb.plastic_update(x_id, grad_embed)
                        model.rnn.plastic_update(h, h_new)
                # For BNN/CNN cores, adaptation happens inside the core (AlwaysPlastic)
            if backend_choice == "none":
                h = h_new.detach()

        # finalize sequence
        x_id = torch.tensor([V.stoi["<eos>"]], device=device)
        if backend_choice == "none":
            _, h, _ = model(x_id, h)
            h = h.detach()
        else:
            _ = model(x_id)
        if args.viz and HAVE_MPL:
            E = model.emb.weight if hasattr(model, 'emb') else model.emb.weight
            viz_embeddings(V, E, title="Embeddings (interactive)")

if __name__ == "__main__":
    main()
