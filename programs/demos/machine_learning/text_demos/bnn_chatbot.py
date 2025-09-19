#!/usr/bin/env python3
"""
BNN Chatbot (freestanding demo)

Goal: A minimal chatbot that learns online as you talk to it.

Core idea:
- Token embedding -> small MLP to PMFlow latent z
- PMFlow-BNN AlwaysPlastic core advances internal state h each token
- Trainable linear readout maps h to vocab logits
- Online plasticity happens inside PMFlow (AlwaysPlastic), while we optionally nudge embeddings via a simple local rule

Controls:
- Type messages; the bot predicts the next token at each step and adapts
- Commands: :add <word>, :quit
"""
import argparse
import os
import sys
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- utilities --------
def tokenize(s: str) -> List[str]:
    return s.strip().split()

class Vocab:
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
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

def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return a_norm @ b_norm.T


# -------- pmflow_bnn setup --------
PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

def setup_pmflow_bnn(nn_lib_path_override: Optional[str] = None):
    global PMFLOW_IMPORT, get_model_v2, get_performance_config
    PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}
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


# -------- BNN-native chatbot model --------
class TokenToLatent(nn.Module):
    """Tiny MLP mapping token embedding -> PMFlow latent vector (d_latent)."""
    def __init__(self, embed_dim: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, max(64, d_latent*2)), nn.Tanh(),
            nn.Linear(max(64, d_latent*2), d_latent)
        )
    def forward(self, x):
        return self.net(x)


class BNNTextCore(nn.Module):
    """PMFlow-BNN core advanced one step per token using latent input.
    We bypass image projection and directly drive PMFlow latent via a small MLP.
    """
    def __init__(self, embed_dim: int, device: torch.device, d_latent: int = 12,
                 perf_profile: str = 'auto', pm_model_type: str = 'always_plastic_v2',
                 plasticity_lr: float = 1e-3, text_encoder: str = 'mlp', gru_hidden: int = 64,
                 plastic_every: int = 1, fast_mode: bool = False):
        super().__init__()
        self.device = device
        # Configure from performance profile and requested model
        cfg = get_performance_config(perf_profile) if PMFLOW_IMPORT['available'] else {}
        cfg_model = cfg.pop('model_type', None)
        model_type = pm_model_type or cfg_model or 'always_plastic_v2'
        cfg = {**cfg, 'd_latent': d_latent, 'plasticity_lr': plasticity_lr}
        self.bnn = get_model_v2(model_type, **cfg).to(device)
        self.bnn.train()
        self.text_encoder = text_encoder
        if text_encoder == 'gru':
            self.gru = nn.GRU(input_size=embed_dim, hidden_size=gru_hidden, batch_first=True).to(device)
            self.to_latent = nn.Linear(gru_hidden, d_latent).to(device)
        else:
            self.gru = None
            self.to_latent = TokenToLatent(embed_dim, d_latent).to(device)
        self.state_dim = int(self.bnn.readout.in_features)
        self.register_buffer('h', torch.zeros(1, self.state_dim))
        self.plastic_every = max(1, int(plastic_every))
        self.fast_mode = bool(fast_mode)
        self._t = 0
    def reset(self, batch_size: int = 1):
        self.h = torch.zeros(batch_size, self.state_dim, device=self.h.device)
    def step_embed(self, x_embed: torch.Tensor) -> torch.Tensor:
        if self.gru is not None:
            # Treat the single token as a seq len 1
            seq = x_embed.unsqueeze(1)  # (B,1,E)
            out, _ = self.gru(seq)
            z_in = self.to_latent(out[:, -1, :])
        else:
            z_in = self.to_latent(x_embed)
        self._t += 1
        # Fast path or decimated plasticity: emulate state advance without plasticity
        if self.fast_mode or (self.plastic_every > 1 and (self._t % self.plastic_every) != 0):
            # Forward through PM (no plasticity), project and add lateral EI
            z_evolved = self.bnn.pm(z_in)
            h_proj = torch.tanh(self.bnn.proj(z_evolved))
            h_new = 0.90 * self.h.to(x_embed.device) + 0.10 * h_proj
            h_final = h_new + self.bnn.ei(z_evolved, h_new)
            self.h = h_final.detach()
            return self.h
        # Full step with internal plasticity
        if hasattr(self.bnn, 'step'):
            _, h_new, _ = self.bnn.step(z_in, self.h.to(x_embed.device))
        else:
            _, h_new, _ = self.bnn.pipeline_stage(z_in, self.h.to(x_embed.device), 0)
        self.h = h_new.detach()
        return self.h

    @torch.no_grad()
    def phrase_plasticity(self, x_embeds: torch.Tensor, mu_lr: float = 1e-3, c_lr: float = 1e-3):
        """Apply additional PMFlow plasticity using a batch derived from a phrase/concept.
        x_embeds: (B, E) token embeddings corresponding to a teaching signal.
        """
        try:
            # Import vectorized_pm_plasticity from nn_lib_v2 if available
            from pmflow_bnn.pmflow import vectorized_pm_plasticity  # type: ignore
        except Exception:
            return
        # Map to latent
        if self.gru is not None:
            seq = x_embeds.unsqueeze(1)
            out, _ = self.gru(seq)
            z_in = self.to_latent(out[:, -1, :])
        else:
            z_in = self.to_latent(x_embeds)  # (B, d_latent)
        # Evolve through PM to get z_evolved
        z_evolved = self.bnn.pm(z_in)
        # Build batch h via one membrane step from zero
        h0 = torch.zeros(z_evolved.size(0), self.state_dim, device=z_evolved.device)
        h_proj = torch.tanh(self.bnn.proj(z_evolved))
        h_new = 0.90 * h0 + 0.10 * h_proj
        h_final = h_new + self.bnn.ei(z_evolved, h_new)
        # Apply vectorized plasticity update
        vectorized_pm_plasticity(self.bnn.pm, z_evolved, h_final, mu_lr=mu_lr, c_lr=c_lr)


class BNNChatbotLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, core: BNNTextCore):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.05)
        self.core = core
        self.out = nn.Linear(self.core.state_dim, vocab_size)
    def reset(self, batch: int = 1):
        self.core.reset(batch)
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        h = self.core.step_embed(x)
        return self.out(h)

    def expand_vocab(self, new_vocab_size: int, device: torch.device):
        """Rebuild output layer to match new vocab size while preserving learned rows."""
        if self.out.out_features == new_vocab_size:
            return
        old_out = self.out
        in_f = old_out.in_features
        old_classes = old_out.out_features
        new_out = nn.Linear(in_f, new_vocab_size).to(device)
        with torch.no_grad():
            # copy existing weights/bias into the head of the new layer
            new_out.weight[:old_classes].copy_(old_out.weight)
            new_out.bias[:old_classes].copy_(old_out.bias)
            # init new rows
            if new_vocab_size > old_classes:
                nn.init.normal_(new_out.weight[old_classes:], mean=0.0, std=0.05)
                nn.init.zeros_(new_out.bias[old_classes:])
        self.out = new_out

    def add_word(self, V: Vocab, word: str, device: torch.device) -> int:
        """Add word to vocab and expand embedding/output layers safely."""
        idx = V.add(word)
        # Expand embedding if needed
        if idx >= self.emb.num_embeddings:
            with torch.no_grad():
                old_w = self.emb.weight.data
                embed_dim = old_w.shape[1]
                new_row = torch.randn(1, embed_dim, device=device) * 0.05
                new_weight = torch.cat([old_w, new_row], dim=0)
            new_emb = nn.Embedding(new_weight.shape[0], embed_dim).to(device)
            with torch.no_grad():
                new_emb.weight.copy_(new_weight)
            self.emb = new_emb
        # Expand output layer to new vocab size
        self.expand_vocab(len(V), device)
        return idx

    def expand_to_vocab(self, V: Vocab, device: torch.device):
        """Ensure embedding/output match V size, expanding in bulk efficiently."""
        target = len(V)
        if self.emb.num_embeddings < target:
            with torch.no_grad():
                old_w = self.emb.weight.data
                embed_dim = old_w.shape[1]
                extra = target - old_w.shape[0]
                new_rows = torch.randn(extra, embed_dim, device=device) * 0.05
                new_weight = torch.cat([old_w, new_rows], dim=0)
            new_emb = nn.Embedding(new_weight.shape[0], embed_dim).to(device)
            with torch.no_grad():
                new_emb.weight.copy_(new_weight)
            self.emb = new_emb
        self.expand_vocab(target, device)


# -------- simple associative memory (co-occurrence and concepts) --------
class AssocMemory:
    def __init__(self):
        self.co_counts: Dict[Tuple[int,int], int] = {}
        self.concepts: Dict[str, List[int]] = {}
        self.def_map: Dict[int, List[int]] = {}  # word_id -> phrase token ids
        self.token_freq: Dict[int, int] = {}
    def add_cooccurrence(self, ids: List[int], window: int = 2):
        n = len(ids)
        for i in range(n):
            # token frequency for biasing
            self.token_freq[ids[i]] = self.token_freq.get(ids[i], 0) + 1
            for j in range(max(0, i-window), min(n, i+window+1)):
                if i == j: continue
                a, b = ids[i], ids[j]
                if a > b: a, b = b, a
                key = (a, b)
                self.co_counts[key] = self.co_counts.get(key, 0) + 1
    def top_pairs(self, k: int = 10) -> List[Tuple[Tuple[int,int], int]]:
        return sorted(self.co_counts.items(), key=lambda kv: -kv[1])[:k]
    def set_concept(self, name: str, word_ids: List[int]):
        self.concepts[name] = list(dict.fromkeys(word_ids))
    def get_concept(self, name: str) -> List[int]:
        return self.concepts.get(name, [])


# light stopword list for pattern answers and biasing
STOPWORDS = {
    "<pad>", "<unk>", "<bos>", "<eos>",
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "is", "are", "was", "were", "be", "been", "am",
    "what", "who", "where", "when", "why", "how", "do", "does", "did", "you", "i", "we", "they", "he", "she", "it",
    "that", "this", "these", "those", "with", "for", "as", "at", "by", "from", "but", "if",
    # common Project Gutenberg header/footer tokens
    "project", "gutenberg", "license", "ebook", "ebooks", "please", "www.gutenberg.org", "1.e.8", "1.e.9", "1.e.10"
}


# -------- main / REPL --------
def build_vocab_from_corpus(lines: List[str]) -> Vocab:
    V = Vocab()
    for ln in lines:
        for t in tokenize(ln):
            V.add(t)
    return V

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
        return load_corpus("tiny")


def main():
    ap = argparse.ArgumentParser(description="BNN Chatbot (text-native PMFlow-BNN)")
    ap.add_argument("--corpus", default="tiny")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--latent", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--nn-lib-path", type=str, default=None)
    ap.add_argument("--pmflow-profile", type=str, default="auto", choices=["auto","cpu","single_gpu","multi_gpu","jetson_nano"],
                    help="Hardware profile for PMFlow configuration")
    ap.add_argument("--pmflow-model", type=str, default="always_plastic_v2",
                    choices=["always_plastic_v2","temporal_pipeline","standard_v2"],
                    help="PMFlow-BNN model variant to use")
    ap.add_argument("--plasticity-lr", type=float, default=1e-3, help="PMFlow plasticity rate (mu/center)")
    ap.add_argument("--text-encoder", type=str, default="mlp", choices=["mlp","gru"], help="Front-end encoder before PMFlow")
    ap.add_argument("--gru-hidden", type=int, default=64, help="Hidden size for GRU text encoder")
    ap.add_argument("--train-script", type=str, action='append', default=None,
                    help="Path to a training script with commands/utterances to bootstrap state. Can be passed multiple times.")
    ap.add_argument("--train-quiet", action='store_true',
                    help="Suppress output while processing training scripts (faster; no per-line prints or generation).")
    ap.add_argument("--fast-repl", action='store_true', help="Faster REPL: state advance without PM plasticity (plasticity decimated).")
    ap.add_argument("--plastic-every", type=int, default=1, help="Apply PM plasticity every N tokens (>=1).")
    ap.add_argument("--renorm-head", action='store_true', help="L2-renormalize output head rows after bulk ingestion.")
    ap.add_argument("--consolidate-steps", type=int, default=0, help="Extra readout-only SGD steps after training scripts (consolidation).")
    args = ap.parse_args()

    setup_pmflow_bnn(args.nn_lib_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pmflow: {PMFLOW_IMPORT['source']} {PMFLOW_IMPORT['version']} | device: {device}")
    if not PMFLOW_IMPORT['available']:
        print("PMFlow BNN not available — this demo requires nn_lib_v2.")
        sys.exit(1)

    lines = load_corpus(args.corpus)
    V = build_vocab_from_corpus(lines)

    core = BNNTextCore(args.embed_dim, device=device, d_latent=args.latent,
                       perf_profile=args.pmflow_profile, pm_model_type=args.pmflow_model,
                       plasticity_lr=args.plasticity_lr, text_encoder=args.text_encoder,
                       gru_hidden=args.gru_hidden, plastic_every=args.plastic_every,
                       fast_mode=args.fast_repl)
    model = BNNChatbotLM(len(V), args.embed_dim, core).to(device)
    if args.renorm_head:
        with torch.no_grad():
            w = model.out.weight
            norms = w.norm(dim=1, keepdim=True).clamp_min(1e-6)
            w.div_(norms)
    assoc = AssocMemory()

    # Warmup over next-token pairs
    pairs: List[Tuple[int, int]] = []
    for ln in lines:
        ids = V.encode(["<bos>"] + tokenize(ln) + ["<eos>"])
        pairs.extend([(ids[i], ids[i+1]) for i in range(len(ids)-1)])
    model.reset(1)
    for step in range(min(args.steps, len(pairs))):
        x_id = torch.tensor([pairs[step][0]], device=device)
        y_id = torch.tensor([pairs[step][1]], device=device)
        logits = model(x_id)
        loss = F.cross_entropy(logits, y_id)
        # Train only the readout (simple SGD)
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.startswith('out.') and p.grad is not None:
                    p.add_(-1e-2 * p.grad)
            model.zero_grad(set_to_none=True)

    print("\nChatbot ready — type text; commands: :add, :define, :alias, :neighbors, :concept, :tag, :gen, :stats, :save, :load, :quit")
    model.reset(1)
    # live settings
    cfg = {
        'gen_tokens': 12,
        'gen_temp': 0.9,
        'gen_topk': 8,
        'top_p': 1.0,          # nucleus sampling cutoff (<=1 disables)
        'rep_penalty': 1.0,    # >= 1.0; >1 discourages repeats
        'emb_alpha': 0.08,
        'bias_co': 0.0,        # co-occurrence bias strength
        'bias_concept': 0.0    # concept membership bias strength
    }

    def process_line(line: str, quiet: bool = False) -> bool:
        """Process one command/utterance. Return False to exit."""
        if not line:
            return True
        if line.startswith(":quit"):
            return False
        if line.startswith(":add "):
            word = line.split(None, 1)[1].strip()
            if word:
                idx = model.add_word(V, word, device)
                if not quiet:
                    print(f"Added '{word}' (id={idx})")
            return True
        if line.startswith(":define ") and "=" in line:
            try:
                _, rest = line.split(":define ", 1)
                wpart, phrase = rest.split("=", 1)
                w = wpart.strip()
                phrase_toks = tokenize(phrase)
                if not w or not phrase_toks:
                    if not quiet:
                        print("Usage: :define <word> = <phrase>")
                    return True
                if w not in V.stoi:
                    idx = model.add_word(V, w, device)
                    if not quiet:
                        print(f"[auto-add] '{w}' (id={idx})")
                # auto-add phrase tokens
                for t in phrase_toks:
                    if t not in V.stoi:
                        idx = model.add_word(V, t, device)
                        if not quiet:
                            print(f"[auto-add] '{t}' (id={idx})")
                wid = V.stoi[w]
                p_ids = [i for i in V.encode(phrase_toks) if i < len(V)]
                with torch.no_grad():
                    if p_ids:
                        mean_vec = model.emb.weight[p_ids].mean(dim=0, keepdim=True)
                        model.emb.weight[wid:wid+1] = (1-cfg['emb_alpha']) * model.emb.weight[wid:wid+1] + cfg['emb_alpha'] * mean_vec
                if p_ids:
                    x_embeds = model.emb.weight[p_ids]
                    core.phrase_plasticity(x_embeds, mu_lr=args.plasticity_lr, c_lr=args.plasticity_lr)
                # store definition mapping
                assoc.def_map[wid] = p_ids
                if not quiet:
                    print(f"Defined '{w}' from phrase of {len(p_ids)} tokens.")
            except Exception as e:
                if not quiet:
                    print(f"define error: {e}")
            return True
        if line.startswith(":alias "):
            parts = line.split()
            if len(parts) >= 3:
                w1, w2 = parts[1], parts[2]
                if w1 not in V.stoi:
                    idx = model.add_word(V, w1, device)
                    if not quiet:
                        print(f"[auto-add] '{w1}' (id={idx})")
                if w2 not in V.stoi:
                    idx = model.add_word(V, w2, device)
                    if not quiet:
                        print(f"[auto-add] '{w2}' (id={idx})")
                i1, i2 = V.stoi[w1], V.stoi[w2]
                with torch.no_grad():
                    mean = 0.5 * (model.emb.weight[i1] + model.emb.weight[i2])
                    model.emb.weight[i1] = (1-cfg['emb_alpha']) * model.emb.weight[i1] + cfg['emb_alpha'] * mean
                    model.emb.weight[i2] = (1-cfg['emb_alpha']) * model.emb.weight[i2] + cfg['emb_alpha'] * mean
                if not quiet:
                    print(f"Aliased '{w1}' ~ '{w2}'")
            else:
                if not quiet:
                    print("Usage: :alias <word1> <word2>")
            return True
        if line.startswith(":neighbors "):
            parts = line.split()
            w = parts[1] if len(parts) >= 2 else None
            k = int(parts[2]) if len(parts) >= 3 else 8
            if w is None:
                if not quiet:
                    print("Usage: :neighbors <word> [k]")
            else:
                if w not in V.stoi:
                    idx = model.add_word(V, w, device)
                    if not quiet:
                        print(f"[auto-add] '{w}' (id={idx})")
                wid = V.stoi[w]
                with torch.no_grad():
                    sims = cosine_sim(model.emb.weight[wid:wid+1], model.emb.weight)[0]
                    vals, idxs = torch.topk(sims, k=min(k+1, len(V)))
                    out = []
                    for val, idx in zip(vals.tolist(), idxs.tolist()):
                        if idx == wid: continue
                        out.append((V.itos[idx], val))
                        if len(out) >= k: break
                if not quiet:
                    print("Neighbors:", ", ".join(f"{t}({s:.2f})" for t,s in out))
            return True
        if line.startswith(":concept ") and "=" in line:
            try:
                _, rest = line.split(":concept ", 1)
                name, words = rest.split("=", 1)
                name = name.strip()
                toks = tokenize(words)
                # auto-add tokens
                for t in toks:
                    if t not in V.stoi:
                        idx = model.add_word(V, t, device)
                        if not quiet:
                            print(f"[auto-add] '{t}' (id={idx})")
                ids = [V.stoi[t] for t in toks if t in V.stoi]
                if not ids:
                    if not quiet:
                        print("No known tokens for concept.")
                else:
                    assoc.set_concept(name, ids)
                    with torch.no_grad():
                        centroid = model.emb.weight[ids].mean(dim=0, keepdim=True)
                        for i in ids:
                            model.emb.weight[i:i+1] = (1-cfg['emb_alpha']) * model.emb.weight[i:i+1] + cfg['emb_alpha'] * centroid
                    if ids:
                        x_embeds = model.emb.weight[ids]
                        core.phrase_plasticity(x_embeds, mu_lr=args.plasticity_lr, c_lr=args.plasticity_lr)
                    if not quiet:
                        print(f"Concept '{name}' set with {len(ids)} tokens.")
            except Exception as e:
                if not quiet:
                    print(f"concept error: {e}")
            return True
        if line.startswith(":tag "):
            parts = line.split()
            if len(parts) >= 3:
                w, name = parts[1], parts[2]
                # auto-add word
                if w not in V.stoi:
                    idx = model.add_word(V, w, device)
                    if not quiet:
                        print(f"[auto-add] '{w}' (id={idx})")
                # auto-create concept if missing
                if name not in assoc.concepts:
                    assoc.set_concept(name, [])
                    if not quiet:
                        print(f"[auto-create] concept '{name}'")
                ids = assoc.get_concept(name)
                if len(ids) > 0:
                    with torch.no_grad():
                        centroid = model.emb.weight[ids].mean(dim=0, keepdim=True)
                        i = V.stoi[w]
                        model.emb.weight[i:i+1] = (1-cfg['emb_alpha']) * model.emb.weight[i:i+1] + cfg['emb_alpha'] * centroid
                assoc.set_concept(name, ids + [V.stoi[w]])
                if not quiet:
                    print(f"Tagged '{w}' to concept '{name}'.")
            else:
                if not quiet:
                    print("Usage: :tag <word> <concept>")
            return True
        if line.startswith(":gen"):
            parts = line.split()
            if len(parts) >= 2: cfg['gen_tokens'] = int(parts[1])
            if len(parts) >= 3: cfg['gen_temp'] = float(parts[2])
            if len(parts) >= 4: cfg['gen_topk'] = int(parts[3])
            if len(parts) >= 5: cfg['top_p'] = float(parts[4])
            if len(parts) >= 6: cfg['rep_penalty'] = float(parts[5])
            if not quiet:
                print(f"Generation set: tokens={cfg['gen_tokens']} temp={cfg['gen_temp']} topk={cfg['gen_topk']} top_p={cfg['top_p']} rep_penalty={cfg['rep_penalty']}")
            return True
        if line.startswith(":bias"):
            # Usage: :bias co 0.2 concept 0.5
            parts = line.split()
            i = 1
            while i + 1 < len(parts):
                key = parts[i].lower(); val = parts[i+1]
                try:
                    f = float(val)
                except Exception:
                    i += 2; continue
                if key in ('co', 'cooccur', 'cooccurrence'):
                    cfg['bias_co'] = f
                if key in ('concept', 'concepts'):
                    cfg['bias_concept'] = f
                i += 2
            if not quiet:
                print(f"Bias set: co={cfg['bias_co']} concept={cfg['bias_concept']}")
            return True
        if line.startswith(":speed "):
            mode = line.split(None, 1)[1].strip().lower()
            if mode in ("fast","normal"):
                model.core.fast_mode = (mode == "fast")
                if not quiet:
                    print(f"Speed mode set to {mode}")
            else:
                if not quiet:
                    print("Usage: :speed fast|normal")
            return True
        if line.startswith(":plastic-every "):
            try:
                n = int(line.split(None, 1)[1].strip())
                model.core.plastic_every = max(1, n)
                if not quiet:
                    print(f"PM plasticity frequency set to every {model.core.plastic_every} token(s)")
            except Exception:
                if not quiet:
                    print("Usage: :plastic-every <N>")
            return True
        if line.startswith(":stats"):
            pairs = assoc.top_pairs(10)
            out = []
            for (a,b), ccount in pairs:
                out.append(f"{V.itos[a]}~{V.itos[b]}({ccount})")
            if not quiet:
                print("Top pairs:", ", ".join(out))
            return True
        if line.startswith(":save "):
            path = line.split(None, 1)[1].strip()
            try:
                state = {
                    'itos': V.itos,
                    'emb': model.emb.weight.detach().cpu(),
                    'out_w': model.out.weight.detach().cpu(),
                    'out_b': model.out.bias.detach().cpu(),
                    'to_latent': model.core.to_latent.state_dict(),
                    'text_encoder': model.core.text_encoder,
                }
                if model.core.text_encoder == 'gru' and hasattr(model.core, 'gru') and model.core.gru is not None:
                    state['gru'] = model.core.gru.state_dict()
                torch.save(state, path)
                if not quiet:
                    print(f"Saved to {path}")
            except Exception as e:
                if not quiet:
                    print(f"save error: {e}")
            return True
        if line.startswith(":load "):
            path = line.split(None, 1)[1].strip()
            try:
                state = torch.load(path, map_location=device)
                itos = state['itos']
                if itos != V.itos:
                    V.itos = itos
                    V.stoi = {w:i for i,w in enumerate(itos)}
                with torch.no_grad():
                    model.emb.weight.data = state['emb'].to(device)
                    model.out.weight.data = state['out_w'].to(device)
                    model.out.bias.data = state['out_b'].to(device)
                    model.core.to_latent.load_state_dict(state['to_latent'])
                    if state.get('text_encoder','mlp') == 'gru' and hasattr(model.core,'gru') and model.core.gru is not None and 'gru' in state:
                        model.core.gru.load_state_dict(state['gru'])
                if not quiet:
                    print(f"Loaded from {path}")
            except Exception as e:
                if not quiet:
                    print(f"load error: {e}")
            return True
        # Utterance: auto-add new words
        toks = ["<bos>"] + tokenize(line)
        new_words = [t for t in toks if t not in V.stoi]
        for w in new_words:
            idx = model.add_word(V, w, device)
            if not quiet:
                print(f"[auto-add] '{w}' (id={idx})")
        ids = V.encode(toks)
        # Simple pattern QA using definitions and concepts
        if not quiet and len(ids) > 2:
            # e.g., "what is X" or "what colour is X"
            ltoks = [t.lower() for t in tokenize(line)]
            if ("what" in ltoks and "is" in ltoks):
                # find candidate subject X as the last non-stopword
                subj = None
                for t in reversed(ltoks):
                    if t not in STOPWORDS and t in V.stoi:
                        subj = t; break
                if subj is not None:
                    wid = V.stoi[subj]
                    if wid in assoc.def_map and assoc.def_map[wid]:
                        ans = " ".join(V.itos[i] for i in assoc.def_map[wid] if V.itos[i] not in STOPWORDS)
                        print(f"bot: {subj} is {ans}")
                        return True
                    # concept lookup: find a concept that contains subj
                    for cname, mids in assoc.concepts.items():
                        if wid in mids:
                            print(f"bot: {subj} is a {cname}")
                            return True
        # Co-occurrence and context shaping
        assoc.add_cooccurrence(ids, window=2)
        with torch.no_grad():
            for i in range(len(ids)):
                ctx = [ids[j] for j in range(max(0, i-2), min(len(ids), i+3)) if j != i]
                if not ctx: continue
                mean_vec = model.emb.weight[ctx].mean(dim=0, keepdim=True)
                model.emb.weight[ids[i]:ids[i]+1] = (1-0.02) * model.emb.weight[ids[i]:ids[i]+1] + 0.02 * mean_vec
        # Predict and online readout update (skip when quiet for speed)
        if not quiet:
            for i, tok_id in enumerate(ids):
                x_id = torch.tensor([tok_id], device=device)
                logits = model(x_id)
                pred_id = int(logits.argmax(dim=-1).item())
                pred_tok = V.itos[pred_id] if 0 <= pred_id < len(V) else "<unk>"
                if not quiet:
                    print(f"  [{i}] input='{V.itos[tok_id]}' → pred='{pred_tok}'")
                if i + 1 < len(ids):
                    y_id = torch.tensor([ids[i+1]], device=device)
                    loss = F.cross_entropy(logits, y_id)
                    loss.backward()
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if n.startswith('out.') and p.grad is not None:
                                p.add_(-1e-2 * p.grad)
                        model.zero_grad(set_to_none=True)
        # finalize with eos and generate reply (skip when quiet)
        if not quiet:
            eos = torch.tensor([V.stoi["<eos>"]], device=device)
            _ = model(eos)
            reply_ids: List[int] = []
            cur_id = torch.tensor([V.stoi.get('<bos>', 2)], device=device)
        toks_plain = ["<bos>"] + tokenize(line)
        # Precompute concept triggers for biasing
        concept_triggers = set()
        if cfg['bias_concept'] > 0.0:
            names_in_text = set(toks_plain)
            for cname, members in assoc.concepts.items():
                # triggered if concept name appears, or any member appears
                if cname in names_in_text or any((m in V.encode(toks_plain)) for m in members):
                    concept_triggers.update(members)
        # Skip generation when quiet to speed scanning large files
        if not quiet:
            for _ in range(cfg['gen_tokens']):
                logits = model(cur_id)  # (1, V)
                # repetition penalty and immediate repeat block
                if cfg['rep_penalty'] > 1.0 and reply_ids:
                    with torch.no_grad():
                        for seen in set(reply_ids):
                            if 0 <= seen < logits.size(-1):
                                logits[0, seen] = logits[0, seen] / cfg['rep_penalty']
                        # block immediate self-repeat
                        last = reply_ids[-1]
                        if 0 <= last < logits.size(-1):
                            logits[0, last] = logits[0, last] - 1e9
                # candidate pool for biasing and sampling
                cand_pool_k = min(max(cfg['gen_topk'], 64), logits.size(-1))
                pool_vals, pool_idxs = torch.topk(logits, k=cand_pool_k, dim=-1)
                # apply co-occurrence and concept biases to candidate pool
                with torch.no_grad():
                    if cfg['bias_co'] > 0.0 and len(pool_idxs[0]) > 0:
                        recent_ids = [i for i in ids[-5:]]  # last few tokens from input
                        for j, cand in enumerate(pool_idxs[0].tolist()):
                            co_score = 0.0
                            for r in recent_ids:
                                a, b = (r, cand) if r <= cand else (cand, r)
                                co_score += float(assoc.co_counts.get((a, b), 0))
                            if co_score > 0:
                                logits[0, cand] = logits[0, cand] + cfg['bias_co'] * co_score
                    if cfg['bias_concept'] > 0.0 and concept_triggers:
                        for cand in pool_idxs[0].tolist():
                            if cand in concept_triggers:
                                logits[0, cand] = logits[0, cand] + cfg['bias_concept']
                    # downweight frequent tokens and stopwords to avoid degenerate outputs
                    for cand in pool_idxs[0].tolist():
                        tok = V.itos[cand]
                        if tok in STOPWORDS:
                            logits[0, cand] -= 0.5
                        freq = assoc.token_freq.get(cand, 0)
                        if freq > 0:
                            logits[0, cand] -= 0.0005 * min(freq, 5000)
                # temperature
                logits = logits / max(1e-6, cfg['gen_temp'])
                # top-p or top-k sampling within candidate pool
                pool_vals, pool_idxs = torch.topk(logits, k=cand_pool_k, dim=-1)
                probs = F.softmax(pool_vals, dim=-1)[0]
                if cfg['top_p'] < 1.0:
                    # sort by prob desc and keep cumulative <= top_p
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=0)
                    mask = cum <= cfg['top_p']
                    # ensure at least one token
                    if not mask.any():
                        mask[0] = True
                    kept = sorted_idx[mask]
                    probs = sorted_probs[mask]
                    cand_pool = pool_idxs[0][kept]
                else:
                    topk = min(cfg['gen_topk'], probs.size(0))
                    cand_pool = pool_idxs[0][:topk]
                    probs = probs[:topk]
                # sample
                samp = torch.multinomial(probs, num_samples=1)
                next_id = cand_pool[samp]
                nid = int(next_id.item())
                if V.itos[nid] == '<eos>':
                    break
                reply_ids.append(nid)
                cur_id = torch.tensor([nid], device=device)
        if not quiet:
            print("bot:", " ".join(V.itos[i] for i in reply_ids) if reply_ids else "(no output)")
        return True

    # Optional one-off training scripts (can be multiple)
    if args.train_script is not None:
        # Pre-scan scripts to expand vocab in bulk (avoid O(N^2) dynamic growth)
        pre_words = set()
        for script_path in args.train_script:
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    for raw in f:
                        ln = raw.strip()
                        if not ln:
                            continue
                        if ln.startswith(":define ") and "=" in ln:
                            try:
                                _, rest = ln.split(":define ", 1)
                                wpart, phrase = rest.split("=", 1)
                                pre_words.add(wpart.strip())
                                pre_words.update(tokenize(phrase))
                            except Exception:
                                pass
                            continue
                        if ln.startswith(":concept ") and "=" in ln:
                            try:
                                _, rest = ln.split(":concept ", 1)
                                _name, words = rest.split("=", 1)
                                pre_words.update(tokenize(words))
                            except Exception:
                                pass
                            continue
                        if ln.startswith(":alias "):
                            parts = ln.split()
                            if len(parts) >= 3:
                                pre_words.add(parts[1]); pre_words.add(parts[2])
                            continue
                        if ln.startswith(":tag "):
                            parts = ln.split()
                            if len(parts) >= 2:
                                pre_words.add(parts[1])
                            continue
                        # regular utterance line
                        pre_words.update(tokenize(ln))
            except Exception as e:
                print(f"[pre-scan warning] could not read '{script_path}': {e}")
        # add to vocab (no prints), then expand model once
        added = 0
        for w in pre_words:
            if w not in V.stoi:
                V.add(w); added += 1
        if added > 0:
            model.expand_to_vocab(V, device)
            print(f"[pre-scan] expanded vocab by {added} tokens (now {len(V)})")
        total_lines = 0
        for script_path in args.train_script:
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    for raw in f:
                        script_line = raw.rstrip('\n')
                        if not script_line.strip():
                            continue
                        total_lines += 1
                        if not args.train_quiet:
                            print(f"# {script_line}")
                        if not process_line(script_line, quiet=args.train_quiet):
                            return
            except Exception as e:
                print(f"Failed to run train script '{script_path}': {e}")
        # Signal REPL readiness explicitly
        print(f"[training] processed {total_lines} lines from {len(args.train_script)} file(s). Entering REPL…", flush=True)
        # Optional head renorm and consolidation after ingestion
        if args.renorm_head:
            with torch.no_grad():
                w = model.out.weight
                norms = torch.norm(w, dim=1, keepdim=True) + 1e-8
                w.div_(norms)
        if args.consolidate_steps > 0 and not args.train_quiet:
            # build simple next-token pairs from seen vocab tokens (heuristic)
            # Here we just do random pairs from vocab as a light consolidation.
            import random
            model.reset(1)
            for _ in range(args.consolidate_steps):
                a = random.randrange(len(V)); b = random.randrange(len(V))
                x_id = torch.tensor([a], device=device)
                y_id = torch.tensor([b], device=device)
                logits = model(x_id)
                loss = F.cross_entropy(logits, y_id)
                loss.backward()
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n.startswith('out.') and p.grad is not None:
                            p.add_(-5e-3 * p.grad)
                    model.zero_grad(set_to_none=True)

        # Optional consolidation pass to adapt the readout head
        if args.consolidate_steps > 0:
            def pairs_from_files(paths, cap: int):
                yielded = 0
                for p in paths:
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            for raw in f:
                                if yielded >= cap:
                                    return
                                ln = raw.strip()
                                if not ln or ln.startswith(":"):
                                    continue
                                ids = V.encode(["<bos>"] + tokenize(ln) + ["<eos>"])
                                for i in range(len(ids)-1):
                                    yield (ids[i], ids[i+1])
                                    yielded += 1
                                    if yielded >= cap:
                                        return
                    except Exception:
                        continue
            model.reset(1)
            steps = 0
            for x, y in pairs_from_files(args.train_script, args.consolidate_steps):
                x_id = torch.tensor([x], device=device)
                y_id = torch.tensor([y], device=device)
                logits = model(x_id)
                loss = F.cross_entropy(logits, y_id)
                loss.backward()
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n.startswith('out.') and p.grad is not None:
                            p.add_(-1e-2 * p.grad)
                    model.zero_grad(set_to_none=True)
                steps += 1
            if not args.train_quiet:
                print(f"[consolidate] adapted readout on {steps} pairs")

    # Interactive REPL
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not process_line(line):
            break


if __name__ == "__main__":
    main()
