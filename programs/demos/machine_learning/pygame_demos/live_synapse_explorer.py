#!/usr/bin/env python3
"""
Live Synapse Explorer (pygame)

- Visual: A grid of neurons with animated connections whose thickness/opacity reflects synaptic weight
- Interaction: User paints input patterns with the mouse; keyboard toggles learning and parameters
- BNN hook: Optionally uses pmflow_bnn to compute features from the grid activity and inject an additional learned drive
- Why: Watch the network visibly rewire itself in response to your inputs (Hebbian plasticity in real time)

Controls:
  Mouse drag: Paint input into neurons under the cursor
  [Space] Pause
  [L] Toggle learning on/off
  [C] Clear activations
  [W] Reset weights (small random)
  [+]/[-] Increase/Decrease learning rate
  [D] Toggle higher weight decay
  [B] Toggle BNN assist (if available)
  [Q/Esc] Quit

Flags:
  --grid-w INT, --grid-h INT
  --adapter {auto,bnn,none}
  --device {auto,cpu,cuda}
  --no-github-install
  --nn-lib-path PATH
  --fps INT
"""
import os
import sys
import math
import time
import random
import argparse
import subprocess
import numpy as np
import pygame

# pmflow_bnn setup (import-first, then optional github, then local path)
PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

def setup_pmflow_bnn(nn_lib_path_override=None, allow_github_install=True):
    global PMFLOW_IMPORT, get_model_v2, get_performance_config, torch
    PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", ".."))
    default_nn_lib_path = os.path.join(repo_root, "programs", "demos", "machine_learning", "nn_lib_v2")
    nn_lib_path = nn_lib_path_override or default_nn_lib_path

    # Try current environment first
    try:
        import torch  # noqa: F401
        from pmflow_bnn import get_model_v2, get_performance_config  # type: ignore
        PMFLOW_IMPORT['available'] = True
        PMFLOW_IMPORT['source'] = 'environment'
        try:
            from pmflow_bnn.version import __version__  # type: ignore
            PMFLOW_IMPORT['version'] = __version__
        except Exception:
            PMFLOW_IMPORT['version'] = 'Development'
        print("✅ PMFlow BNN found in current environment")
        print(f"📦 Version: {PMFLOW_IMPORT['version']}")
        return
    except Exception:
        pass

    # Optional GitHub install
    install_outcome = 'local'
    if allow_github_install:
        try:
            print("🚀 Attempting to install PMFlow BNN v0.2.0 from GitHub...")
            print("📦 Installing: git+https://github.com/experimentech/Pushing-Medium.git#subdirectory=programs/demos/machine_learning/nn_lib_v2")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/experimentech/Pushing-Medium.git#subdirectory=programs/demos/machine_learning/nn_lib_v2'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                install_outcome = 'github'
            else:
                print(f"⚠️ GitHub installation failed: {result.stderr.splitlines()[-1] if result.stderr else 'unknown error'}")
        except Exception as e:
            print(f"⚠️ GitHub installation error: {e}")

    # Local path fallback
    if install_outcome != 'github':
        if os.path.exists(nn_lib_path) and nn_lib_path not in sys.path:
            sys.path.insert(0, nn_lib_path)
            print(f"📂 Added local library path: {nn_lib_path}")

    # Import
    try:
        import torch  # noqa: F401
        from pmflow_bnn import get_model_v2, get_performance_config  # type: ignore
        PMFLOW_IMPORT['available'] = True
        PMFLOW_IMPORT['source'] = install_outcome
        try:
            from pmflow_bnn.version import __version__  # type: ignore
            PMFLOW_IMPORT['version'] = __version__
        except Exception:
            PMFLOW_IMPORT['version'] = 'Development'
        print("✅ PMFlow BNN library imported successfully")
        print(f"📍 Installation source: {PMFLOW_IMPORT['source']}")
        print(f"📦 Version: {PMFLOW_IMPORT['version']}")
    except Exception as e:
        PMFLOW_IMPORT['available'] = False
        PMFLOW_IMPORT['source'] = 'none'
        print(f"❌ PMFlow BNN not available: {e}")
        print("📝 Proceeding without BNN assist.")


# Colors and window
WIDTH, HEIGHT = 1000, 700
WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
BLUE = (70, 120, 230)
RED = (230, 70, 70)
GREEN = (60, 180, 120)
GRAY = (160, 160, 160)


class BNNFeatureAdapter:
    """Optional feature extractor using pmflow_bnn; trains a small head to map features -> grid drive."""
    def __init__(self, input_len, output_len, device="cpu"):
        if 'torch' not in globals():
            raise RuntimeError("torch not available for BNNFeatureAdapter")
        self.device = torch.device(device)
        cfg = get_performance_config("cpu")
        self.model = get_model_v2(**cfg).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Infer dims
        self.input_dim = None
        self.feat_dim = None
        probe_dims = [28*28, 256, 128, input_len]
        with torch.no_grad():
            for d in probe_dims:
                try:
                    dummy = torch.zeros(1, d, device=self.device)
                    out = self.model(dummy)
                    if isinstance(out, tuple):
                        out = out[0]
                    out = out.view(1, -1)
                    self.input_dim = d
                    self.feat_dim = out.shape[1]
                    break
                except Exception:
                    continue
        if self.input_dim is None or self.feat_dim is None:
            raise RuntimeError("Failed to infer model input/feature dims")

        self.head = torch.nn.Linear(self.feat_dim, output_len).to(self.device)
        self.opt = torch.optim.Adam(self.head.parameters(), lr=2e-3)
        self.loss_fn = torch.nn.MSELoss()

    def _pad(self, x_np):
        x = np.zeros((self.input_dim,), dtype=np.float32)
        n = min(len(x_np), self.input_dim)
        x[:n] = x_np[:n]
        return x

    def forward(self, x_np):
        x = self._pad(x_np)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            feats = self.model(xt)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.view(1, -1)
        y = self.head(feats)
        return y.detach().cpu().view(-1).numpy()

    def step(self, x_np, target_np):
        x = self._pad(x_np)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        t = torch.tensor(target_np, dtype=torch.float32, device=self.device).view(1, -1)
        self.opt.zero_grad()
        feats = self.model(xt)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats.view(1, -1)
        y = self.head(feats)
        loss = self.loss_fn(y, t)
        loss.backward()
        self.opt.step()
        return float(loss.item())


def build_grid_positions(grid_w, grid_h, margin=60):
    xs = np.linspace(margin, WIDTH - margin, grid_w)
    ys = np.linspace(margin, HEIGHT - margin, grid_h)
    coords = [(int(x), int(y)) for y in ys for x in xs]
    return coords


def draw_network(screen, coords, grid_w, grid_h, acts, w_h, w_v, wmax):
    # Draw edges (right and down neighbors)
    for j in range(grid_h):
        for i in range(grid_w):
            idx = j * grid_w + i
            x1, y1 = coords[idx]
            # Right neighbor
            if i < grid_w - 1:
                w = w_h[j, i]
                thickness = 1 + int(4 * min(1.0, abs(w) / wmax))
                color = GREEN if w >= 0 else RED
                x2, y2 = coords[idx + 1]
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), thickness)
            # Down neighbor
            if j < grid_h - 1:
                w = w_v[j, i]
                thickness = 1 + int(4 * min(1.0, abs(w) / wmax))
                color = GREEN if w >= 0 else RED
                x2, y2 = coords[idx + grid_w]
                pygame.draw.line(screen, color, (x1, y1), (x2, y2), thickness)

    # Draw nodes; color by activation
    for idx, (x, y) in enumerate(coords):
        a = float(np.clip(acts[idx], 0.0, 1.0))
        color = (int(50 + 150 * a), int(100 + 120 * a), 255)
        pygame.draw.circle(screen, color, (x, y), 6)


def main():
    parser = argparse.ArgumentParser(description="Live Synapse Explorer (nn_lib_v2 hook)")
    parser.add_argument('--grid-w', type=int, default=18)
    parser.add_argument('--grid-h', type=int, default=12)
    parser.add_argument('--adapter', choices=['auto','bnn','none'], default='auto')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--no-github-install', action='store_true')
    parser.add_argument('--nn-lib-path', type=str, default=None)
    parser.add_argument('--fps', type=int, default=60)
    args = parser.parse_args()

    setup_pmflow_bnn(nn_lib_path_override=args.nn_lib_path, allow_github_install=not args.no_github_install)

    # Device
    device = 'cpu'
    try:
        import torch  # noqa: F401
        if args.device == 'cuda':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cpu'
    except Exception:
        device = 'cpu'

    grid_w, grid_h = args.grid_w, args.grid_h
    N = grid_w * grid_h
    coords = build_grid_positions(grid_w, grid_h)

    # States
    acts = np.zeros((N,), dtype=np.float32)
    ext = np.zeros((N,), dtype=np.float32)  # external input field from mouse
    w_h = np.random.uniform(-0.05, 0.05, size=(grid_h, grid_w - 1)).astype(np.float32)
    w_v = np.random.uniform(-0.05, 0.05, size=(grid_h - 1, grid_w)).astype(np.float32)
    lr = 0.02
    decay_w = 0.001
    wmax = 0.8
    learn = True
    bnn_assist = (args.adapter == 'bnn') or (args.adapter == 'auto' and PMFLOW_IMPORT.get('available', False))

    # Optional BNN feature assist
    feature_adapter = None
    if bnn_assist and PMFLOW_IMPORT.get('available', False):
        try:
            feature_adapter = BNNFeatureAdapter(input_len=N, output_len=N, device=device)
        except Exception as e:
            print(f"⚠️ BNN assist unavailable: {e}")
            feature_adapter = None
            bnn_assist = False

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Live Synapse Explorer (nn_lib_v2)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    paused = False
    running = True
    brush = 1.0
    higher_decay = False

    def draw_text(txt, x, y, color=BLACK):
        screen.blit(font.render(txt, True, color), (x, y))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_l:
                    learn = not learn
                elif event.key == pygame.K_c:
                    acts[:] = 0
                    ext[:] = 0
                elif event.key == pygame.K_w:
                    w_h[:] = np.random.uniform(-0.05, 0.05, size=w_h.shape)
                    w_v[:] = np.random.uniform(-0.05, 0.05, size=w_v.shape)
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    lr *= 1.2
                elif event.key == pygame.K_MINUS:
                    lr /= 1.2
                elif event.key == pygame.K_d:
                    higher_decay = not higher_decay
                elif event.key == pygame.K_b:
                    bnn_assist = not bnn_assist

        if not paused:
            # Mouse painting
            mx, my = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:
                # activate nearest node
                dists = [(mx - x)**2 + (my - y)**2 for (x, y) in coords]
                idx = int(np.argmin(dists))
                ext[idx] = min(1.0, ext[idx] + 0.6 * brush)
                # splash to neighbors
                i = idx % grid_w
                j = idx // grid_w
                for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < grid_w and 0 <= nj < grid_h:
                        nidx = nj*grid_w + ni
                        ext[nidx] = min(1.0, ext[nidx] + 0.3 * brush)

            # Decay external input and activations
            ext *= 0.90
            # Compute synaptic input from neighbors
            syn = np.zeros_like(acts)
            for j in range(grid_h):
                for i in range(grid_w):
                    idx = j*grid_w + i
                    a_i = acts[idx]
                    # right neighbor contributes via w_h[j,i]
                    if i < grid_w - 1:
                        w = w_h[j, i]
                        syn[idx] += w * acts[idx + 1]
                        syn[idx + 1] += w * acts[idx]
                    # down neighbor via w_v[j,i]
                    if j < grid_h - 1:
                        w = w_v[j, i]
                        syn[idx] += w * acts[idx + grid_w]
                        syn[idx + grid_w] += w * acts[idx]

            # Optional BNN feature-driven drive
            if bnn_assist and feature_adapter is not None:
                # Train the feature head to reconstruct user input pattern ext (smoothed)
                target_drive = ext.copy()
                feature_adapter.step(acts.copy(), target_drive)
                drive = feature_adapter.forward(acts.copy())
            else:
                drive = 0.0

            # Update activations (leaky integrator)
            acts = 0.90 * acts + 0.08 * syn + 0.25 * ext + 0.10 * (drive if isinstance(drive, np.ndarray) else 0.0)
            acts = np.clip(acts, 0.0, 1.0)

            # Plasticity (Hebbian)
            if learn:
                dw_scale = lr
                dw_decay = (0.004 if higher_decay else 0.001)
                for j in range(grid_h):
                    for i in range(grid_w - 1):
                        idx = j*grid_w + i
                        jdx = idx + 1
                        hebb = acts[idx] * acts[jdx]
                        w_h[j, i] += dw_scale * hebb - dw_decay * w_h[j, i]
                        w_h[j, i] = float(np.clip(w_h[j, i], -wmax, wmax))
                for j in range(grid_h - 1):
                    for i in range(grid_w):
                        idx = j*grid_w + i
                        jdx = idx + grid_w
                        hebb = acts[idx] * acts[jdx]
                        w_v[j, i] += dw_scale * hebb - dw_decay * w_v[j, i]
                        w_v[j, i] = float(np.clip(w_v[j, i], -wmax, wmax))

        # Draw
        screen.fill(WHITE)
        draw_network(screen, coords, grid_w, grid_h, acts, w_h, w_v, wmax)

        # HUD
        info1 = f"lr: {lr:.4f} | decay: {'high' if higher_decay else 'low'} | learn: {'on' if learn else 'off'}"
        info2 = f"pmflow: {PMFLOW_IMPORT['source']} {PMFLOW_IMPORT['version']} | BNN assist: {'on' if bnn_assist and feature_adapter is not None else 'off'}"
        controls = "Mouse=Paint  [L]=Learn  [C]=Clear  [W]=ResetW  [+/-]=LR  [D]=Decay  [B]=BNN  [Space]=Pause  [Q/Esc]=Quit"
        screen.blit(pygame.font.SysFont("consolas", 18).render(info1, True, BLACK), (10, 10))
        screen.blit(pygame.font.SysFont("consolas", 18).render(info2, True, BLACK), (10, 30))
        screen.blit(pygame.font.SysFont("consolas", 18).render(controls, True, BLACK), (10, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
