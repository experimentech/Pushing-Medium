#!/usr/bin/env python3
"""
Sensory Remapping Demo (pygame)

- Visual: 2D environment with a target stimulus moving along a path
- Interaction: Mapping between sensors (perceived target) and motors (agent) changes mid-run
- BNN hook: Tries to import pmflow_bnn (nn_lib_v2) as an adaptive controller; falls back to linear online adapter
- Why: Demonstrates resilience/adaptation akin to sensory-motor remapping

Controls:
  [Space] Toggle pause
  [M] Cycle remap mode (Identity -> Swap -> Invert -> Rotate)
  [R] Force remap now
  [Q/Esc] Quit

Requirements: pygame, numpy, torch (optional for BNN)
"""
import math
import sys
import time
import pygame
import numpy as np

import os
import subprocess
import argparse

#
# pmflow_bnn import/install setup (mirrors notebook flow)
# Tries GitHub pip install (subdirectory) then falls back to local path
#
PMFLOW_IMPORT = {
    'available': False,
    'source': 'none',
    'version': 'Unknown',
}

def setup_pmflow_bnn(nn_lib_path_override=None, allow_github_install=True):
    global PMFLOW_IMPORT, get_model_v2, get_performance_config, torch
    PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

    # Optional local override path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", ".."))
    default_nn_lib_path = os.path.join(repo_root, "programs", "demos", "machine_learning", "nn_lib_v2")
    nn_lib_path = nn_lib_path_override or default_nn_lib_path

    # Step 0: Try importing from the current environment first
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

    # Step 1: Try GitHub pip install (subdirectory)
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

    # Step 2: Local development fallback path (prepend if available)
    if install_outcome != 'github':
        if os.path.exists(nn_lib_path) and nn_lib_path not in sys.path:
            sys.path.insert(0, nn_lib_path)
            print(f"📂 Added local library path: {nn_lib_path}")

    # Step 3: Try to import again
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
        print("📝 Falling back to Linear adapter. You can specify --adapter linear to silence this message.")

WIDTH, HEIGHT = 800, 600
WHITE = (240, 240, 240)
BLACK = (10, 10, 10)
RED = (230, 70, 70)
BLUE = (70, 120, 230)
GREEN = (60, 180, 120)
YELLOW = (240, 200, 60)
GRAY = (150, 150, 150)

class LinearAdapter:
    """Simple online linear adapter: y = W x + b, trained with SGD to minimize position error."""
    def __init__(self, in_dim=2, out_dim=2, lr=0.2):
        self.W = np.eye(out_dim, in_dim).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.lr = lr

    def forward(self, x):
        return self.W @ x + self.b

    def step(self, x, target):
        # gradient of 0.5*||y - target||^2 w.r.t W,b
        y = self.forward(x)
        err = (y - target)
        # dL/dW = err[:,None] @ x[None,:]
        self.W -= self.lr * (err[:, None] @ x[None, :])
        self.b -= self.lr * err
        return float(0.5 * np.dot(err, err))

class BNNAdapter:
    """Optional BNN-based adapter using nn_lib_v2 pmflow_bnn to learn mapping online.
    Fallback to simple linear head on top of frozen BNN if needed.
    """
    def __init__(self, device="cpu"):
        if 'torch' not in globals():
            raise RuntimeError("torch not available for BNNAdapter")
        self.device = torch.device(device)
        # Choose conservative config; model will be used as frozen feature extractor
        cfg = get_performance_config("cpu")
        self.model = get_model_v2(**cfg).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Discover acceptable input dimension and feature dimension
        self.input_dim = None
        self.feat_dim = None
        probe_dims = [28*28, 128, 64, 16, 2]
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
            # Fallback to identity linear adapter behavior
            raise RuntimeError("Failed to infer model input/feature dims for BNNAdapter")

        self.head = torch.nn.Linear(self.feat_dim, 2).to(self.device)
        self.opt = torch.optim.Adam(self.head.parameters(), lr=5e-3)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x_np):
        # Zero-pad 2D sensed input to model's expected input length
        x = np.zeros((self.input_dim,), dtype=np.float32)
        x[:min(2, self.input_dim)] = x_np[:min(2, self.input_dim)]
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            feats = self.model(xt)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.view(1, -1)
        y = self.head(feats)
        return y.detach().cpu().view(-1).numpy()

    def step(self, x_np, target_np):
        # Build padded input
        x = np.zeros((self.input_dim,), dtype=np.float32)
        x[:min(2, self.input_dim)] = x_np[:min(2, self.input_dim)]
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

class Remap:
    @staticmethod
    def identity():
        return np.eye(2, dtype=np.float32)
    @staticmethod
    def swap():
        return np.array([[0,1],[1,0]], dtype=np.float32)
    @staticmethod
    def invert():
        return np.array([[-1,0],[0,-1]], dtype=np.float32)
    @staticmethod
    def rotate(theta_deg=90):
        t = math.radians(theta_deg)
        return np.array([[math.cos(t), -math.sin(t)],[math.sin(t), math.cos(t)]], dtype=np.float32)

REMAPPINGS = [
    ("Identity", Remap.identity()),
    ("Swap", Remap.swap()),
    ("Invert", Remap.invert()),
    ("Rotate90", Remap.rotate(90)),
]

class Agent:
    def __init__(self, use_bnn=False, device="cpu"):
        self.pos = np.array([WIDTH*0.5, HEIGHT*0.5], dtype=np.float32)
        # Choose adapter
        if use_bnn and PMFLOW_IMPORT.get('available', False):
            try:
                self.adapter = BNNAdapter(device=device)
                self.use_bnn = True
            except Exception as e:
                print(f"⚠️ Falling back to Linear adapter (BNN init failed): {e}")
                self.adapter = LinearAdapter(lr=0.15)
                self.use_bnn = False
        else:
            self.adapter = LinearAdapter(lr=0.15)
            self.use_bnn = False

    def update(self, sensed, target_pos):
        # sensed is the remapped observation of target relative position
        # adapter learns to output motor command to reduce error
        # Form target motor command as delta towards true target
        delta = (target_pos - self.pos) * 0.05  # desired motion
        delta = np.clip(delta, -5.0, 5.0)
        loss = self.adapter.step(sensed, delta)
        cmd = self.adapter.forward(sensed)
        self.pos += np.clip(cmd, -5.0, 5.0)
        self.pos = np.clip(self.pos, [10,10], [WIDTH-10, HEIGHT-10])
        return loss

class World:
    def __init__(self, remap_idx=0, auto_switch_s=8.0):
        self.remap_idx = remap_idx
        self.remap_name, self.M = REMAPPINGS[self.remap_idx]
        self.auto_switch_s = auto_switch_s
        self.last_switch_t = time.time()
        self.t = 0.0
        self.target_pos = np.array([WIDTH*0.2, HEIGHT*0.3], dtype=np.float32)
        self.target_vel = np.array([2.0, 1.5], dtype=np.float32)

    def step_target(self):
        # Move bouncing target
        self.target_pos += self.target_vel
        if self.target_pos[0] < 10 or self.target_pos[0] > WIDTH-10:
            self.target_vel[0] *= -1
        if self.target_pos[1] < 10 or self.target_pos[1] > HEIGHT-10:
            self.target_vel[1] *= -1

    def remap_sensor(self, agent_pos):
        rel = self.target_pos - agent_pos
        return (self.M @ rel).astype(np.float32)

    def maybe_switch(self):
        if time.time() - self.last_switch_t >= self.auto_switch_s:
            self.remap_idx = (self.remap_idx + 1) % len(REMAPPINGS)
            self.remap_name, self.M = REMAPPINGS[self.remap_idx]
            self.last_switch_t = time.time()


def draw_text(surf, text, x, y, color=BLACK, size=18):
    font = pygame.font.SysFont("consolas", size)
    surf.blit(font.render(text, True, color), (x, y))


def main():
    # CLI
    parser = argparse.ArgumentParser(description="Sensory Remapping Demo (nn_lib_v2 hook)")
    parser.add_argument('--adapter', choices=['auto','bnn','linear'], default='auto', help='Adapter selection')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto', help='Computation device')
    parser.add_argument('--nn-lib-path', type=str, default=None, help='Override local nn_lib_v2 path')
    parser.add_argument('--no-github-install', action='store_true', help='Do not attempt GitHub pip install')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS')
    parser.add_argument('--auto-switch', type=float, default=8.0, help='Seconds between auto remap switches')
    args = parser.parse_args()

    # Setup pmflow_bnn import flow
    setup_pmflow_bnn(nn_lib_path_override=args.nn_lib_path, allow_github_install=not args.no_github_install)

    # Device selection
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

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sensory Remapping Demo (nn_lib_v2 hook)")
    clock = pygame.time.Clock()

    paused = False
    # Adapter selection
    if args.adapter == 'bnn':
        use_bnn = PMFLOW_IMPORT.get('available', False)
    elif args.adapter == 'linear':
        use_bnn = False
    else:
        use_bnn = PMFLOW_IMPORT.get('available', False)

    agent = Agent(use_bnn=use_bnn, device=device)
    world = World(remap_idx=0, auto_switch_s=float(args.auto_switch))

    losses = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    world.remap_idx = (world.remap_idx + 1) % len(REMAPPINGS)
                    world.remap_name, world.M = REMAPPINGS[world.remap_idx]
                    world.last_switch_t = time.time()
                elif event.key == pygame.K_r:
                    world.last_switch_t = 0  # force immediate switch

        if not paused:
            world.maybe_switch()
            world.step_target()
            sensed = world.remap_sensor(agent.pos)
            loss = agent.update(sensed, world.target_pos)
            losses.append(loss)
            if len(losses) > 300:
                losses.pop(0)

        # draw
        screen.fill(WHITE)
        # target
        pygame.draw.circle(screen, RED, world.target_pos.astype(int), 8)
        # agent
        pygame.draw.circle(screen, BLUE, agent.pos.astype(int), 10)
        # sensed vector (in agent space) draw arrow relative to agent
        sensed_draw = (agent.pos + np.clip(sensed, -80, 80)).astype(int)
        pygame.draw.line(screen, GRAY, agent.pos.astype(int), sensed_draw, 2)
        pygame.draw.circle(screen, GRAY, sensed_draw, 3)

        # HUD
        draw_text(screen, f"Remap: {world.remap_name}", 10, 10)
        draw_text(screen, f"Adapter: {'BNN (nn_lib_v2)' if agent.use_bnn else 'Linear'}", 10, 30)
        draw_text(screen, f"Loss: {np.mean(losses[-50:]):.3f}", 10, 50)
        info = f"pmflow: {PMFLOW_IMPORT['source']} {PMFLOW_IMPORT['version']} | device: {device} | FPS target: {args.fps}"
        draw_text(screen, info, 10, 70)
        draw_text(screen, "Keys: [Space]=Pause  [M]=Cycle Map  [R]=Force Remap  [Q/Esc]=Quit", 10, 570)

        pygame.display.flip()
        clock.tick(int(args.fps))

    pygame.quit()

if __name__ == "__main__":
    main()
