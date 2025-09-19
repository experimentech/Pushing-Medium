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

# Try to import nn_lib_v2 locally
# Ensures we use the exact library from this repo without pip
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", ".."))
# repo_root points to .../gravity
nn_lib_path = os.path.join(repo_root, "programs", "demos", "machine_learning", "nn_lib_v2")
if os.path.exists(nn_lib_path) and nn_lib_path not in sys.path:
    sys.path.insert(0, nn_lib_path)

# Optional BNN adapter
BNN_AVAILABLE = False
try:
    import torch
    from pmflow_bnn import get_model_v2, get_performance_config
    BNN_AVAILABLE = True
except Exception:
    BNN_AVAILABLE = False

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
        self.device = torch.device(device)
        cfg = get_performance_config("cpu")
        self.model = get_model_v2(**cfg).to(self.device)
        # A tiny linear head to map model latent to (dx, dy)
        # If model outputs logits, we just take as features
        self.head = torch.nn.Linear(10, 2).to(self.device)
        self.opt = torch.optim.Adam(list(self.model.parameters()) + list(self.head.parameters()), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            feats = self.model(x)  # expect shape [1,10]
            if isinstance(feats, tuple):
                feats = feats[0]
        y = self.head(feats)
        return y.detach().cpu().view(-1).numpy()

    def step(self, x_np, target_np):
        self.opt.zero_grad()
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device).view(1, -1)
        t = torch.tensor(target_np, dtype=torch.float32, device=self.device).view(1, -1)
        feats = self.model(x)
        if isinstance(feats, tuple):
            feats = feats[0]
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
    def __init__(self, use_bnn=False):
        self.pos = np.array([WIDTH*0.5, HEIGHT*0.5], dtype=np.float32)
        self.adapter = BNNAdapter() if use_bnn and BNN_AVAILABLE else LinearAdapter(lr=0.15)
        self.use_bnn = use_bnn and BNN_AVAILABLE

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
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sensory Remapping Demo (nn_lib_v2 hook)")
    clock = pygame.time.Clock()

    paused = False
    use_bnn = BNN_AVAILABLE  # auto-enable if library present
    agent = Agent(use_bnn=use_bnn)
    world = World(remap_idx=0, auto_switch_s=8.0)

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
        draw_text(screen, "Keys: [Space]=Pause  [M]=Cycle Map  [R]=Force Remap  [Q/Esc]=Quit", 10, 570)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
