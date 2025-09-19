#!/usr/bin/env python3
"""
Foraging Arena (pygame)

- Visual: 2D arena with an agent that collects food and avoids hazards
- Sensors: Egocentric multi-ray vision (relative to agent heading), each ray returns nearest hit type and normalized distance
- Controls: Controller outputs turn delta and forward speed (no absolute coordinates exposed to controller)
- Interaction: Paint/erase food and hazards; background growth (occasional random and clump growths)
- Learning: Online reward-modulated controller (BNN head over pmflow_bnn if available; Linear fallback)

Controls:
  Mouse left drag: Paint food
  Mouse right drag: Paint hazards
  Mouse middle drag or [E]: Erase
  [G]: Toggle growth mode (Occasional / Clump)
  [C]: Clear all
  [Space]: Pause
  [H]: Toggle human drive (arrows) vs AI
  [1]/[2]/[3]: Select paint mode (Food / Hazard / Erase)
  Scroll: Change brush size
  [Q]/Esc: Quit

Flags:
  --rays INT  (number of vision rays)
  --fov DEG   (field of view around heading)
  --adapter {auto,bnn,linear}
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

# pmflow_bnn setup (import-first -> optional github -> local path)
PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

def setup_pmflow_bnn(nn_lib_path_override=None, allow_github_install=True):
    global PMFLOW_IMPORT, get_model_v2, get_performance_config, torch
    PMFLOW_IMPORT = {'available': False, 'source': 'none', 'version': 'Unknown'}

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", ".."))
    default_nn_lib_path = os.path.join(repo_root, "programs", "demos", "machine_learning", "nn_lib_v2")
    nn_lib_path = nn_lib_path_override or default_nn_lib_path

    # Try environment first
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

    # Local fallback
    if install_outcome != 'github':
        if os.path.exists(nn_lib_path) and nn_lib_path not in sys.path:
            sys.path.insert(0, nn_lib_path)
            print(f"📂 Added local library path: {nn_lib_path}")

    # Try import
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
        print("📝 Falling back to Linear controller.")


# World constants
WIDTH, HEIGHT = 1000, 700
WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
GREEN = (60, 180, 120)
RED = (230, 70, 70)
BLUE = (70, 120, 230)
GRAY = (160, 160, 160)
ORANGE = (240, 180, 80)
PURPLE = (150, 70, 200)

# Discrete fields to store food/hazards (grid-based for fast ray march)
CELL = 10
GRID_W = WIDTH // CELL
GRID_H = HEIGHT // CELL

# Per-ray feature layout: [empty, food, hazard, wall, norm_dist]
PER_RAY = 5


def world_to_cell(x, y):
    return int(x // CELL), int(y // CELL)


class Field:
    def __init__(self):
        self.food = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.hazard = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    def clear(self):
        self.food.fill(0.0)
        self.hazard.fill(0.0)

    def paint(self, x, y, brush, mode):
        cx, cy = world_to_cell(x, y)
        r = max(1, int(brush / CELL))
        for j in range(cy - r, cy + r + 1):
            for i in range(cx - r, cx + r + 1):
                if 0 <= i < GRID_W and 0 <= j < GRID_H:
                    if (i - cx) ** 2 + (j - cy) ** 2 <= r * r:
                        if mode == 'food':
                            self.food[j, i] = np.clip(self.food[j, i] + 0.35, 0.0, 1.0)
                        elif mode == 'hazard':
                            self.hazard[j, i] = np.clip(self.hazard[j, i] + 0.35, 0.0, 1.0)
                        elif mode == 'erase':
                            self.food[j, i] = max(0.0, self.food[j, i] - 0.9)
                            self.hazard[j, i] = max(0.0, self.hazard[j, i] - 0.9)

    def grow(self, mode='occasional'):
        if mode == 'occasional':
            # Gentle, sparse random growths
            for _ in range(3):
                i = random.randrange(GRID_W)
                j = random.randrange(GRID_H)
                if random.random() < 0.75:
                    self.food[j, i] = np.clip(self.food[j, i] + 0.12, 0.0, 1.0)
                else:
                    self.hazard[j, i] = np.clip(self.hazard[j, i] + 0.12, 0.0, 1.0)
        elif mode == 'clump':
            # Seed a clump and grow in radius
            i0 = random.randrange(GRID_W)
            j0 = random.randrange(GRID_H)
            rad = random.randint(2, 4)
            pick_food = random.random() < 0.75
            for j in range(j0 - rad, j0 + rad + 1):
                for i in range(i0 - rad, i0 + rad + 1):
                    if 0 <= i < GRID_W and 0 <= j < GRID_H and (i - i0) ** 2 + (j - j0) ** 2 <= rad * rad:
                        if pick_food:
                            self.food[j, i] = np.clip(self.food[j, i] + 0.15, 0.0, 1.0)
                        else:
                            self.hazard[j, i] = np.clip(self.hazard[j, i] + 0.15, 0.0, 1.0)

    def decay(self):
        # Natural decay to prevent overgrowth/flooding
        self.food *= 0.996
        self.hazard *= 0.997
        # Remove tiny residues
        self.food[self.food < 0.02] = 0.0
        self.hazard[self.hazard < 0.02] = 0.0
        # Soft global cap: if mean density too high, apply extra decay
        if float(self.food.mean()) > 0.08:
            self.food *= 0.990
        if float(self.hazard.mean()) > 0.06:
            self.hazard *= 0.990
    def seed_initial(self, food_clumps=2, hazard_clumps=1):
        # Place a few gentle clumps at start so agent can experience food/hazards
        for _ in range(food_clumps):
            self.grow('clump')
        # temporarily bias toward hazards for seeding
        for _ in range(hazard_clumps):
            i0 = random.randrange(GRID_W)
            j0 = random.randrange(GRID_H)
            rad = random.randint(2, 3)
            for j in range(j0 - rad, j0 + rad + 1):
                for i in range(i0 - rad, i0 + rad + 1):
                    if 0 <= i < GRID_W and 0 <= j < GRID_H and (i - i0) ** 2 + (j - j0) ** 2 <= rad * rad:
                        self.hazard[j, i] = np.clip(self.hazard[j, i] + 0.12, 0.0, 1.0)


def raycast(field: Field, origin, angle, max_dist=300):
    """Ray march in grid space. Returns (hit_type, distance).
    hit_type:
      0: empty
      1: food
      2: hazard
      3: wall (arena boundary)
    """
    ox, oy = origin
    dx = math.cos(angle)
    dy = math.sin(angle)
    step = CELL * 0.7
    d = 0.0
    while d < max_dist:
        x = ox + dx * d
        y = oy + dy * d
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return 3, d  # wall (boundary)
        i, j = world_to_cell(x, y)
        if field.hazard[j, i] > 0.25:
            return 2, d
        if field.food[j, i] > 0.25:
            return 1, d
        d += step
    return 0, max_dist


# Controllers (egocentric inputs only)
class LinearAdapter:
    def __init__(self, in_dim, out_dim=2, lr=0.01, weight_decay=1e-3):
        self.w = np.zeros((in_dim,), dtype=np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)  # outputs
        self.Wo = np.zeros((out_dim, in_dim), dtype=np.float32)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        y = self.Wo @ x + self.b
        return y.astype(np.float32)

    def step(self, x, target_y, reward=0.0):
        y = self.forward(x)
        err = y - target_y
        # Prevent sign flip on negative rewards so we don't reinforce bad behavior
        scale = max(0.0, 1.0 + float(reward))
        self.Wo -= self.lr * scale * (err[:, None] @ x[None, :])
        self.b -= self.lr * scale * err
        # L2 regularization to avoid drift
        self.Wo -= self.lr * self.weight_decay * self.Wo
        self.b  -= self.lr * self.weight_decay * self.b
        return float(0.5 * float(np.dot(err, err)))


class BNNAdapter:
    def __init__(self, in_dim, out_dim=2, device='cpu'):
        if 'torch' not in globals():
            raise RuntimeError("torch not available for BNNAdapter")
        self.device = torch.device(device)
        cfg = get_performance_config('cpu')
        self.model = get_model_v2(**cfg).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Infer dims
        self.input_dim = None
        self.feat_dim = None
        with torch.no_grad():
            for d in [28*28, 128, 64, max(16, in_dim), in_dim]:
                try:
                    dummy = torch.zeros(1, d, device=self.device)
                    out = self.model(dummy)
                    if isinstance(out, tuple): out = out[0]
                    out = out.view(1, -1)
                    self.input_dim = d
                    self.feat_dim = out.shape[1]
                    break
                except Exception:
                    continue
        if self.input_dim is None or self.feat_dim is None:
            raise RuntimeError("Failed to infer dims")
        self.head = torch.nn.Linear(self.feat_dim, out_dim).to(self.device)
        self.opt = torch.optim.Adam(self.head.parameters(), lr=2e-3, weight_decay=1e-4)
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
            if isinstance(feats, tuple): feats = feats[0]
            feats = feats.view(1, -1)
        y = self.head(feats)
        return y.detach().cpu().view(-1).numpy()

    def step(self, x_np, target_np, reward=0.0):
        x = self._pad(x_np)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        t = torch.tensor(target_np, dtype=torch.float32, device=self.device).view(1, -1)
        self.opt.zero_grad()
        feats = self.model(xt)
        if isinstance(feats, tuple): feats = feats[0]
        feats = feats.view(1, -1)
        y = self.head(feats)
        loss = self.loss_fn(y, t)
        # Prevent sign flip on negative rewards so we don't reinforce bad behavior
        scale = max(0.0, 1.0 + float(reward))
        (loss * scale).backward()
        self.opt.step()
        return float(loss.item())


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.heading = 0.0  # radians
        self.gaze_heading = 0.0  # radians (optional decoupled look)
        self.speed = 0.0
        self.radius = 10
        self.energy = 0.7  # 0..1 hunger/energy index (1 full, 0 starving)

    def step(self, dtheta, speed):
        # bound and integrate
        self.heading += float(np.clip(dtheta, -0.2, 0.2))
        self.heading = (self.heading + math.tau) % math.tau
        self.speed = float(np.clip(speed, 0.0, 6.0))
        self.x += math.cos(self.heading) * self.speed
        self.y += math.sin(self.heading) * self.speed
        # clamp within arena
        self.x = float(np.clip(self.x, self.radius, WIDTH - self.radius))
        self.y = float(np.clip(self.y, self.radius, HEIGHT - self.radius))

    def pos(self):
        return (self.x, self.y)


def body_front_proximity(field: Field, agent: Agent, fov_deg=60.0, rays=3, max_dist=160):
    """Scan a small fan of rays relative to BODY heading (not gaze) to detect near walls/hazards.
    Returns: (near_bool, bias, min_nd)
      near_bool: whether a wall/hazard is near within threshold
      bias: signed turn suggestion away from threat (>0 turn right, <0 turn left)
      min_nd: minimum normalized distance observed among threat hits
    """
    half = math.radians(fov_deg) / 2.0
    if rays <= 1:
        rel_angles = [0.0]
    else:
        rel_angles = np.linspace(-half, +half, rays)
    center = (rays - 1) / 2.0
    min_nd = 1.0
    bias = 0.0
    found = False
    for idx, rel in enumerate(rel_angles):
        ang = agent.heading + float(rel)
        t, d = raycast(field, agent.pos(), ang, max_dist=max_dist)
        if t in (2, 3):  # hazard or wall
            nd = d / max_dist
            found = True
            if nd < min_nd:
                min_nd = nd
            # push away from the side of the hit, weight by closeness
            rel_index = (idx - center)
            side = np.sign(rel_index)  # left negative, right positive
            # If hit is central (rel_index ~ 0), steer right by default
            if side == 0:
                side = 1.0
            bias += float(side) * (1.0 - nd)
    near = found and (min_nd < 0.25)
    # scale bias into a turn magnitude suggestion
    turn_suggestion = float(np.clip(0.18 * bias, -0.25, 0.25))
    return near, turn_suggestion, min_nd


def build_egocentric_input(field: Field, agent: Agent, rays=9, fov_deg=120, max_dist=250, use_gaze=False, include_rel_look=False):
    # Cast rays centered on heading or gaze, returns vector per ray: [onehot(4: empty, food, hazard, wall), norm_dist]
    # Optionally appends relative look vs body as [cos(delta), sin(delta)] at the end
    half = math.radians(fov_deg) / 2.0
    if rays <= 1:
        angles = [0.0]
    else:
        angles = np.linspace(-half, +half, rays)
    out = []
    for rel in angles:
        base = agent.gaze_heading if use_gaze else agent.heading
        ang = base + rel
        t, d = raycast(field, agent.pos(), ang, max_dist=max_dist)
        onehot = [0.0, 0.0, 0.0, 0.0]
        if t == 0:      # empty
            onehot[0] = 1.0
        elif t == 1:    # food
            onehot[1] = 1.0
        elif t == 2:    # hazard
            onehot[2] = 1.0
        elif t == 3:    # wall
            onehot[3] = 1.0
        nd = d / max_dist
        out.extend(onehot + [nd])
    if include_rel_look:
        delta = ((agent.gaze_heading - agent.heading + math.tau + math.pi) % math.tau) - math.pi
        out.extend([math.cos(delta), math.sin(delta)])
    return np.array(out, dtype=np.float32)


def teacher_policy(ego_x, energy=0.7, prefer_food=True, out_dim=2, decoupled=False):
    # Simple heuristic: steer toward nearby food ray, away from hazard, with speed based on certainty
    rays = len(ego_x) // PER_RAY
    # Extract per-ray
    desirability = 0.0
    turn = 0.0
    best = None
    for i in range(rays):
        base = i * PER_RAY
        is_empty, is_food, is_haz, is_wall, ndist = ego_x[base:base+PER_RAY]
        dist_reward = (1.0 - ndist)
        if is_food > 0.5:
            score = (1.7 if prefer_food else 1.5) * dist_reward
        elif is_haz > 0.5:
            score = -1.2 * dist_reward
        elif is_wall > 0.5:
            # Avoid walls more strongly than empty, less than hazards
            score = -1.0 * dist_reward
        else:
            score = 0.0
        if best is None or score > best[0]:
            best = (score, i)
    if best is not None:
        center = (rays - 1) / 2.0
        rel = (best[1] - center) / center if center > 0 else 0.0
        turn = float(np.clip(rel * 0.10, -0.15, 0.15))
        # Speed scales with energy to encourage seeking when hungry
        speed_base = 0.3 + (1.0 - energy) * 2.0
        speed = float(np.clip(speed_base + max(0.0, best[0]) * 0.8, 0.2, 5.0))
    else:
        # No salient signals: add gentle exploration bias (small random turn) and move based on energy
        turn = float(np.clip(random.uniform(-0.05, 0.05), -0.08, 0.08))
        speed = 0.6 + (1.0 - energy) * 1.2
    if out_dim == 3:
        # Basic default: gaze_turn tracks the same turn as body; caller may add scan
        return np.array([turn, turn, speed], dtype=np.float32)
    else:
        return np.array([turn, speed], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Foraging Arena (nn_lib_v2 hook)")
    parser.add_argument('--rays', type=int, default=9)
    parser.add_argument('--fov', type=float, default=120.0)
    parser.add_argument('--adapter', choices=['auto','bnn','linear'], default='auto')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--no-github-install', action='store_true')
    parser.add_argument('--nn-lib-path', type=str, default=None)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--decouple-look', action='store_true', help='Cast rays using a separate gaze heading and lightly decouple look vs motion')
    parser.add_argument('--outputs', type=int, choices=[2,3], default=2, help='Controller outputs: 2=[body_turn,speed], 3=[gaze_turn,body_turn,speed]')
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

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Foraging Arena (nn_lib_v2)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    field = Field()
    # Initial seeding so the agent can experience food/hazards early
    field.seed_initial(food_clumps=2, hazard_clumps=1)
    agent = Agent(WIDTH * 0.5, HEIGHT * 0.5)
    agent.gaze_heading = agent.heading

    # Controller selection
    rays = max(3, int(args.rays))
    input_dim = rays * PER_RAY
    if args.adapter == 'linear':
        use_bnn = False
    elif args.adapter == 'bnn':
        use_bnn = PMFLOW_IMPORT.get('available', False)
    else:
        use_bnn = PMFLOW_IMPORT.get('available', False)
    out_dim = int(args.outputs)
    controller = (BNNAdapter(input_dim, out_dim=out_dim, device=device)
                  if use_bnn else LinearAdapter(input_dim, out_dim=out_dim, lr=0.02, weight_decay=5e-4))

    # Modes
    paint_mode = 'food'  # 'food'|'hazard'|'erase'
    brush = 18.0
    paused = False
    human = False
    growth_mode = 'occasional'
    grow_timer = 0.0
    # Time delta based on FPS (used for growth timing and metabolism)
    dt = 1.0 / max(1, int(args.fps))
    dtheta_filt = 0.0
    speed_filt = 0.0
    turn_bias = 0.0  # EWMA of turn to detect directional bias
    score = 0
    last_reward = 0.0
    rolling_loss = []

    def draw_text(txt, x, y, color=BLACK):
        screen.blit(font.render(txt, True, color), (x, y))

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
                elif event.key == pygame.K_c:
                    field.clear()
                elif event.key == pygame.K_h:
                    human = not human
                elif event.key == pygame.K_g:
                    growth_mode = 'clump' if growth_mode == 'occasional' else 'occasional'
                elif event.key == pygame.K_1:
                    paint_mode = 'food'
                elif event.key == pygame.K_2:
                    paint_mode = 'hazard'
                elif event.key == pygame.K_3 or event.key == pygame.K_e:
                    paint_mode = 'erase'
            elif event.type == pygame.MOUSEWHEEL:
                brush = float(np.clip(brush + event.y * 3.0, 5.0, 60.0))

        if not paused:
            # Painting
            mx, my = pygame.mouse.get_pos()
            buttons = pygame.mouse.get_pressed()
            # Fixed mapping: LMB=food, RMB=hazard, MMB=erase
            if buttons[0]:
                field.paint(mx, my, brush, 'food')
            if buttons[2]:
                field.paint(mx, my, brush, 'hazard')
            if buttons[1]:
                field.paint(mx, my, brush, 'erase')

            # Growth + decay
            grow_timer += dt
            if grow_timer >= 2.5:  # slower periodic growth
                field.grow(growth_mode)
                grow_timer = 0.0
            field.decay()

            # Egocentric sensing (optionally using gaze) with relative look vector when enabled
            ego = build_egocentric_input(field, agent, rays=rays, fov_deg=float(args.fov), use_gaze=bool(args.decouple_look), include_rel_look=bool(args.decouple_look))

            # Reward events: consume food within agent footprint, penalize hazard contact area
            reward = 0.0
            cx, cy = world_to_cell(agent.x, agent.y)
            r_cells = max(1, int((agent.radius + 3) / CELL))
            consumed = 0.0
            hazard_contact = 0.0
            for jj in range(cy - r_cells, cy + r_cells + 1):
                for ii in range(cx - r_cells, cx + r_cells + 1):
                    if 0 <= ii < GRID_W and 0 <= jj < GRID_H:
                        # circle mask matches agent
                        dx = (ii + 0.5) * CELL - agent.x
                        dy = (jj + 0.5) * CELL - agent.y
                        if dx*dx + dy*dy <= (agent.radius + 2)**2:
                            if field.food[jj, ii] > 0.05:
                                take = min(0.18, field.food[jj, ii])
                                field.food[jj, ii] -= take
                                consumed += take
                            if field.hazard[jj, ii] > 0.2:
                                hit = min(0.12, field.hazard[jj, ii])
                                field.hazard[jj, ii] -= hit * 0.2
                                hazard_contact += hit
            if consumed > 0.0:
                reward += +0.8 * min(1.0, consumed)
                # Replenish energy proportionally to consumed amount (slightly stronger gain)
                agent.energy = float(np.clip(agent.energy + 0.50 * consumed, 0.0, 1.0))
                score += int(4 * consumed)
            if hazard_contact > 0.0:
                reward += -0.8 * min(1.0, hazard_contact)
                agent.energy = float(np.clip(agent.energy - 0.15 * hazard_contact, 0.0, 1.0))

            # Small movement penalty to encourage efficient foraging
            reward += -0.003

            # Human vs AI control
            if human:
                keys = pygame.key.get_pressed()
                dtheta = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.15
                speed = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 4.0
                target = np.array([dtheta, speed], dtype=np.float32)
                # Apply smoothing
                dtheta_filt = 0.7 * dtheta_filt + 0.3 * dtheta
                speed_filt = 0.7 * speed_filt + 0.3 * speed
                agent.step(dtheta_filt, speed_filt)
                if args.decouple_look:
                    agent.gaze_heading = (0.9 * agent.gaze_heading + 0.1 * agent.heading) % math.tau
            else:
                # Teacher target to shape learning, plus online update
                target = teacher_policy(ego, energy=agent.energy, prefer_food=True, out_dim=out_dim, decoupled=bool(args.decouple_look))
                # Penalize spinning to reduce oscillation
                reward -= 0.02 * abs(float(target[0]))
                # Minor penalty for wall contact (agent hugging arena edge) BEFORE learning step
                if (agent.x <= agent.radius + 0.5 or agent.x >= WIDTH - agent.radius - 0.5 or
                    agent.y <= agent.radius + 0.5 or agent.y >= HEIGHT - agent.radius - 0.5):
                    if abs(speed_filt) > 0.2:
                        reward -= 0.04
                # Shaping: small pull toward central food, small push from central hazards/walls
                # Use only the ray portion even if extra features are present
                blocks = ego[:rays*PER_RAY].reshape(-1, PER_RAY)
                center = (rays - 1) / 2.0
                score_center = 0.0
                food_sal = 0.0
                hazwall_sal = 0.0
                for i in range(rays):
                    w = 1.0 - abs(i - center) / max(1.0, center)
                    is_food = blocks[i, 1] > 0.5
                    is_haz = blocks[i, 2] > 0.5
                    is_wall = blocks[i, 3] > 0.5
                    nd = blocks[i, 4]
                    if is_food:
                        contrib = w * (1.0 - nd)
                        score_center += contrib * 0.02
                        food_sal += contrib
                    if is_haz or is_wall:
                        contrib = w * (1.0 - nd)
                        score_center -= contrib * 0.02
                        hazwall_sal += contrib
                reward += score_center
                # Determine nearest food normalized distance (0=close,1=far) for speed modulation
                nearest_food_nd = None
                for i in range(rays):
                    if blocks[i, 1] > 0.5:
                        nd = blocks[i, 4]
                        nearest_food_nd = nd if nearest_food_nd is None else min(nearest_food_nd, nd)
                # Symmetry augmentation: randomly mirror left-right rays and flip turn sign for training only
                if random.random() < 0.5:
                    ego_train = ego.copy()
                    # Reverse rays in blocks of PER_RAY values (onehot+dist)
                    blocks = ego_train[:rays*PER_RAY].reshape(-1, PER_RAY)
                    ego_train = np.flip(blocks, axis=0).reshape(-1)
                    if out_dim == 3:
                        target_train = np.array([-target[0], -target[1], target[2]], dtype=np.float32)
                    else:
                        target_train = np.array([-target[0], target[1]], dtype=np.float32)
                else:
                    ego_train = ego
                    target_train = target
                loss = controller.step(ego_train, target_train, reward=reward)
                rolling_loss.append(loss)
                if len(rolling_loss) > 300: rolling_loss.pop(0)
                # Use only the ray portion for controller input for stability
                act = controller.forward(ego[:rays*PER_RAY])
                if out_dim == 3:
                    dtheta_gaze_cmd = float(np.clip(act[0], -0.2, 0.2))
                    dtheta_cmd = float(np.clip(act[1], -0.2, 0.2))
                    speed_cmd = float(np.clip(act[2], 0.0, 6.0))
                else:
                    dtheta_cmd = float(np.clip(act[0], -0.2, 0.2))
                    speed_cmd = float(np.clip(act[1], 0.0, 6.0))
                # If decoupled, only align body toward gaze when food is salient vs hazards/walls; or use explicit gaze command in 3-output mode
                if args.decouple_look:
                    delta = ((agent.gaze_heading - agent.heading + math.tau + math.pi) % math.tau) - math.pi
                    if out_dim == 3:
                        # Apply the gaze turn directly, with small scan bias when no strong signal
                        tnow = time.time()
                        scan = 0.06 * math.sin(0.7 * tnow)
                        dtheta_gaze_cmd += (scan if abs(target[0]) < 0.02 else 0.0)
                        agent.gaze_heading = (agent.gaze_heading + 0.75 * dtheta_gaze_cmd) % math.tau
                    elif food_sal > max(0.02, hazwall_sal * 1.25):
                        dtheta_cmd += float(np.clip(0.08 * delta, -0.15, 0.15))
                    # In both 2- and 3-output modes, add a small assist to body turn toward gaze when it's safe
                    if food_sal > max(0.03, hazwall_sal * 1.10):
                        assist = 0.12 * math.sin(delta)
                        dtheta_cmd += float(np.clip(assist, -0.15, 0.15))
                # Approach-aware deceleration: slow target speed when close to food to avoid overshoot
                if nearest_food_nd is not None:
                    target_speed = 1.0 + 5.0 * max(0.0, nearest_food_nd - 0.15)
                    speed_cmd = min(speed_cmd, target_speed)
                # Low-pass filter commands to reduce jitter/spin
                dtheta_filt = 0.75 * dtheta_filt + 0.25 * dtheta_cmd
                # Acceleration limiting for speed to reduce oscillations and overshoot
                desired_speed = 0.75 * speed_filt + 0.25 * speed_cmd
                max_accel = 0.20
                if desired_speed > speed_filt:
                    speed_filt = min(desired_speed, speed_filt + max_accel)
                else:
                    speed_filt = max(desired_speed, speed_filt - max_accel)
                # Reflex: if front rays see a very near wall, add a quick turn away
                blocks = ego[:rays*PER_RAY].reshape(-1, PER_RAY)
                front_left = int(math.floor((rays - 1) / 2.0))
                front_right = int(math.ceil((rays - 1) / 2.0))
                near_front_wall = False
                wall_bias = 0.0
                for i in range(front_left, front_right + 1):
                    if 0 <= i < rays and blocks[i, 3] > 0.5 and blocks[i, 4] < 0.18:
                        near_front_wall = True
                        rel = (i - (rays - 1) / 2.0)
                        wall_bias += -np.sign(rel) * (0.15 * (1.0 - blocks[i, 4]))
                if near_front_wall:
                    dtheta_filt = float(np.clip(dtheta_filt + np.clip(wall_bias, -0.2, 0.2), -0.2, 0.2))
                # Wall-aware speed clamp: if central rays see near walls, slow down
                near_wall = False
                for i in range(rays):
                    if abs(i - center) <= max(1, rays // 4):
                        is_wall = blocks[i, 3] > 0.5
                        nd = blocks[i, 4]
                        if is_wall and nd < 0.25:
                            near_wall = True
                            break
                if near_wall:
                    speed_filt *= 0.6
                # Strong body-centric reflex: if body-facing fan sees near wall/hazard, turn away and clamp speed
                near_threat, turn_suggest, min_nd = body_front_proximity(field, agent, fov_deg=70.0, rays=5, max_dist=180)
                if near_threat:
                    dtheta_filt = float(np.clip(dtheta_filt + turn_suggest, -0.2, 0.2))
                    speed_filt = min(speed_filt, 1.0 + 4.0 * max(0.0, min_nd - 0.1))
                # Update gaze to follow target turn gently (scan when no strong signal)
                if args.decouple_look:
                    tnow = time.time()
                    scan = 0.06 * math.sin(0.7 * tnow)
                    gaze_delta = 0.6 * float(target[0]) + (scan if abs(target[0]) < 0.02 else 0.0)
                    agent.gaze_heading = (agent.gaze_heading + gaze_delta) % math.tau
                agent.step(dtheta_filt, speed_filt)
                # Turn friction and bias penalty to avoid persistent CCW/CW drift (reduced)
                dtheta_filt *= 0.98
                turn_bias = 0.98 * turn_bias + 0.02 * dtheta_filt
                if hazwall_sal < 0.05:
                    reward -= 0.005 * abs(turn_bias)

            # Approach shaping: reward moving closer to nearest food (if any)
            if 'nearest_food_nd' in locals() and nearest_food_nd is not None:
                if prev_food_dist is not None:
                    dprog = (prev_food_dist - nearest_food_nd)
                    reward += 0.05 * dprog
                prev_food_dist = nearest_food_nd
            else:
                prev_food_dist = None
            
            # Metabolic drain after moving (scaled by dt) — slowed to make satiety last longer
            drain = (0.02 * dt) + (0.02 * dt * abs(speed_filt) / 6.0) + (0.02 * dt * min(1.0, abs(dtheta_filt) / 0.2))
            agent.energy = float(np.clip(agent.energy - drain, 0.0, 1.0))
            # Hunger shaping (penalize low energy)
            reward += -0.01 * (1.0 - agent.energy)
            last_reward = reward

        # Draw
        screen.fill(WHITE)

        # Draw fields
        # Food
        food_coords = np.argwhere(field.food > 0.05)
        for j, i in food_coords:
            alpha = float(np.clip(field.food[j, i], 0.0, 1.0))
            x = i * CELL + CELL//2
            y = j * CELL + CELL//2
            r = 3 + int(3 * alpha)
            pygame.draw.circle(screen, GREEN, (x, y), r)
        # Hazards
        haz_coords = np.argwhere(field.hazard > 0.05)
        for j, i in haz_coords:
            alpha = float(np.clip(field.hazard[j, i], 0.0, 1.0))
            x = i * CELL + CELL//2
            y = j * CELL + CELL//2
            r = 3 + int(3 * alpha)
            pygame.draw.circle(screen, RED, (x, y), r)

        # Draw agent (heading indicator)
        pygame.draw.circle(screen, BLUE, (int(agent.x), int(agent.y)), agent.radius)
        hx = int(agent.x + math.cos(agent.heading) * 18)
        hy = int(agent.y + math.sin(agent.heading) * 18)
        pygame.draw.line(screen, ORANGE, (int(agent.x), int(agent.y)), (hx, hy), 3)
        # Draw gaze if decoupled
        if args.decouple_look:
            gx = int(agent.x + math.cos(agent.gaze_heading) * 24)
            gy = int(agent.y + math.sin(agent.gaze_heading) * 24)
            pygame.draw.line(screen, PURPLE, (int(agent.x), int(agent.y)), (gx, gy), 2)

        # HUD
        draw_text(f"pmflow: {PMFLOW_IMPORT['source']} {PMFLOW_IMPORT['version']} | adapter: {'BNN' if use_bnn else 'Linear'} | device: {device}", 10, 10)
        draw_text(f"score: {score} | hunger: {1.0 - agent.energy:.2f} | satiety: {agent.energy:.2f} | speed: {speed_filt:.2f} | reward: {last_reward:+.2f} | loss: {np.mean(rolling_loss[-60:]) if rolling_loss else 0:.3f} | turnBias: {turn_bias:+.3f}", 10, 30)
        if args.decouple_look:
            delta = ((agent.gaze_heading - agent.heading + math.tau + math.pi) % math.tau) - math.pi
            draw_text(f"paint: {paint_mode} | brush: {int(brush)} | growth: {growth_mode} | rays: {rays} FOV: {int(args.fov)} | decouple-look: {args.decouple_look} | rel-look: cos={math.cos(delta):+.2f} sin={math.sin(delta):+.2f}", 10, 50)
        else:
            draw_text(f"paint: {paint_mode} | brush: {int(brush)} | growth: {growth_mode} | rays: {rays} FOV: {int(args.fov)} | decouple-look: {args.decouple_look}", 10, 50)
        draw_text("LMB=Food  RMB=Hazard  MMB/E=Erase  [1/2/3]=Modes  [G]=Growth  Scroll=Brush  [H]=Human  [C]=Clear  [Space]=Pause  [Q]=Quit", 10, HEIGHT-30)

        pygame.display.flip()
        clock.tick(int(args.fps))

    pygame.quit()


if __name__ == '__main__':
    main()
