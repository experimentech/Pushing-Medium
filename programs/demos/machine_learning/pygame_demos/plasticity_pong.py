#!/usr/bin/env python3
"""
Plasticity Pong (pygame)

- Visual: Classic Pong, left paddle is controlled by an adaptive controller (BNN via nn_lib_v2 if available, else Linear)
- Interaction: Ball speed/spin/spawn vary unpredictably; hits give positive reward, misses give negative reward; optional user reinforcement
- Learning: Online, reward-modulated update so paddle improves adaptively without retraining cycles

Controls:
  [Space] Pause
  [H] Toggle human control for left paddle (arrow keys) vs AI
  [J/K] Manual reward: J = negative (-1), K = positive (+1)
  [R] Respawn ball (random speed/spin/spawn)
  [Q/Esc] Quit

Flags:
  --adapter {auto,bnn,linear}
  --device {auto,cpu,cuda}
  --no-github-install
  --nn-lib-path PATH
  --fps N

Why: Everyone knows Pong — the novelty is that the paddle learns in real time with non-stationary ball dynamics.
"""
import os
import sys
import time
import math
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
        print("📝 Falling back to Linear adapter.")


# Colors and layout
WIDTH, HEIGHT = 900, 600
WHITE = (240, 240, 240)
BLACK = (10, 10, 10)
RED = (230, 70, 70)
BLUE = (70, 120, 230)
GREEN = (60, 180, 120)
GRAY = (150, 150, 150)
ORANGE = (245, 170, 70)

PADDLE_W, PADDLE_H = 12, 100
BALL_SIZE = 12
LEFT_X = 40
RIGHT_X = WIDTH - 40 - PADDLE_W


# Controllers
class LinearAdapter:
    """Online linear controller: y = w^T x + b (scalar velocity); SGD with reward-modulated loss."""
    def __init__(self, in_dim, lr=0.01):
        self.w = np.zeros((in_dim,), dtype=np.float32)
        self.b = 0.0
        self.lr = lr

    def forward(self, x):
        return float(np.dot(self.w, x) + self.b)

    def step(self, x, target_v, reward=0.0):
        # MSE scaled by (1 + reward); reward>0 emphasizes updates, reward<0 flips gradient emphasis
        y = self.forward(x)
        err = (y - target_v)
        scale = 1.0 + float(reward)
        self.w -= self.lr * scale * err * x
        self.b -= self.lr * scale * err
        return float(0.5 * err * err)


class BNNAdapter:
    """BNN-backed controller: model as frozen feature extractor, small linear head outputs paddle velocity."""
    def __init__(self, state_dim, device="cpu"):
        if 'torch' not in globals():
            raise RuntimeError("torch not available for BNNAdapter")
        self.device = torch.device(device)
        cfg = get_performance_config("cpu")
        self.model = get_model_v2(**cfg).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Discover dims
        self.input_dim = None
        self.feat_dim = None
        probe_dims = [28*28, 128, 64, 16, max(8, state_dim), state_dim]
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

        self.head = torch.nn.Linear(self.feat_dim, 1).to(self.device)
        self.opt = torch.optim.Adam(self.head.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

    def _pad_state(self, state_np):
        x = np.zeros((self.input_dim,), dtype=np.float32)
        n = min(len(state_np), self.input_dim)
        x[:n] = state_np[:n]
        return x

    def forward(self, state_np):
        x = self._pad_state(state_np)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            feats = self.model(xt)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.view(1, -1)
        y = self.head(feats)
        return float(y.detach().cpu().view(-1).item())

    def step(self, state_np, target_v, reward=0.0):
        x = self._pad_state(state_np)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        t = torch.tensor([target_v], dtype=torch.float32, device=self.device).view(1, -1)
        self.opt.zero_grad()
        feats = self.model(xt)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats.view(1, -1)
        y = self.head(feats)
        loss = self.loss_fn(y, t)
        # Reward modulation: scale loss by (1+reward) (clip to non-negative)
        scale = max(0.0, 1.0 + float(reward))
        (loss * scale).backward()
        self.opt.step()
        return float(loss.item())


# Game objects
class Ball:
    def __init__(self):
        self.reset(random_dir=True)

    def reset(self, random_dir=True):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        # Random velocity magnitude and spin
        speed = random.uniform(6.0, 10.0)
        angle = random.uniform(-0.8, 0.8) if random_dir else 0.0
        self.vx = speed * (1 if random.random() < 0.5 else -1)
        self.vy = speed * math.sin(angle)
        self.spin = random.uniform(-0.3, 0.3)

    def spawn_random(self, to_left=True):
        # Spawn from a random x side and y position
        self.x = RIGHT_X - 30 if not to_left else LEFT_X + 30
        self.y = random.randint(40, HEIGHT - 40)
        speed = random.uniform(6.0, 10.0)
        angle = random.uniform(-0.9, 0.9)
        self.vx = speed * (-1 if to_left else 1)
        self.vy = speed * math.sin(angle)
        self.spin = random.uniform(-0.4, 0.4)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        # Spin slowly alters vy
        self.vy += self.spin * 0.2
        # Bounce vertically
        if self.y < BALL_SIZE//2 or self.y > HEIGHT - BALL_SIZE//2:
            self.vy *= -1
            self.y = max(BALL_SIZE//2, min(HEIGHT - BALL_SIZE//2, self.y))

    def rect(self):
        return pygame.Rect(int(self.x - BALL_SIZE//2), int(self.y - BALL_SIZE//2), BALL_SIZE, BALL_SIZE)


class Paddle:
    def __init__(self, x, is_left=True):
        self.x = x
        self.y = HEIGHT // 2 - PADDLE_H // 2
        self.is_left = is_left
        self.speed = 9.0

    def move(self, dy):
        self.y += dy
        self.y = int(max(10, min(HEIGHT - 10 - PADDLE_H, self.y)))

    def rect(self):
        return pygame.Rect(self.x, int(self.y), PADDLE_W, PADDLE_H)


def build_state(ball: Ball, paddle: Paddle, right_paddle: Paddle):
    # State vector for controller (left paddle): relative positions and velocities
    dy = (ball.y - (paddle.y + PADDLE_H / 2)) / HEIGHT
    dx = (ball.x - paddle.x) / WIDTH
    vyn = ball.vy / 12.0
    vxn = ball.vx / 12.0
    spin = np.clip(ball.spin, -1.0, 1.0)
    right_dy = (ball.y - (right_paddle.y + PADDLE_H / 2)) / HEIGHT
    return np.array([dy, dx, vyn, vxn, spin, right_dy], dtype=np.float32)


def target_velocity(ball: Ball, paddle: Paddle):
    # Move towards intercept point, with smoothing
    center = paddle.y + PADDLE_H / 2
    err = (ball.y - center)
    v = np.clip(err * 0.25, -10.0, 10.0)
    return float(v)


def main():
    parser = argparse.ArgumentParser(description="Plasticity Pong (nn_lib_v2 hook)")
    parser.add_argument('--adapter', choices=['auto','bnn','linear'], default='auto')
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

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Plasticity Pong (nn_lib_v2)")
    clock = pygame.time.Clock()

    ball = Ball()
    left = Paddle(LEFT_X, is_left=True)
    right = Paddle(RIGHT_X, is_left=False)

    # Simple right paddle AI: follow ball with delay
    def right_ai():
        target = ball.y - (right.y + PADDLE_H/2)
        return np.clip(target * 0.25, -8.0, 8.0)

    # Controller selection
    state_dim = len(build_state(ball, left, right))
    if args.adapter == 'linear':
        use_bnn = False
    elif args.adapter == 'bnn':
        use_bnn = PMFLOW_IMPORT.get('available', False)
    else:
        use_bnn = PMFLOW_IMPORT.get('available', False)

    if use_bnn:
        controller = BNNAdapter(state_dim=state_dim, device=device)
    else:
        controller = LinearAdapter(in_dim=state_dim, lr=0.02)

    paused = False
    human_left = False
    score_left = 0
    score_right = 0
    total_hits = 0
    total_misses = 0
    last_reward = 0.0
    rolling_loss = []

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
                elif event.key == pygame.K_h:
                    human_left = not human_left
                elif event.key == pygame.K_r:
                    ball.spawn_random(to_left=random.random()<0.5)
                elif event.key == pygame.K_j:  # negative reward
                    last_reward = -1.0
                elif event.key == pygame.K_k:  # positive reward
                    last_reward = +1.0

        if not paused:
            # Step ball
            ball.step()

            # Collisions with paddles
            brect = ball.rect()
            lrect = left.rect()
            rrect = right.rect()

            hit_left = brect.colliderect(lrect) and ball.vx < 0
            hit_right = brect.colliderect(rrect) and ball.vx > 0

            if hit_left:
                ball.vx *= -1
                # Add spin from paddle movement
                ball.vy += (random.uniform(-2.0, 2.0))
                total_hits += 1
                score_left += 1
                last_reward = max(last_reward, +1.0)
            if hit_right:
                ball.vx *= -1
                ball.vy += (random.uniform(-2.0, 2.0))
                score_right += 1

            # Out of bounds (miss)
            if ball.x < 0:
                total_misses += 1
                score_right += 1
                last_reward = min(last_reward, -1.0)
                ball.spawn_random(to_left=False)
            elif ball.x > WIDTH:
                score_left += 1
                ball.spawn_random(to_left=True)

            # Move right paddle
            right.move(right_ai())

            # Left paddle control
            if human_left:
                keys = pygame.key.get_pressed()
                dy = 0
                if keys[pygame.K_UP]:
                    dy -= 10
                if keys[pygame.K_DOWN]:
                    dy += 10
                left.move(dy)
            else:
                # Build state, compute target and update controller
                state = build_state(ball, left, right)
                v_target = target_velocity(ball, left)
                loss = controller.step(state, v_target, reward=last_reward)
                rolling_loss.append(loss)
                if len(rolling_loss) > 300:
                    rolling_loss.pop(0)
                # Apply action
                v = controller.forward(state)
                left.move(v)
                # Decay reward quickly
                last_reward *= 0.9

        # Draw
        screen.fill(WHITE)
        pygame.draw.rect(screen, BLUE, left.rect())
        pygame.draw.rect(screen, RED, right.rect())
        pygame.draw.rect(screen, ORANGE, ball.rect())

        # HUD
        font = pygame.font.SysFont("consolas", 18)
        def draw_text(txt, x, y, color=BLACK):
            screen.blit(font.render(txt, True, color), (x, y))

        adapter_txt = f"Adapter: {'BNN' if use_bnn else 'Linear'}"
        pm_info = f"pmflow: {PMFLOW_IMPORT['source']} {PMFLOW_IMPORT['version']} | device: {device}"
        stats = f"Hits: {total_hits}  Misses: {total_misses}  Loss: {np.mean(rolling_loss[-60:]) if rolling_loss else 0:.3f}"
        controls = "[Space]=Pause [H]=Human [J/K]=Reward- / Reward+ [R]=Respawn [Q/Esc]=Quit"
        draw_text(adapter_txt, 10, 10)
        draw_text(pm_info, 10, 30)
        draw_text(stats, 10, 50)
        draw_text(controls, 10, HEIGHT-30)

        # Center line and scores
        for y in range(0, HEIGHT, 20):
            pygame.draw.line(screen, GRAY, (WIDTH//2, y), (WIDTH//2, y+10), 2)
        draw_text(f"{score_left}", WIDTH//2 - 40, 20)
        draw_text(f"{score_right}", WIDTH//2 + 25, 20)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
