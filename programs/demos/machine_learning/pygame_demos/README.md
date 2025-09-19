# Sensory Remapping Demo (pygame)

A real-time demo of sensory–motor remapping and online adaptation. An agent (blue) tries to follow a moving target (red). Mid-run, the mapping between sensed input and motor commands changes (identity → swap → invert → rotate). The controller must adapt online without resetting.

## What you should see
- A white window with a red target moving and a blue agent following it.
- A grey arrow from the agent shows the currently sensed vector (after remapping) toward the target.
- The HUD shows:
  - Current remap mode (Identity/Swap/Invert/Rotate90)
  - Adapter type (BNN via nn_lib_v2 if available, otherwise Linear)
  - Rolling loss (lower is better tracking)
  - pmflow install source and version, selected device, and FPS target
- Every few seconds (default 8s), the remap switches automatically; the agent re-adapts on the fly.

## Controls
- Space: Pause/resume
- M: Manually cycle the remap mode
- R: Force an immediate remap switch
- Q or Esc: Quit

## Requirements
- pygame, numpy
- torch and `pmflow_bnn` (optional, for the BNN adapter)

## Using pmflow_bnn (nn_lib_v2)
This demo mirrors the notebook’s install/import flow:
1) Try a GitHub pip install from the nn_lib_v2 subdirectory
2) Fall back to a local path in this repo: `programs/demos/machine_learning/nn_lib_v2`
3) Import `pmflow_bnn` and display its version/source in the HUD

If pmflow is not available, the demo automatically falls back to a simple linear adapter.

## Run it
From the repo root:

```bash
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py
```

Useful flags:
```bash
# Don’t attempt GitHub install; only use local path or existing env
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --no-github-install

# Force adapter selection
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --adapter bnn
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --adapter linear

# Device selection (auto tries CUDA if available)
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --device auto
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --device cpu
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --device cuda

# Override local nn_lib_v2 path
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --nn-lib-path /abs/path/to/nn_lib_v2

# Adjust visuals/behavior
python programs/demos/machine_learning/pygame_demos/sensory_remapping.py --fps 30 --auto-switch 5
```

## How it works (high level)
- The world spawns a target that bounces around the window.
- The agent senses a remapped version of the vector to the target.
- The controller outputs a 2D motor command to move the agent.
- Online learning updates the controller so it keeps the agent close to the target despite mapping changes.

### Adapters
- Linear Adapter: y = W x + b, trained with simple SGD on every frame to minimize position error.
- BNN Adapter: Uses `pmflow_bnn` as a frozen feature extractor. A small trainable linear head maps features → 2D command. The 2D sensed input is zero‑padded to the model’s expected input length, and only the head is trained online.

## Tips
- If the BNN adapter fails to initialize (e.g., torch missing), the demo falls back to Linear automatically.
- Watch the rolling loss when a remap happens—loss spikes, then declines as the adapter relearns the mapping.

## Troubleshooting
- No window / pygame error: Ensure `pygame` is installed in your active environment.
- pmflow not found: Use `--no-github-install` to skip pip, or pass `--nn-lib-path` to point to your nn_lib_v2.
- CUDA not used: Use `--device cuda` and ensure your torch install supports CUDA on your machine.

---

# Plasticity Pong (pygame)

Classic Pong where the left paddle is controlled by an adaptive controller (BNN via nn_lib_v2 if available, otherwise Linear). The ball’s speed, spin, and spawn location vary unpredictably; the controller adapts online from sparse rewards and non-stationary dynamics.

## What you should see
- A standard Pong arena. Left paddle (blue) is AI, right paddle (red) is simple scripted.
- Ball (orange) spawns with changing speed/spin and positions. Collisions and out-of-bounds update hits/misses.
- HUD shows adapter type, pmflow source/version, device, rolling loss, and controls.

## Controls
- Space: Pause/resume
- H: Toggle human control for the left paddle (arrow keys) vs AI
- J/K: Manual reward modulation — J = negative (-1), K = positive (+1)
- R: Respawn ball with new speed/spin/spawn
- Q or Esc: Quit

## Run it
```bash
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py

# Use local only (skip pip)
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py --no-github-install

# Force adapter selection
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py --adapter bnn
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py --adapter linear

# Device selection
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py --device auto
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py --device cpu
python programs/demos/machine_learning/pygame_demos/plasticity_pong.py --device cuda
```

## How it adapts
- State encodes relative positions, ball velocity/spin, and opponent cue.
- Controller outputs paddle vertical velocity each frame.
- Online update uses a reward-modulated objective:
  - Hits give positive reward; misses give negative reward.
  - You can inject manual reward with J/K to steer behavior.
- BNN adapter freezes `pmflow_bnn` base model as a feature extractor and trains only a small head. Linear adapter directly updates a weight vector.

## Tips
- If the BNN adapter can’t initialize, it falls back to Linear. You can force either mode via `--adapter`.
- Watch the rolling loss and hit/miss counters as difficulty shifts; good adaptation drives down loss and increases hits over time.
