# Plastic Text Demo: "Teach Me Your Slang"

A tiny recurrent plastic model that adapts online while you chat. It starts with simple embeddings and a small RNN cell; both update with a Hebbian-like rule based on prediction error and recency.

## What it does
- Next-token prediction on a tiny corpus (optional pre-training warmup)
- Interactive loop: you type a message, it predicts the next token(s)
- Online plasticity updates to embeddings and recurrent weights
- Optional feature backend using nn_lib_v2:
	- BNN: PMFlow-BNN v0.2.0 as a frozen feature extractor over a learned image projection of token embeddings
	- CNN: Baseline CNN as fallback when BNN is unavailable
- Optional live visualization showing a 2D projection of embeddings morphing over time

## Why
To show plasticity in action with text: new words and usages can be learned mid-conversation. It’s not a chatbot; it’s a pedagogical demo focusing on on-line adaptation behavior.

## Run
```bash
python programs/demos/machine_learning/text_demos/plastic_text_demo.py --corpus tiny --steps 200 --viz
```

### Dependencies
- Required: PyTorch (CPU is fine)
- Optional: Matplotlib (for live 2D embedding visualization)

Options:
- `--corpus tiny|shakespeare|custom.txt`  (tiny is built-in; files are relative or absolute)
- `--embed-dim 64`                        (embedding size)
- `--hidden 128`                          (RNN hidden size)
- `--lr 1e-2`                             (plasticity learning rate)
- `--beta 0.98`                           (eligibility/recency decay)
- `--viz`                                 (enable matplotlib projection of embeddings)
- `--backend auto|bnn|cnn|none`           (feature backend; default auto prefers BNN if available)
- `--nn-lib-path PATH`                    (optional path to local nn_lib_v2 if not installed)

Controls (interactive):
- Type a line and press Enter. The model predicts next token(s) and adapts.
- Type `:add <word>` to add a new vocabulary entry during the session.
- Type `:quit` to exit.

## Notes
- The demo uses simple whitespace tokenization. For better behavior, a small BPE can be dropped in.
- Plastic updates are Hebbian-inspired (outer-products and eligibility traces), not a full optimizer.
- CPU-only and intended to be lightweight.
- Backend details: the backend never trains; it serves as a feature extractor. The demo projects the token embedding to a 28x28 image and feeds it to the selected backend (BNN or CNN). The trainable part is the small linear readout that combines plastic-RNN state and backend features.

## Roadmap / Ideas
- Swap in `pmflow_bnn` as an optional feature extractor front-end.
- Expose plasticity switches per-module (embeddings only, recurrent only, both).
- Save/restore the adapted state and compare trajectories.
- Add a tiny reward signal for task-guided shaping (e.g., echo, rhyme, or style-matching).
