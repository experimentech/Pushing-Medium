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

---

## New: BNN-native Chatbot

`bnn_chatbot.py` is a freestanding demo that uses PMFlow-BNN directly for sequence memory (no image projection):

- Token embedding → tiny MLP → PMFlow latent z
- PMBNNAlwaysPlastic (v0.2.0) advances state each token
- Trainable readout to vocab; adapts online as you chat

Run:
```bash
python programs/demos/machine_learning/text_demos/bnn_chatbot.py --corpus tiny --steps 200
```

Notes:
- Requires nn_lib_v2 (pmflow_bnn) available in environment or via local path.
- Commands: `:add <word>`, `:quit`.

Teaching interface (extra commands):
- `:define <word> = <phrase>`  Pulls the word’s embedding toward the mean of the phrase, shaping meaning.
- `:alias w1 w2`               Brings the two words’ embeddings closer, teaching synonyms.
- `:neighbors <word> [k]`      Shows nearest neighbors in embedding space.
- `:concept name = w1 w2 ...`  Defines a concept set and clusters members toward a centroid.
- `:tag <word> <concept>`      Adds a word to a concept and pulls it toward the concept centroid.
- `:gen [n] [temp] [topk]`     Adjusts reply generation length/temperature/top-k.
- `:stats`                     Displays top co-occurring word pairs observed.
- `:save path` / `:load path`  Save/restore embeddings and readout.

### Effective usage: teaching workflow

1) Start with optimized PMFlow settings
- Prefer the always-plastic model and an autoscaled profile:
	- `--pmflow-model always_plastic_v2 --pmflow-profile auto --plasticity-lr 2e-3`

2) Seed a tiny base corpus (optional)
- Use `--corpus tiny --steps 200` to pre-warm the readout layer on simple next-token pairs.

3) Introduce vocabulary in-context
- Just type sentences; unknown words are auto-added and begin adapting via co-occurrence:
	- `A banana is a type of fruit.`
	- `An apple is red or green.`

4) Teach definitions and clusters explicitly
- Definitions pull a word toward a phrase’s mean embedding and apply PMFlow plasticity from that phrase:
	- `:define apple = a sweet fruit`
- Concepts cluster multiple tokens and adapt PMFlow to the cluster:
	- `:concept fruit = apple banana orange pear`
	- `:tag mango fruit`

5) Inspect and refine
- Nearest neighbors reveal semantic neighborhoods:
	- `:neighbors apple 10`
- Co-occurrence stats highlight frequent pairs:
	- `:stats`

6) Generate replies and tune decoding
- Adjust length/temperature/top-k for the reply generator:
	- `:gen 16 0.9 8`

7) Save progress
- `:save runs/fruit_session.pt` and later `:load runs/fruit_session.pt`

### Troubleshooting
- The bot repeats `<unk>`: add words with `:add` or define them with `:define ...`.
- Meanings feel unstable: reduce `--plasticity-lr` (e.g., `1e-3`) or increase warmup `--steps`.
- Associations are too weak: repeat `:define`/`:concept` or temporarily increase `--plasticity-lr`.
- Performance: set `--pmflow-profile cpu` on low-memory systems or `single_gpu`/`multi_gpu` if available.

### Worst Chatbot Ever: why it doesn’t work (on purpose)

This chatbot is intentionally “bad” as a teaching artifact. It’s a tiny plastic memory with almost no language prior. When you feed it a whole book, it happily absorbs statistics it can’t use for fluent conversation.

What it is not:
- A trained language model. No large pretraining; only a small readout with light warmup.
- An expert system. `:define / :concept / :alias` nudge vectors and PM centers; no symbolic rules/facts.
- A capable sequence decoder. PMFlow-BNN here is a plastic attractor memory, not a predictive LM.

Why it fails at chat:
- Frequency dominance: ingesting big text floods the head with high-frequency tokens → loops/echoes.
- Unlabeled ingestion: reading lines doesn’t optimize a next-token objective; it shapes co-occurrence only.
- Thin decoding: even with top-p and repetition penalties, coherence is limited without a strong model.
- Catastrophic drift: online plasticity pulls meanings; recent inputs dominate.
- Naive tokenization & boilerplate: whitespace tokens and public-domain headers pollute the space.

Why keep the BNN core:
- PMFlow-BNN (AlwaysPlastic) is a neat, continuous, local memory that adapts in real time—great for demos of plasticity, concepts, and neighborhoods.

Mitigations (to make it less terrible, not good):
- Ingestion: `--train-quiet`, multiple `--train-script`, prepare cleaner input (filter headers/stopwords).
- Stability: `--fast-repl`, `--plastic-every N`, `--renorm-head`, optional `--consolidate-steps K`.
- Decoding: `:gen … top_p rep_penalty`, `:bias co X concept Y`.
- Retrieval: “what is X” returns stored `:define` phrases; simple concept membership answers.

Reproduce the failure and explore:
```bash
python programs/demos/machine_learning/text_demos/bnn_chatbot.py \
	--pmflow-profile cpu \
	--pmflow-model always_plastic_v2 \
	--train-quiet --fast-repl --plastic-every 8 --renorm-head \
	--train-script programs/demos/machine_learning/text_demos/training/fruit_bootstrap.txt \
	--train-script programs/demos/machine_learning/text_demos/training/the_call_of_cthulhu.txt
```
Then try `:define` / `:concept` and ask “what is X”. It will still be a terrible chatbot—by design—but you’ll see plasticity in action.
