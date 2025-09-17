# Contributing Guide

## Branching & Workflow
- **Never push directly to `main`**. Treat `main` as protected.
- Create feature branches from `main`: `feature/<short-topic>` or `fix/<issue>`.
- For large, multi-step efforts, use a scaffolding branch (`topic/<name>`) and PR early (draft) for visibility.
- Open a Pull Request (PR) to merge into `main`. Require all tests to pass locally first.

## Commits
- Write concise, conventional-style messages when possible:
  - `feat(galaxy_dynamics): add AIC/BIC model scoring`
  - `fix(fitting): handle zero error bars gracefully`
  - `docs: add Section 11.13 scaling relations`
- Avoid committing large binary artifacts (>500 KB) unless essential. Use links or generate-at-build instructions instead.

## Large Files & Data
- Do **not** commit raw dataset archives (MNIST, CIFAR, etc.).
- If small sample data is required for tests, keep under ~50 KB per file.
- If you truly need a larger static asset, consider Git LFS and document its usage.

## Testing
- Add or update tests for any public API change.
- Run the full suite: `pytest -q` and ensure 100% pass before pushing.
- Prefer deterministic tests: seed RNGs (`random.Random(42)`) for stochastic fits.

## Code Style
- Keep dependencies minimal (stdlib + existing project scope).
- Avoid introducing heavy ML / plotting libs into core packages; isolate them in `programs/demos/` if needed.

## Adding New Physics / Models
1. Start with a clear docstring + parameter definitions.
2. Provide at least one analytic or limiting test (e.g., scaling, asymptote, invariance).
3. Integrate into comparison or fitting pipelines only once baseline tests succeed.

## Documentation
- Update `docs/markdown/pushing_medium_comprehensive_documentation.md` when adding significant capabilities (new section numbering: append 11.x subsections for galaxy-related work).
- Keep the TL;DR current (snapshot test counts, major new modules).

## Model Comparison Pipeline Additions
- Expose new comparison metrics via `compare.py` and reflect in CSV export + docs.
- Include aggregation changes in `aggregate_statistics` tests.

## Performance Considerations
- Avoid O(N^2) loops for large rotation curve sampling unless N is small (< few hundred).
- Use primitive numeric loops (no heavy pandas) for hot paths.

## Opening a Pull Request
Include in the PR description:
- Summary (what & why)
- Key files touched
- Test evidence (`pytest` summary) and any new tests
- Backwards compatibility notes / migrations

## After Merge
- Delete merged feature branches to keep repo tidy.

## Emergency Fixes
If `main` is broken and blocking work: create `hotfix/<issue>` branch, fix + tests, PR, merge after review.

## Contact / Questions
Open a GitHub Issue with the label `discussion` for design proposals or uncertainties.

---
Thank you for contributing and keeping the history clean!
