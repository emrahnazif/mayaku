# AGENTS.md: guide for AI agents working on mayaku

mayaku trains real-time detection, instance segmentation, keypoint, oriented-box and
classification models on the user's own data, and exports them to ONNX, Core ML, OpenVINO and
TensorRT, int8 included. Pure PyTorch, no custom kernels, one `pip install`. Its users are
developers who install it from PyPI. Every decision here is judged by what is best for them.

## Ownership

One agent session maintains each project, and only that session changes its repo.
You are the owner and long-term maintainer of this codebase, and its design is up to you.

- You may read other repos (Mozo, PixelFlow, Babasu), for example to see how they use mayaku.
  Never edit them, commit in them, or run git commands that write there.
- Find another project's session with `ListAgents`.
- mayaku depends only on released packages from PyPI, never on a local checkout's unreleased
  code.

## Requests between projects

Other projects may ask mayaku for things, and mayaku may ask them. Always with `SendMessage` to
the owning session. Never work around a missing feature by editing the other repo or copying
its code into mayaku.

A request says:

1. **What** is needed, as the caller would write it (`model.export("onnx", int8=True)`, not
   "quantisation support").
2. **Why**: the user-facing problem it solves, not only the requester's own workflow.
3. **Proposed API**, if any, and what the requester does today without it.

When mayaku receives a request:

- **Judge it only by mayaku's users.** Would a typical mayaku user want this? Accept, propose a
  more general version, or refuse with a reason. A need only one project has stays in that
  project.
- **Prefer the general fix** over the one shaped around the requester's workflow.
- **New or changed public API needs the user's approval** before it is committed. Removing
  public API is a breaking change.
- **Reply** with the decision, and later with the version that ships it.

## The deploy contract

Every exported model is a graph of five op types and nothing else:

| Op        | Allowed form                          |
|-----------|---------------------------------------|
| `Conv`    | 1x1 and 3x3, stride 1 or 2, with bias |
| `Relu`    |                                       |
| `Add`     | same shape                            |
| `Resize`  | 2x nearest                            |
| `MaxPool` | 3x3, stride 1                         |

plus `QuantizeLinear` / `DequantizeLinear` in int8 exports. This is why one model runs on every
runtime and NPU. The set is `DEPLOY_OPS` in `src/mayaku/model/contract.py`, documented in
`src/mayaku/model/blocks.py`, and asserted in `tests/unit/test_model.py`.

- Never add an op to the deploy graph, and never loosen the check. A new block must fuse
  (`Fusible.fuse()`) into these five ops, or it does not ship.
- int8 keeps accuracy: an int8 export normally scores within 0.3 AP of its fp32 model on the
  same validation set. A larger drop is a bug to find, not a tolerance to raise. The export
  parity check (`parity()` in `src/mayaku/inference/export/__init__.py`) runs on noise input
  and only catches broken conversions; judge int8 changes by AP.

## Rules

- Never commit, push, tag or release without the user asking. No `Co-Authored-By` lines and no
  mention of Claude in commits.
- Changes go under `## Unreleased` at the top of the root `CHANGELOG.md`, in plain language.
- Keep the public API small: adding API later is easy, removing it breaks users.
- Export runtimes (onnxruntime, OpenVINO, Core ML, TensorRT) are installed on first use
  (`src/mayaku/utils/requirements.py`). Never add them to `dependencies`.
- The lint and type tools are pinned in the `dev` and `typecheck` groups of `pyproject.toml`.
  Bump them deliberately, never as a side effect.
- Numbers in docs, README and commit messages are measured, with the model, dataset and
  hardware they came from.

## Docs

The docs in `docs/` are published on https://emrah.fr/mayaku/. Before writing or changing
anything in `docs/`, read `DOCS-GUIDE.md` and follow its rules.

## Before you finish

```bash
ruff check src tests
mypy
python -m pytest tests/ -q
```

Tests carry markers for the hardware they need (`cuda`, `mps`, `multi_gpu`) and for long runs
(`slow`). mayaku is tested on every engine: run the tests the change touches on each backend
and export runtime it affects, not only on the CPU. CI runs only the CPU subset
(`-m "not cuda and not mps and not multi_gpu"`), so a green CI is not proof for the other
engines.

All checks must pass. Show the result; don't just say it passed.

## Releasing (only when the user asks)

1. Set `__version__` in `src/mayaku/__init__.py` (the package version is read from there).
2. In `CHANGELOG.md`, turn `## Unreleased` into a `## vX.Y.Z: <title>` section.
3. Run the checks above, commit, then tag `vX.Y.Z` (or `vX.Y.ZrcN` for a pre-release). Nothing
   checks that the tag matches `__version__`: check it yourself.
4. Pushing the tag starts `.github/workflows/release.yml`, which builds and, after manual
   approval in the `pypi` environment, publishes to PyPI. A published version cannot be
   replaced: the user decides when.

## Stop and ask the user

- New, changed or removed public API.
- Any change to the deploy op set, or an int8 AP drop above 0.3.
- Releases, tags, pushes, version numbers.
- New dependencies, new models or tasks, or licence changes (weights licences included).
- Deleting or rewriting files or commits you did not make.
