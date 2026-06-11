# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read AGENTS.md first

`AGENTS.md` is the canonical guide for the step pipeline: the `StepBase` contract, how to add a step, key conventions (BGR uint8 ndarrays, lazy model loading, `config.py` paths, exceptions caught by the worker). Everything there still applies. This file covers what AGENTS.md predates or gets wrong for this machine.

## Environment

AGENTS.md documents Windows (`.venv\Scripts\python.exe`, `run.bat`). **This checkout runs on Linux** — the venv interpreter is `app/.venv/bin/python` (Python 3.11). Run everything from `app/`:

```bash
cd app
.venv/bin/python main.py [path/to/image]      # launch the GUI (PyQt6)
.venv/bin/python download_models.py --check    # verify all model weights are present
.venv/bin/python download_models.py            # pre-download large models for offline use
```

There is **no automated test suite** and no lint config. `app/test_scunet.py` and `app/test_gpu.py` are standalone diagnostic scripts (run directly with the venv python), not pytest tests. Verify changes by launching the GUI.

Dependency install is delicate — see `requirements.txt` header. PyTorch installs from a separate index; `protobuf==3.20.3` is pinned because mediapipe breaks on protobuf ≥ 4; `torch-directml` and `onnxruntime-directml` are unsupported (they break gfpgan/basicsr/mediapipe). After install, run `fix_packages.py` once (patches basicsr's removed `torchvision.transforms.functional_tensor` import).

## Architecture beyond AGENTS.md

The single-image step pipeline (`core/pipeline.py` → `PipelineWorker` QThread, `steps/ALL_STEPS`) is as AGENTS.md describes. Since then the app grew several subsystems layered on top of it:

- **MainApp is composed from mixins.** `ui/main_window.py` defines `MainApp` by inheriting `ui/mixins/*` (pipeline, preview, params, views, file_io, batch, editors) plus `QMainWindow`. When extending the main window, add the method to the relevant mixin, not to `main_window.py` directly. `BatchWindow` follows the same pattern with `ui/batch_mixins/*` (run, nav, params, preview, exports, artifacts, redzones, vlm).

- **Batch mode** (`core/batch.py`, `ui/batch_window.py`): processes a folder. `BatchSession` holds per-image `BatchImageConfig` (each image has its own step order, enabled flags, params, and instance-state masks). Session state persists to `.batch_session.json` in the source folder.

- **Versioned exports** (`core/export_manager.py`, `ui/export_mosaic.py`, `ui/export_detail_panel.py`): an image can have multiple exports. Each `ExportEntry` is written to the output dir as `{stem}.export.{NNN}.*` — image + recipe JSON + optional masks (inpaint / redeye / redzone). Exports are compared, restored, and deleted purely from the output directory. Works in both single-image and batch mode. Design rationale is in `app/cahier-des-charges-evolution-export.md`.

- **History & config diff** (`core/history.py`, `core/config_diff.py`): `HistoryManager` keeps up to 30 runs (LRU on the cached images). `config_diff.changed_step_details` compares two pipeline snapshots (order + enabled + params) to decide which steps actually changed — drives "recompute only what changed" and the diff UI.

- **Artifact detection + VLM refinement** (`core/artifact_detect.py`, `core/redzone.py`, `core/vlm_refine.py`): BOPBTL detects scratches/creases; a **local VLM** (Qwen3-VL / Gemma, lazy singleton, unloadable to free VRAM) reviews suspect mask components (long lines, large blobs) to drop false positives, sending a global view + local zoom per candidate and asking a binary JSON question. The full conversation is kept for inspection in the UI. This requires a GPU and a HuggingFace token; the HF cache lives under `models/hf_cache/` (`HF_HOME` is set in `config.py` before any transformers import).

- **Couleur par référence IA** (`core/genref.py`, `steps/step_genref.py`, `ui/genref_dialog.py`): the "genref" step applies a 30-coefficient polynomial LUT learned from a FLUX.1 Kontext "reference" generation of the same image (prompt written by the local VLM, cast color auto-named — naming the cast color in the prompt is what makes Kontext edit strongly). Heavy generation happens only via explicit user action; `process()` just applies the cached β (instant, previewable). **Single mode**: panel button "Générer la référence IA…" → dialog; disk cache per (image digest, style, seed) under `models/genref_cache/`. **Batch mode**: dedicated "Réf. IA" tab (`ui/batch_mixins/genref.py`, `core/genref_store.py`) — each generation is a sidecar *version* `{stem}.genref.NNN.{json,png}` next to the source (excluded from the batch folder scan, never auto-deleted); `cfg.genref_version` selects the active one (persisted in recipes/exports); the step gets the reference injected via `set_batch_ref()` (like masks) and re-fits β on the current base automatically (~0.3 s, cached). Needs `diffusers` (installed `--no-deps`), `bitsandbytes`, `sentencepiece` — see requirements.txt notes. Design study in `do-not-commit/color-study/` and `do-not-commit/flux-ref/`.

## Conventions specific to this repo

- **`do-not-commit/`** is a local scratch dir for experiments — never import from it in `app/`, never commit it (it's gitignored).
- **Model weights are gitignored** (`*.pth`, `*.pt`, `*.onnx`, `*.safetensors`, `models/ddcolor_modelscope/`, `models/hf_cache/`). Don't commit them. `download_models.py` fetches them.
- **All model paths go through `config.py` constants** — never hardcode. `BASE_DIR` is the repo root (parent of `app/`); most weights live under `models/` or at the root (`GFPGANv1.4.pth`).
- **UI strings, comments, and docstrings are in French.** Match this when editing.
