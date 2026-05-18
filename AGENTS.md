# Restauration Photo — Agent Instructions

## Project Overview

PyQt6 GUI application for scanning and AI-based restoration of old photos. The app loads an image and applies a configurable pipeline of processing **steps** (face restoration, denoising, color correction, red-eye removal, etc.). The main application code lives entirely in `app/`.

**Language:** Python 3. UI strings and comments are in French.

---

## Running the App

```batch
cd app
.venv\Scripts\python.exe main.py [path/to/image]
```

Or use the launcher:

```batch
app\run.bat [path/to/image]
```

### Installing Dependencies

```bash
cd app
.venv\Scripts\pip install -r requirements.txt
# PyTorch must be installed separately:
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Architecture

```
app/
  main.py            ← Entry point; creates QApplication + MainApp
  config.py          ← All model paths and base directories
  core/
    pipeline.py      ← PipelineWorker (QThread): runs steps sequentially
    history.py       ← HistoryManager: stores per-run results
    lama_inpaint.py  ← LaMa inpainting wrapper
  steps/
    __init__.py      ← ALL_STEPS list: singleton step instances, pipeline order
    base.py          ← StepBase abstract class
    step_*.py        ← One file per step
  ui/
    main_window.py   ← MainApp(QMainWindow): top-level UI orchestrator
    control_panel.py ← Left panel: run/stop button + step list
    step_panel.py    ← Per-step collapsible panel with params
    image_view.py    ← Zoomable image widget
    history_panel.py ← Version chip bar
    thumbnail_strip.py ← Per-step result thumbnails
    param_widgets.py ← Slider/combo widgets bound to param_defs
    mask_editor.py   ← Paint-a-mask overlay for inpainting
    wb_picker.py     ← White-balance color picker
```

### Data Flow

1. User opens image → stored as `MainApp._original` (BGR `uint8` ndarray)
2. User clicks **Lancer** → `PipelineWorker` started in background thread
3. For each enabled step (in `ALL_STEPS` order):
   - `step.process(img, params, context)` → `(result_img, extras_dict)`
   - `extras` merged into shared `context` dict (e.g. `face_bboxes` from GFPGAN → consumed by SCUNet)
   - Signals emitted: `step_started`, `step_done`, `step_failed`
4. `all_done` → `HistoryEntry` saved, UI updated

---

## Adding a New Step

1. Create `app/steps/step_<name>.py` with a class extending `StepBase`:
   ```python
   from .base import StepBase

   class MyStep(StepBase):
       id                = "my_step"
       name              = "Mon étape"
       short_name        = "MyStep"
       slow              = False           # True → shows spinner
       enabled_by_default = True
       has_overlay       = False
       has_mask_editor   = False
       has_color_picker  = False

       param_defs = [
           {"key": "strength", "label": "Force", "type": "float",
            "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
       ]

       def process(self, img, params, context):
           # img: BGR uint8 ndarray
           # params: dict matching param_defs keys
           # context: shared dict (read face_bboxes etc.)
           result = ...
           extras = {}  # optional keys to add to context
           return result, extras
   ```

2. Register in `app/steps/__init__.py` — add a singleton instance to `ALL_STEPS` at the desired position.

3. The UI (params panel, thumbnail, enable checkbox) is generated automatically from `param_defs`.

---

## Key Conventions

- **Model paths** are defined in `app/config.py`. Always use `config.py` constants, never hardcode paths.
- **Models are lazy-loaded** on first `process()` call. Cache on the step instance (e.g. `self._model = None`).
- **BGR color order** throughout (OpenCV convention). Convert RGB↔BGR only at model boundaries.
- **Images are `np.ndarray` (uint8, BGR)**. Steps receive and return this format.
- **Exceptions in `process()` are caught** by `PipelineWorker` — the step is skipped and previous image is used. No need to swallow exceptions in steps.
- **UI state is owned by `MainApp`**. Steps must not import or reference UI classes.
- `param_defs` supports types: `"float"`, `"int"`, `"choice"` (with `"choices": [...]`).

---

## Model Files

| Model | Path |
|-------|------|
| GFPGANv1.4 | `GFPGANv1.4.pth` (root) |
| SCUNet GAN | `models/scunet_color_real_gan.pth` |
| SCUNet PSNR | `models/scunet_color_real_psnr.pth` |
| SCUNet network def | `models/network_scunet.py` |
| Face detection | `app/gfpgan/weights/detection_Resnet50_Final.pth` |
| Face parsing | `app/gfpgan/weights/parsing_parsenet.pth` |
| MediaPipe landmarks | `app/face_landmarker.task` |

---

## do-not-commit/

`do-not-commit/` is a **local working directory** for experiments, backups, and studies. It must not be committed. Do not reference or import from it in `app/` code.
