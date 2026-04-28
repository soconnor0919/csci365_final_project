# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All Python work runs in the `dip26` conda environment (Python 3.12):

```bash
# Run a script
/opt/homebrew/Caskroom/miniconda/base/envs/dip26/bin/python script.py

# Run Jupyter
conda run -n dip26 jupyter notebook
```

Key packages: `torch 2.10`, `torchvision 0.26`, `opencv-python 4.11`, `scipy 1.17`, `simple-lama-inpainting 0.1.2`, `yt-dlp`. MPS (Apple Silicon GPU) is available and used for model inference.

## Project Structure

Single-notebook project. **`halo_inpainting 3.ipynb`** is the only source of truth — do not create additional notebooks or `.py` modules. The only other tracked files are `data/suzuka_raw.mp4` (source video) and `output/` (exported MP4s).

## Pipeline Architecture

The notebook runs in strict section order (enforced by AGENTS.md):

1. **Setup** (`cell-setup`) — imports only; no torchvision VGG
2. **Data load** (`cell-download`, `cell-load`) — yt-dlp download + clip extraction. Working window: `T_START=5.0s`, `DURATION=5.0s` → 300 frames at 60fps
3. **Frame exploration** (`cell-explore`) — visualization only
4. **Initial T-mask** (`cell-rough-mask`, `cell-refine-mask`) — static arch+strut mask; dark-pixel refinement was permanently discarded (wide-angle vignetting defeats it)
5. **Per-frame mask detection** (`cell-lk-track`) — `build_frame_mask()` returns `(mask, keel_cx_out)`; the build loop threads `prev_keel_cx` forward as the occlusion fallback
6. **Drift check** (`cell-warp-masks`, `cell-drift`) — sets `warped_masks = masks`; assigns `masks = warped_masks` (no-op for visor cam which always drifts)
7. **Method 1 — LaMa** (`cell-spatial`) — `SimpleLama(device=DEVICE)` on MPS; ~2 min for 300 frames
8. **Method 2 — RAFT temporal** (`cell-temporal-fn`, `cell-temporal-run`) — RAFT-small flow + backward warp + distance-transform blend; resets every 60 frames
9. **Output** (`cell-animation`, `cell-write-video`) — side-by-side animation + `output/{original,spatial,temporal}.mp4`

## Key Design Decisions

**Mask detection (`build_frame_mask`)**
- Arch: Sobel-Y per-column argmax in top 24% of frame; F1 logo region (x<230, y<88) replaced with sky brightness before computing Sobel
- Keel: 6 y-probes at rows 195–270, search x in [40%, 63%] of width; probes rejected if within 15px of search boundary or contrast < 10 gray levels
- Slope fit: `np.polyfit` on `(y, cx)` probe pairs; outlier probes with residual > 20px are dropped and the fit is redone; hard clamp at ±1.5 px/row only for pathological frames
- Occlusion fallback: when 0 probes detected (hands over keel), use `prev_keel_cx` instead of frame center
- Keel mask shape: tapered half-width (wider at junction with arch, narrower at bottom); junction bonus for top 40 rows

**Why VGG PatchMatch was replaced by LaMa**
PatchMatch copies patches from the same-frame unmasked pixels. For a full-width arch mask the only source pixels are cockpit/track, not sky — the result is wrong-colored blocks. LaMa uses Fast Fourier Convolutions with global receptive field, trained on natural scenes, so it synthesizes coherent sky content.

**Why LK tracking was abandoned**
Steering-wheel hands inject non-rigid motion into any affine estimate, causing drift. Independent per-frame detection is more robust.

**Output filenames**
Never overwrite `output/original.mp4`. New method outputs go to distinct names (e.g., `spatial.mp4`, `temporal.mp4`).

## Running Diagnostics Outside the Notebook

To debug mask quality without running the full notebook:

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/dip26/bin/python - << 'EOF'
import cv2, numpy as np
# paste build_frame_mask definition here, then iterate over frames
EOF
```

Useful metrics to track: per-frame slope, probe count, keel_cx, number of outlier probes dropped.
