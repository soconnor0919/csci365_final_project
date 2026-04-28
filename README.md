# F1 Halo Removal via Video Inpainting

**CSCI 365 — Digital Image Processing, Spring 2026**

Removes the Halo titanium safety arch from F1 onboard (visor-cam) footage using classical computer vision for mask detection and neural inpainting to fill the removed region.

---

## The Problem

Modern Formula 1 cars carry a mandatory titanium safety structure called the **Halo** — a curved arch that wraps over the driver's head and connects to a central keel strut pointing down between their eyes. It saves lives, but it also cuts directly through the most interesting part of onboard camera footage: the driver's view of the circuit ahead.

The Halo appears as a **T-shape** in visor-cam footage:
- A **full-width arch band** across the top ~24% of the frame — dark titanium against bright sky
- A **central keel strut** descending from the arch midpoint down to ~65% of frame height

Removing it cleanly requires:
1. Knowing *exactly* which pixels belong to the Halo in every frame (mask detection)
2. Synthesizing believable replacement content — primarily sky, grandstands, and track — behind it (inpainting)

Both are non-trivial. The visor camera is body-mounted to the driver's helmet, so the entire image rotates as the driver looks left and right through corners. The Halo drifts across the frame, tilts, and is intermittently occluded by the driver's hands and steering wheel.

---

## Pipeline Overview

```
suzuka_raw.mp4
      │
      ▼
[1] Frame extraction (OpenCV)
      │  T_START=5.0s, DURATION=5.0s → 300 frames @ 60fps, 1280×720
      ▼
[2] Per-frame mask detection (classical CV)
      │  Sobel-Y filled arch body + fitted keel + F1 logo contour
      ▼
[3] Method 1 — LaMa spatial inpainting (neural, per-frame)
      │  Fast Fourier Convolution network, pretrained, MPS/CUDA inference
      ▼
[4] Method 2 — RAFT temporal propagation (neural, frame-to-frame)
      │  Dense optical flow → backward warp → distance-transform blend
      ▼
[5] Export: output/spatial.mp4, output/temporal.mp4
```

Everything runs in **`halo_inpainting.ipynb`**. There is no other source file.

---

## Mask Detection (Section 4)

### Arch

The arch's bottom edge is a sharp dark→bright transition (titanium to sky) that is always visible — it sits above the driver's head and is never occluded. Detection per frame:

1. Restrict to top 24% of frame. Replace the F1 logo region (x < 230 px, y < 88 px) with the median sky brightness sampled from adjacent unaffected columns — prevents the logo's dark pixels from corrupting the edge detector.
2. Apply **Sobel-Y** (vertical gradient, ksize=5). Keep only positive values (dark-to-bright transitions going downward).
3. Per column, take the row of the maximum Sobel response as the arch bottom. Columns with weak response (< 15% of max) default to the band boundary.
4. Smooth with median filter (size=21) then Gaussian (σ=30px) to produce a continuous contour.
5. Use that contour as the lower anchor for a controlled filled arch body. This avoids masking the whole roof/sky slab while still covering the Halo interior and bright sunlit edges.

### Keel

The keel is partially occluded in many frames (steering wheel, driver's hands). Detection:

1. Sample 6 horizontal probe rows at y = {195, 210, 225, 240, 255, 270} px, within x ∈ [40%, 63%] of frame width.
2. Each probe finds the column of minimum brightness (darkest point = keel centre). Reject probes within 15 px of the search boundary (likely picking up the cockpit surround) and probes where contrast < 10 gray levels (keel is occluded).
3. Fit a line `cx = slope·y + intercept` through accepted `(y, cx)` probe pairs using `np.polyfit`.
4. **Outlier removal**: compute per-probe residuals; drop any probe > 20 px from the fit and refit. This handles cases where the steering wheel bar registers at one probe level, corrupting the slope.
5. **Pathological guard**: if `|slope| > 1.5 px/row` after outlier removal, fall back to a vertical mask at the median probe cx.
6. **Occlusion fallback**: if zero probes pass the quality gate (hands fully covering the keel), carry forward the previous frame's keel centre rather than jumping to frame centre.

The resulting keel mask is drawn per-row at the fitted `cx`, with half-width that tapers from wider at the arch junction (extra coverage for the T-joint blend zone) to narrower at the bottom.

### Controlled mask construction

The active mask is now built from explicit geometry instead of allowing dark contours to flood the region:

1. Build a filled arch-body region upward from the detected Sobel lower edge to the top video border, with only a tiny lower pad for antialiased rim pixels.
2. Draw a tapered keel mask along the robust fitted keel line.
3. Add a widened yoke patch plus a rounded ellipse where the keel meets the arch.
4. Extract the fixed F1 logo as white contours from the top-left crop and add only those contour pixels.
5. Apply close, hole-fill, and a modest final dilate cleanup to cover sunlit bright rims and remove internal gaps.

This keeps the roof/visor strip outside the mask while covering the Halo body, its brighter borders, and the permanent broadcast logo.

### Why slope matters

The visor camera rotates with the driver's head. Through fast corners, a physically vertical keel projects at up to ~−0.9 px/row of image-plane tilt (top of keel displaced 65+ px right of bottom over the probe range). A vertical mask would miss the top or bottom of the keel. The tilt-aware fit keeps the mask aligned.

---

## Method 1 — LaMa Spatial Inpainting (Section 5)

**Model:** `simple-lama-inpainting` (wraps Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions*, WACV 2022).

LaMa replaces the masked region with plausible content synthesized from global image context. Its key property — **Fast Fourier Convolutions** — gives every layer an effective receptive field equal to the full image. This matters here because the arch mask is ~24% of frame height and full-width; local patch-copy methods have no valid source pixels (cockpit and track are below the mask, not sky).

LaMa is run with pretrained weights (trained on Places365 + other natural scene datasets). No fine-tuning on this video is needed or done.

After LaMa inference, a 7×7 Gaussian soft-blend at mask boundaries merges the filled region with the original frame.

**Why not VGG PatchMatch?**
The earlier Method 1 used VGG-16 relu2_2 features to find best-matching 16-px patches in the unmasked region of the same frame, then copied them. This fails because the unmasked pixels for the arch region are cockpit, steering wheel, and track — there is no sky content to copy. LaMa *generates* sky rather than copying it.

---

## Method 2 — RAFT Temporal Propagation (Section 6)

**Model:** `torchvision` RAFT-small (Teed & Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020).

For each frame *t*:
1. Estimate dense optical flow between the previous *clean* frame and the current LaMa-inpainted frame using RAFT. Using the LaMa frame rather than the raw frame avoids the Halo pixels contaminating the flow estimate.
2. Backward-warp the previous clean frame into the current frame's coordinates.
3. Blend: pixels far from the mask boundary (deep inside the mask region, measured by distance transform) are taken from the warp; pixels near the boundary are taken from the current frame. This avoids "ghosting" at edges where the warp is inaccurate.
4. Every 60 frames, reset the temporal chain to the LaMa result to prevent accumulated warp errors from drifting.

**Why not Farnebäck?**
Farnebäck estimates flow via local polynomial expansion, which breaks down at the fast lateral camera panning through Suzuka's fast corners and under the complex layered backgrounds (ferris wheel, grandstands, pit lane structures all moving at different apparent speeds). RAFT's learned feature representations handle large displacements and motion blur far more robustly.

---

## What Still Needs Doing

### Known issues
- **Frames 13–59**: ~47-frame occlusion window where the keel is fully covered by hands/wheel. The mask correctly holds the last-known cx, but the inpainter has to synthesize a large contiguous region without temporal reference. Temporal quality may be lower here.
- **Arch/keel junction**: The junction zone (where the keel meets the arch) is an edge case — the entire row is masked so the horizontal scanline fill has no source pixels. Currently falls through to LaMa's output for that zone.
- **Keel lower boundary**: The keel mask terminates at 65% of frame height (`strut_y1=0.65`). Frames where the keel extends further (driver leaning back) may show residual keel at the bottom of the mask region.

### Possible improvements
- **Temporal mask smoothing**: Apply a small temporal median filter to `keel_cx` values to reduce per-frame jitter, especially during the transition back from occlusion at frame 60.
- **Longer clip**: The current 5-second / 300-frame window was chosen for iteration speed. A full-length export would need `DURATION` extended and the `output/` filenames versioned to avoid overwriting baseline results.
- **Evaluation metrics**: PSNR/SSIM against a hand-composited ground truth for a few keyframes would give a quantitative comparison of Method 1 vs Method 2.
- **Probe robustness**: The 6 probe rows are hardcoded in pixel coordinates. For clips from different moments in the race (different helmet height in frame), these may need adjustment. Deriving probe rows from the detected arch bottom would make this automatic.

---

## Running

```bash
# Install dependencies (first time)
uv sync

# Launch notebook
uv run jupyter notebook "halo_inpainting.ipynb"
```

Run cells top-to-bottom. The two slow cells are:
- **`cell-spatial`** (LaMa): ~2 min on MPS, ~25 min on CPU for 300 frames
- **`cell-temporal-run`** (RAFT): ~3–5 min on MPS for 300 frames

Tune `T_START` and `DURATION` in `cell-load` to work on a shorter window during development.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.11 | frame I/O, image ops, Sobel, remap |
| `torch` / `torchvision` | 2.10 / 0.26 | RAFT model, MPS/CUDA backend |
| `simple-lama-inpainting` | 0.1.2 | LaMa pretrained inference |
| `scipy` | 1.17 | `median_filter` for arch contour smoothing |
| `yt-dlp` | 2026.3 | source video download |
| `matplotlib` | — | visualization, animation export |

All packages are installed in the `dip26` conda environment (Python 3.12, Apple Silicon).
