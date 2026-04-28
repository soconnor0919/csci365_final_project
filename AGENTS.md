# AGENTS.md

## Purpose
Project-scoped instructions for the DIP S26 final demo workflow.

## Core Workflow
- Keep one primary runnable notebook as the source of truth for the demo pipeline.
- Preserve section order: data load, mask detection, spatial inpaint, temporal inpaint, evaluation, export.
- Centralize tunable parameters in one notebook section; avoid hidden constants in later cells.

## Reproducibility
- Prefer minimal dependency changes.
- If a package must be added, install it in the notebook and briefly document why.
- Keep lab-machine compatibility in mind: deterministic frame windows and explicit seeds where applicable.
- Re-run from a clean kernel before finalizing outputs.

## Diagnostics and Evaluation
- Track difficult-segment diagnostics:
- Keel center stability
- Keel skip-rate under occlusion
- Mask area variation
- Masked-region temporal jitter
- Standardize visual comparisons with fixed keyframes and identical clip settings (FPS, duration, frame indices).

## Artifacts
- Keep baseline and improved outputs side-by-side; do not overwrite baseline artifacts.
- Save exports to the project output folder with clear filenames.
- Add concise markdown describing what changed, why it helps, and known failure cases.

## Runtime Practicality
- Keep runtime practical for classroom demos.
- Tune on shorter frame windows first, then run a full export for final results.
