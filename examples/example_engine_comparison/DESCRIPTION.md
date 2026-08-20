# Engine comparison: GSAS-II vs easydiffraction on the same recipe

Runs `examples/example_LaB6_easydiff/input.json` (LaB6, synchrotron λ = 0.1665 Å,
cell `a` + scale + U/V/W/X/Y + 6-term Chebyshev background refined) through both
engines via the same `powderline.run()` call, then writes a side-by-side
comparison.

## Run it

From the repo root (the `easydiff` environment contains both engines):

```bash
pixi run -e easydiff python examples/example_engine_comparison/compare_engines.py
```

Takes ~2 minutes (one full refinement per engine). Outputs land in `output/`:
the standard per-engine reports (`output/gsasii/`, `output/easydiffraction/`),
`output/comparison.csv` (merged refined parameters ± ESDs plus fit statistics),
and `output/comparison.png` (two-panel observed/calculated/difference plot).

## Reference results

| | Rwp | cell a (Å) |
|---|---|---|
| GSAS-II | 6.01 % | 4.15719 ± 0.00002 |
| easydiffraction | 9.63 % | 4.15754 ± 0.00003 |
| NIST SRM 660 | — | 4.15682 |

## How to read the comparison

- **The lattice parameter is the meaningful cross-check.** The engines agree to
  0.0004 Å (≈85 ppm) on independent implementations of the same physics.
- **Both fits are statistically excellent.** The recipe's nonzero SH/L makes the
  easydiffraction engine select the Thompson–Cox–Hastings profile with fixed FCJ
  axial-divergence asymmetry (CrysFML calculator), and both engines reach
  reduced χ² < 1 — the recipe weights overestimate σ, as is typical for
  azimuthally integrated 2D-detector data.
- **The remaining Rwp gap is expected, not a bug.** GSAS-II additionally models
  background peaks, and profile details differ; Rwp is still not an
  apples-to-apples metric across engines.
- **Scale factors are not comparable** between engines (different intensity
  normalizations); they are reported but should not be cross-read.
- Broadening parameters (U/V/W/X/Y) are reported in GSAS-II conventions by both
  engines, but they compensate for the profile-model differences above, so
  agreement is not expected to be tight.
