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
| easydiffraction | 19.77 % | 4.15747 ± 0.00005 |
| NIST SRM 660 | — | 4.15682 |

## How to read the comparison

- **The lattice parameter is the meaningful cross-check.** The engines agree to
  0.0003 Å (≈70 ppm) on independent implementations of the same physics.
- **The Rwp gap is expected, not a bug.** easydiffraction's pseudo-Voigt profile
  has no SH/L axial-divergence asymmetry model and no background peaks; the
  GSAS-II model includes both. The residual panel makes this visible — the
  largest easydiffraction misfit is the asymmetric low-angle peak at 2θ ≈ 2.3°.
- **Scale factors are not comparable** between engines (different intensity
  normalizations); they are reported but should not be cross-read.
- Broadening parameters (U/V/W/X/Y) are reported in GSAS-II conventions by both
  engines, but they compensate for the profile-model differences above, so
  agreement is not expected to be tight.
