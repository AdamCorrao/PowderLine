# Refinement-Engine Survey — Candidates for Future PowderLine Backends

**Date:** 2026-08-19 (web survey + codebase assessment)

PowderLine's engine seam (`src/powderline/engine.py` + one self-contained
subpackage per engine, all returning the result-dict shape locked by
`tests/test_api.py`) makes new backends cheap to add. This document records
which refinement packages were evaluated, what each would take, and the
recommended order. EasyDiffraction was selected first — see
`docs/superpowers/specs/2026-08-19-easydiffraction-engine-design.md`.

## How to add an engine (the established pattern)

1. New subpackage `src/powderline/<engine>/` containing all engine logic:
   translation (`builder`/`writer`), execution (`runner`), result parsing,
   unit conversions, and a `run_<engine>_recipe(recipe, output_dir, ...)`
   adapter returning the locked result dict.
2. One lazy-import branch in `powderline/engine.py` (`_ENGINES` tuple).
3. No schema changes: engines translate the existing `GSASII_*` recipes.
4. Unsupported recipe features are rejected loudly (translation error),
   never silently ignored.
5. Optional pixi feature/environment if the engine needs dependencies the
   default environment shouldn't carry.
6. Known deferred refactor: when engine #4 lands, extract the GSAS-II
   internals from `kicker.py` (~3,900 lines) into `powderline/gsasii/` so
   every engine follows the same subpackage layout.

## Candidates evaluated (2026-08 web survey)

### DansDiffraction — not a refinement engine

[Dans_Diffraction](https://github.com/DanPorter/Dans_Diffraction) (Dan Porter,
Diamond Light Source) is **simulation-only**: CIF in → structure factors,
powder patterns, reciprocal-space maps (X-ray/neutron/magnetic/resonant). It
has no least-squares fitting of measured data, so it cannot be a backend.
Where it *could* fit: an alternative pattern-simulation provider next to the
Materials Project simulation path (pure Python, pip-installable, trivial in
pixi).

### EasyDiffraction — selected first ✅

[easydiffraction](https://easydiffraction.github.io/) (EasyScience/ESS,
BSD-3-Clause, PyPI). Real refinement: lmfit/bumps/DFO-LS minimizers over
CrysPy/CrysFML calculation engines; X-ray CW powder supported (also neutron
CW/TOF, single-crystal, PDF via bundled diffpy.pdffit2). Direct Python API —
no input files, no subprocess, no external binary — so it exercises the
multi-engine plumbing at minimum cost. Caveats: requires Python ≥3.12
(default env pins ≥3.10 → optional pixi feature), library is young and
neutron-centric; treat as a working pilot engine, not yet a production
workhorse.

### BGMN (Profex backend) — recommended second

[BGMN/Profex](https://journals.iucr.org/j/issues/2015/05/00/kc5013/) —
open source (GPL), Linux-native, built for automation: all I/O is plain ASCII
control files; the headless `bgmn` executable runs one refinement per control
file. Fast, robust, the community standard for quantitative phase analysis
(QPA), and genuinely complementary to GSAS-II (fundamental-parameters
profiles). Fits the writer→runner→parser subpackage pattern exactly (the
TOPAS subpackage is the template). Main cost: **no conda-forge package** —
build from [profex-xrd.org](https://www.profex-xrd.org/) source or bundle a
binary (GPL permits; a conda-forge recipe would be a community contribution).
BGMN also underpins several automated phase-ID tree-search tools, adjacent to
our autonomous-analysis interests.

### FullProf — feasible, on demand

Freeware (closed source, not redistributable → manual install step, like
TOPAS). Headless [`fp2k` console binary runs on Linux](https://www.ill.eu/sites/fullprof/php/FullProf_News_2011.htm)
with a built-in sequential/batch mode. Prior art to crib from:
[AutoFP](https://github.com/xpclove/autofp/),
[AlgoRun](https://github.com/bkocis/AlgoRun),
[FullProfAPP](https://fullprofapp.readthedocs.io/). Main cost: generating the
fixed-format `.pcr` control file programmatically is notoriously finicky.
Do it when a user actually asks for FullProf.

### MAUD via MILK — for texture/stress workloads

[MILK](https://onlinelibrary.wiley.com/doi/abs/10.1107/S1600576723005472)
(LANL, published J. Appl. Cryst. 2023) is a Python scripting interface to
MAUD with parallel batch refinement. The right answer if users need texture,
residual stress, or 2D-image Rietveld. Costs: Java runtime, MAUD's parameter
model maps awkwardly onto our recipes. Wait for a concrete need.

### Not worth pursuing as engines

- **powerxrd** — "minimum viable Rietveld," cubic-only; too limited.
- **RIETAN-FP** — free but closed, Windows-centric distribution.
- **SrRietveld** — dormant diffpy-era wrapper of GSAS/FullProf.
- **CrysPy directly** — already reached through easydiffraction.

### Adjacent, not engines

- **[Spotlight](https://www.nature.com/articles/s41598-025-92452-4)** (2025) —
  global-optimization *orchestrator* over MAUD/GSAS-II; would sit on top of
  PowderLine (driving many `run()` calls), not inside it.
- **diffpy.pdffit2 / diffpy-cmi** — PDF refinement (conda-forge, Python API).
  A future PowderLine capability rather than a Bragg engine; note that
  easydiffraction already bundles pdffit2, giving us a low-cost path to a PDF
  workflow.

## Recommended order

1. **EasyDiffraction** (in progress) — proves the engine seam end-to-end with
   zero external binaries.
2. **BGMN** — biggest scientific complement (QPA, FPA profiles), open source,
   Linux-native; needs packaging work.
3. **FullProf** — when user demand appears.
4. **MAUD/MILK** — when texture/stress workloads appear.
5. Alongside any of these: `powderline/gsasii/` extraction refactor once the
   engine count justifies it.
