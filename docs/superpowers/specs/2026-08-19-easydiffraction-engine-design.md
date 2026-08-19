# EasyDiffraction Engine for PowderLine — Design

**Date:** 2026-08-19 (rev 2, after critical review against schema.py, test_api.py,
the LaB6 example recipe, and the easydiffraction source)
**Status:** Approved design, pre-implementation
**Baseline:** public `NSLS2/PowderLine` at `46ffe4a` (v0.1.0). Note: origin/main
is 2 commits ahead (`827124f`, mp-api compatibility fix — unrelated to engines);
rebase the feature branch onto origin/main before opening a PR.

## Goal

Add [easydiffraction](https://github.com/easyscience/diffraction-lib) (the
EasyScience diffraction library, BSD-3-Clause, pure-PyPI install) as a third
refinement engine behind `powderline.run()`, following the exact pattern the
TOPAS engine established: **zero schema changes, zero kicker.py changes**, one
new self-contained subpackage, one new branch in the engine dispatcher.

Guiding constraint from the maintainer: *do not change what works — extend.*
The existing GSAS-II and TOPAS paths must be byte-identical in behavior, and
the default pixi environment (and its lockfile) must be untouched.

## User-facing behavior

```python
import powderline
result = powderline.run(recipe, output_dir, engine="easydiffraction")
```

- The recipe is an unmodified `GSASII_Rietveld` recipe (`schema_version`
  0.25.x). No new `schema_name`, no payload changes. The engine *translates*
  the recipe, exactly as `engine="topas"` does.
- The return value is the locked standardized dict (see "Result contract").
- Standardized report files (`refined_parameters.csv`, `fit_profile.txt`,
  `<phase>_unit_cell_report.csv`) are written to `output_dir` with the same
  bare names the other engines use.
- `GSASII_SPF` recipes are rejected loudly (v1 has no single-peak-fit analog).
- Missing library: the dispatcher's lazy `import powderline.easydiff.engine`
  raises `ImportError` — same behavior as `engine="gsasii"` on a box without
  GSAS-II — but with an actionable message naming the remedy
  (`pixi run -e easydiff ...`). Once the import succeeds, **no failure
  crashes**: translation and runtime errors return `success=False` with a
  clear `error` string (TOPAS-adapter contract).

## Recipe → easydiffraction mapping (v1)

Verified against `schema.py`, `examples/example_LaB6/input.json`, and
easydiffraction's integration tests / source.

| Recipe (actual payload fields) | easydiffraction API |
|---|---|
| `phases[name].structure` — explicit dict: `phase_name`, `space_group` (H-M), `unit_cell` (a…γ), `atoms{label: x,y,z,occupancy,Uiso,element}` | `StructureFactory.from_scratch(name=…)`; `model.space_group.name_h_m`; `model.cell.length_a…angle_gamma`; `model.atom_sites.create(id, type_symbol, fract_x/y/z, occupancy, adp_iso)` — mirrors their own LBCO integration test. No CIF round-trip needed. |
| `xrd_data` — `tth`, `Itth`, `Itth_weights` (1/σ²) arrays | crop to `fit_range`, convert weights → σ = 1/√w (rows with w ≤ 0 dropped with a warning), write temp `.xye` → `ExperimentFactory.from_data_path(...)` with X-ray CW settings |
| `fit_range: [min, max]` | applied by cropping the arrays before the experiment is built (simplest; excluded-region support can come later) |
| `instrument.initialization[0]` (GSAS-II Iparm1: `Lam`, `Zero`, `U V W X Y Z`, `SH/L`, `Polariz.`, …) | `Lam` → `expt.instrument.setup_wavelength`; `Zero` → `expt.instrument.calib_twotheta_offset`; `U/V/W` → `expt.peak.broad_gauss_u/v/w`; `X/Y` → `expt.peak.broad_lorentz_x/y` — every numeric transfer goes through `conversions.py` (units and sign conventions pinned by tests) |
| `instrument.parameterization.broadening` + `corrections.zero_shift` 4-tuples | `.free = True` on the mapped parameter |
| `phases[name].parameterization.unit_cell` flags | `model.cell.length_*.free = True`; rely on the library's space-group constraint handling (Wyckoff-aware) to keep symmetry-equivalent parameters tied — verified in spike |
| `background.chebyshev` — `coefficients`, `num_coefficients`, `refine_flag` | easydiffraction's native **Chebyshev background** (`PolynomialTerm(order, coef)`), one term per coefficient; all coefs freed when `refine_flag` is true. Direct model-to-model map — no line-segment approximation. |
| scale (per phase) | `expt.linked_structures[name].scale` (+ `.free`) |
| 4-tuple `[value, refine_flag, min, max]` bounds | min/max → the easydiffraction `Parameter` bounds so lmfit respects them; `value=null` means "keep the initialization value" |
| `refinement_controls.refinement_cycles` | GSAS-II counts least-squares cycles; lmfit runs to convergence. Map to the minimizer's max-iteration control if exposed, else ignore **with a recorded warning** (spike decides which) |
| Refinement | `project.analysis.minimizer.type = 'lmfit'`; single `project.analysis.fit()` (schema 0.25 is single-pass by design) |
| Fit statistics | `project.analysis.fit_results`; Rwp/Rexp computed directly from obs/calc/σ arrays if the library exposes only reduced χ² |

### Unsupported-parameter policy (the honesty rule)

An unmappable recipe feature is handled by *refine-flag severity*, matching
the TOPAS writer's warnings mechanism:

- **Present but fixed** (`refine_flag=False`) and not representable —
  `SH/L` (FCJ axial divergence), Gaussian `Z`, `Polariz.`, `Azimuth` — the
  engine **omits it and records a warning** (surfaced via `verbose` and in
  the result's `error=None` path as a `warnings` list inside the engine
  logs). Known consequence, documented in the engine docstring: without FCJ
  asymmetry, low-angle peak shapes differ from GSAS-II, so Rwp values are
  not directly comparable between engines. That is acceptable for v1.
- **Flagged for refinement** (`refine_flag=True`) and not representable —
  raise `EasyDiffractionTranslationError` (loud, actionable message naming
  the offending parameter). Never silently ignore something the user asked
  to refine.

Applies identically to the explicit v1 rejections: atom-level flags
(coordinates, occupancies, Uiso/aniso ADPs — the library supports these, so
they are the natural v2), SPF recipes, and two-wavelength lab data
(`Lam1`/`Lam2` doublets — v1 handles single-`Lam` synchrotron data only).

Simulation mode follows the TOPAS precedent: zero refined parameters is a
legitimate success (`method="easydiffraction_simulation"`) returning the
calculated profile with an empty (but correctly typed) `refined_parameters`
table.

## Architecture

```
src/powderline/
  engine.py                  # +1 branch: "easydiffraction" in _ENGINES, lazy import
  easydiff/                  # NEW subpackage — everything else lives here
    __init__.py
    engine.py                # run_easydiffraction_recipe() → locked result dict
    builder.py               # recipe dict → easydiffraction Project + free-parameter manifest
    conversions.py           # GSAS-II ↔ easydiffraction units/conventions
    errors.py                # EasyDiffractionTranslationError
    kicker.py                # optional thin CLI (parity with topas-kicker), step 8
```

- `powderline/engine.py`: add `"easydiffraction"` to `_ENGINES` and a lazy
  `from powderline.easydiff.engine import run_easydiffraction_recipe` branch.
  Signature mirrors the others: `(recipe, output_dir, *, verbose,
  validate_only)`. `execution_mode` and `topas_*` kwargs are ignored by this
  engine (documented, as TOPAS ignores `execution_mode`).
- `builder.py` is pure translation (no fitting): it returns a configured
  `Project` plus a manifest mapping each freed easydiffraction parameter back
  to the recipe's 9-column `refined_parameters` naming (`parameter_name`,
  `descriptive_name`, `phase_name`, `phase_idx`, `atom_name`, `atom_idx`,
  `value`, `esd`, `category`), plus the accumulated warnings list.
- `engine.py` composes: build → fit (or skip for simulation) → extract →
  write standardized outputs → assemble result dict. `validate_only=True`
  runs the builder (raising `EasyDiffractionTranslationError` on unsupported
  features) and returns the slim summary shape used by the other engines
  (no `run_id`, `simulation_mode` flag included — see `test_api.py`).
- `conversions.py` isolates every unit/convention decision, each pinned by a
  test against a known-good cross-check. Known open conversions for the
  spike: GSAS-II U/V/W units (centidegrees²) vs easydiffraction's Caglioti
  units; `Zero` sign convention vs `calib_twotheta_offset`; recipe `Uiso`
  (Å², U-convention) vs `adp_iso` (U or B = 8π²U?) — needed for v2 but the
  *convention* must be pinned now because structures are built with fixed
  ADP values in v1.

## Result contract

Identical to the shape locked by `tests/test_api.py`:
`success`, `run_id` (canonical UUID4, unique per run), `rwp`, `r_exp`, `gof`,
`elapsed_time`, `method` (`"easydiffraction"` / `"easydiffraction_simulation"`),
`output_files`, `fit_profile` (DataFrame: two_theta, y_obs, y_weights, y_calc,
y_diff, y_bkg, q_values, d_spacings — q/d computed from `Lam`),
`unit_cell_data` / `peak_list_data` (`{phase: DataFrame}`),
`refined_parameters` (9-column DataFrame with ESDs from the lmfit fit),
`spf_peaks` / `spf_convergence_diagnostics` (empty DataFrames), `error`,
`traceback`. Failures return `success=False` with a clear `error` string —
never a crash (same contract as the TOPAS adapter).

`peak_list_data` ships empty in v1 unless the API spike shows easydiffraction
exposes hkl reflection lists cleanly; if it does, populate them.

## Environment (pixi)

New optional feature + environment; **default environment untouched**:

```toml
[feature.easydiff.dependencies]
python = ">=3.12"            # easydiffraction requires ≥3.12; default env stays >=3.10

[feature.easydiff.pypi-dependencies]
easydiffraction = "*"

[environments]
easydiff = ["easydiff"]      # + default feature if GSAS-II solves on ≥3.12
```

During implementation, first try an `easydiff` environment that includes the
default (GSAS-II) feature on Python ≥3.12; if the solve fails, `easydiff`
becomes a standalone environment (easydiffraction-only), which works because
the dispatcher imports engines lazily and `powderline.schema` is
GSAS-II-free. Either way the default environment's resolution in `pixi.lock`
must not change (verify with `git diff pixi.lock` scoped to the default env).

## Testing

- **Unit** (`tests/test_easydiff_builder.py`, `tests/test_easydiff_conversions.py`):
  recipe→object mapping (structure, background terms, bounds, fit_range
  cropping, weight→σ conversion incl. w ≤ 0 rows), free-parameter manifest,
  unit conversions, warning-vs-error policy (fixed-unsupported warns,
  refined-unsupported raises). Use `pytest.importorskip("easydiffraction")`
  so the default-env suite passes unchanged without the library.
- **Engine/API** (`tests/test_easydiff_engine.py`): result-dict shape parity
  (mirroring `test_api.py` assertions incl. unique UUID4 `run_id` and the
  `validate_only` slim shape), simulation mode, missing-library ImportError
  message.
- **Integration**: end-to-end refinement of the LaB6 example
  (`examples/example_LaB6/input.json`) asserting `success=True`, finite
  `rwp` below a sanity threshold, correctly-shaped tables, refined lattice
  parameter within a physical tolerance of the GSAS-II reference value
  (loose — profile models differ), and report files present.
- The entire existing test suite must pass unmodified in the default
  environment (no test file outside `tests/test_easydiff_*.py` changes,
  except adding easydiffraction cases to `test_api.py`'s shape checks).

## Implementation order

1. **API spike (throwaway script):** pin the exact easydiffraction calls —
   `ExperimentFactory` arguments for X-ray/CW (probe/beam-mode enums), whether
   `atom_sites.create` requires `wyckoff_letter` (their test passes it),
   Chebyshev background attachment for pd-cwl experiments, what `fit_results`
   exposes (Rwp? per-parameter ESDs? calculated pattern array?), parameter
   bounds API, minimizer iteration control, hkl list availability, and the
   three unit-convention questions in `conversions.py` above.
2. `conversions.py` + tests (TDD).
3. `builder.py` + tests.
4. `easydiff/engine.py` + result-contract tests.
5. Dispatcher branch + `_ENGINES` + tests.
6. pixi feature/environment + lockfile check.
7. Integration test + docs section in README / `docs/integration-guide.md`.
8. *(Optional, if time permits)* thin `powderline.easydiff.kicker` CLI + pixi
   task `easydiff-kicker` for parity with `topas-kicker`.

## Out of scope (v1) — recorded for later

- Atom-level refinement flags (library supports; v2 candidate — the `adp_iso`
  convention question is already settled by then via `conversions.py`).
- FCJ/axial-divergence asymmetry (`SH/L`) — revisit if easydiffraction grows
  an asymmetry model; until then low-angle profile mismatch is a documented
  v1 limitation.
- Two-wavelength (Kα₁/Kα₂) lab data; SPF analog; neutron CW/TOF (library
  supports TOF via CrysFML); joint fits.
- PDF refinement — easydiffraction bundles `diffpy.pdffit2` and has a PDF
  workflow; a natural future PowderLine capability.
- Persistent-server mode (easydiffraction imports fast; not needed).
- Extracting the GSAS-II monolith from `kicker.py` into its own
  `powderline/gsasii/` subpackage — do this when engine #4 lands, not now.

See `docs/engine-survey.md` for the survey of further candidate engines
(BGMN, FullProf, MAUD/MILK, …) and the reasoning that selected
easydiffraction first.
