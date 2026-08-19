# EasyDiffraction Engine for PowderLine — Design

**Date:** 2026-08-19
**Status:** Approved design, pre-implementation
**Baseline:** public `NSLS2/PowderLine` at `46ffe4a` (v0.1.0 initial public release)

## Goal

Add [easydiffraction](https://github.com/easyscience/diffraction-lib) (the
EasyScience diffraction library, BSD-3-Clause, pure-PyPI install) as a third
refinement engine behind `powderline.run()`, following the exact pattern the
TOPAS engine established: **zero schema changes, zero kicker.py changes**, one
new self-contained subpackage, one new branch in the engine dispatcher.

Guiding constraint from the maintainer: *do not change what works — extend.*
The existing GSAS-II and TOPAS paths must be byte-identical in behavior, and
the default pixi environment must be untouched.

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
- If the `easydiffraction` package is not installed, dispatch raises
  `ImportError` with the exact remedy: `pixi run -e easydiff ...` /
  `pixi add --feature easydiff --pypi easydiffraction`.

## v1 scope (MVP Rietveld)

Supported recipe features, mapped onto the easydiffraction API:

| Recipe concept (GSAS-II conventions) | easydiffraction API |
|---|---|
| Phase from CIF dict | `StructureFactory` (from CIF; `from_scratch` + cell/space group/atom_sites as fallback) |
| Measured pattern (arrays in payload) | temp `.xye` file → `ExperimentFactory.from_data_path(...)` with X-ray CW settings |
| Wavelength | `expt.instrument.setup_wavelength` |
| Zero/2θ offset | `expt.instrument.calib_twotheta_offset` |
| Phase scale | `expt.linked_structures[name].scale` |
| Unit cell refinement flag | `model.cell.length_a.free = True` (etc., per crystal system) |
| Caglioti U, V, W | `expt.peak.broad_gauss_u/v/w` (unit conversion in `conversions.py`) |
| Lorentzian X, Y | `expt.peak.broad_lorentz_x/y` |
| Background | `expt.background.create(id=…, position=…, intensity=…)` points; intensities refined when the recipe refines background |
| Refinement | `project.analysis.minimizer.type = 'lmfit'`; `project.analysis.fit()` |
| Fit statistics | `project.analysis.fit_results` (+ direct Rwp computation from obs/calc/weights if the library exposes only reduced χ²) |

Explicitly **rejected in v1** with a loud translation error (mirroring
`TopasTranslationError`): atom-level refinement flags (coordinates,
occupancies, iso/aniso ADPs), SPF recipes, non-CW instrument settings.
The library supports atom-level refinement natively
(`model.atom_sites['La'].adp_iso.free = True`), so that is the natural v2 —
rejecting rather than silently ignoring keeps results honest until then.

Simulation mode (calculate-only recipes) follows the TOPAS precedent: zero
refined parameters is a legitimate success (`method="easydiffraction_simulation"`),
returning the calculated profile with an empty `refined_parameters` table.

## Architecture

```
src/powderline/
  engine.py                  # +1 branch: "easydiffraction" in _ENGINES, lazy import
  easydiff/                  # NEW subpackage — everything else lives here
    __init__.py
    engine.py                # run_easydiffraction_recipe() → locked result dict
    builder.py               # recipe dict → easydiffraction Project/Structure/Experiment
    conversions.py           # GSAS-II ↔ easydiffraction units/conventions
    errors.py                # EasyDiffractionTranslationError
```

- `powderline/engine.py`: add `"easydiffraction"` to `_ENGINES` and a lazy
  `from powderline.easydiff.engine import run_easydiffraction_recipe` branch.
  Signature mirrors the others: `(recipe, output_dir, *, verbose,
  validate_only)`. `execution_mode` and `topas_*` kwargs are ignored by this
  engine (documented, as TOPAS ignores `execution_mode`).
- `builder.py` is pure translation (no fitting): it returns a configured
  `Project` plus a manifest of which easydiffraction parameters were set free
  and how they map back to the recipe's 9-column `refined_parameters` naming
  (`parameter_name`, `descriptive_name`, `phase_name`, `phase_idx`,
  `atom_name`, `atom_idx`, `value`, `esd`, `category`).
- `engine.py` composes: build → fit (or skip for simulation) → extract →
  write standardized outputs → assemble result dict. `validate_only=True`
  runs the builder (raising `EasyDiffractionTranslationError` on unsupported
  features) and returns the slim summary shape used by the other engines.
- `conversions.py` isolates every unit/convention decision (e.g. GSAS-II
  U/V/W in centidegrees² vs easydiffraction's Caglioti units, zero-shift sign
  convention), mirroring `topas/conversions.py`. Each conversion gets a test
  pinned against a known-good cross-check.

## Result contract

Identical to the shape locked by `tests/test_api.py`:
`success`, `run_id`, `rwp`, `r_exp`, `gof`, `elapsed_time`, `method`
(`"easydiffraction"` / `"easydiffraction_simulation"`), `output_files`,
`fit_profile` (DataFrame), `unit_cell_data` / `peak_list_data`
(`{phase: DataFrame}`), `refined_parameters` (9-column DataFrame with ESDs),
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
becomes a standalone environment (easydiffraction-only), which is fine because
the dispatcher imports engines lazily. Either way `pixi.lock` for the default
environment must not change.

## Testing

- **Unit** (`tests/test_easydiff_builder.py`, `tests/test_easydiff_conversions.py`):
  recipe→object mapping, parameter-flag manifest, unit conversions, loud
  rejection of unsupported features. Use `pytest.importorskip("easydiffraction")`
  so the default-env suite passes unchanged without the library.
- **Engine/API** (`tests/test_easydiff_engine.py`): result-dict shape parity
  (mirroring `test_api.py` assertions), simulation mode, missing-library
  ImportError message, `validate_only` summary shape.
- **Integration**: end-to-end refinement of the LaB6 example
  (`tests/data/topas/example_LaB6.xye` + the LaB6 example recipe) asserting
  `success=True`, finite `rwp` below a sanity threshold, correctly-shaped
  tables, and report files present.
- The entire existing test suite must pass unmodified in the default
  environment (no test file outside `tests/test_easydiff_*.py` changes,
  except adding easydiffraction cases to `test_api.py`'s shape checks).

## Implementation order

1. **API spike (throwaway script, committed under `scripts/` only if useful):**
   pin the exact easydiffraction calls for X-ray CW data — `ExperimentFactory`
   arguments for xray/CW, CIF loading path, background API, what
   `fit_results` exposes (Rwp? ESDs per parameter? calculated pattern
   array?), hkl list availability. Everything in the mapping table above is
   evidence-based from their integration tests except these details.
2. `conversions.py` + tests (TDD).
3. `builder.py` + tests.
4. `easydiff/engine.py` + result-contract tests.
5. Dispatcher branch + `_ENGINES` + tests.
6. pixi feature/environment + lockfile check.
7. Integration test + example recipe + README/docs section.

## Out of scope (v1) — recorded for later

- Atom-level refinement flags (library supports; v2 candidate).
- SPF analog, neutron CW/TOF (library supports TOF via CrysFML), joint fits.
- PDF refinement — easydiffraction bundles `diffpy.pdffit2` and has a PDF
  workflow; a natural future PowderLine capability.
- Persistent-server mode (easydiffraction imports fast; not needed).
- Extracting the GSAS-II monolith from `kicker.py` into its own
  `powderline/gsasii/` subpackage — do this when engine #4 lands, not now.

See `docs/engine-survey.md` for the survey of further candidate engines
(BGMN, FullProf, MAUD/MILK, …) and the reasoning that selected
easydiffraction first.
