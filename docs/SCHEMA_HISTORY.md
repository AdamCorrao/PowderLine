# Schema Evolution History

This document provides a concise summary of PowderLine's schema evolution for context. For detailed commit history, see the repository's git log. Current schema: **0.26.0**

---

## Schema 0.26.0: Per-Parameter Refinement Flags Honored (Current)

**Status**: Active (Q3 2026)

**Key Changes**:
- **Per-parameter refine flags are now honored** *(behavioral change, no shape change)* for the
  three parameter classes GSAS-II exposes only as lumped flags:
  - **Unit-cell parameters** (`a, b, c, alpha, beta, gamma`) — previously ANY true flag refined
    the whole symmetry-allowed cell; false flags on other cell parameters were silently ignored.
  - **Atomic coordinates** (`x, y, z`) — previously any true flag refined all three together.
  - **Anisotropic displacement components** (`U11..U23`) — previously any true flag refined all
    symmetry-allowed components together.
  A parameter now refines iff it is present with `refine_flag=true`; **absent or false means
  fixed**. Internally, PowderLine emits GSAS-II "Hold" constraints for the not-refined degrees
  of freedom (new module `powderline.constraints`).
- **Symmetry-linked parameters refine together (group-OR)**: parameters coupled by the phase's
  Laue class or site symmetry (e.g. cubic `a=b=c`; monoclinic `a`/`c`/`beta`; `x=y` on an
  `(x,x,z)` site) form one degree-of-freedom group that refines if any member is requested.
  Flags on symmetry-fixed parameters (e.g. cubic angles) have no effect. PowderLine does not
  validate flag/symmetry consistency — producing consistent flags is the recipe author's (or an
  upstream recipe builder's) responsibility (see KI-01/KI-02 in
  `docs/known_issues.md`).
- **Partial parameterization is equivalent to explicit flags**: listing only the parameters you
  wish to refine (absence = fixed) is equivalent to listing all with explicit true/false flags.
  Shipped examples use the explicit style for clarity.
- **Fix**: a phase `parameterization` that omits the `peak_broadening` section no longer raises
  (`set_phase_parameterization` previously called `.get` on the `None` that `model_dump` emits
  for an absent section); minimal recipes now run.

**Rationale**:
- The recipe is the point of truth: a `refine_flag: false` that is silently overridden (e.g.
  a hexagonal example's `c` refined despite `c: false` in schema 0.25) misrepresents what the
  refinement actually did. Schema 0.26 makes the flags mean what they say.

**Migration**: Update `"schema_version"` from `"0.25.4"` to `"0.26.0"` in all recipe JSON files,
then **review your refine flags**: any parameter previously piggy-backing on a lumped flag (e.g.
cubic `b`, `c` marked false next to `a: true`, or `y`/`z` false next to `x: true`) is now
genuinely held. Set every parameter you want refined to `true` explicitly (symmetry-linked
partners refine together regardless, but explicit flags keep the recipe honest).

**Breaking Changes**:
- No changes to recipe format.
- **Refinement behavior changes** for recipes whose per-parameter flags disagree within a
  symmetry-linked group or across a formerly-lumped class: previously-ignored false flags are
  now honored, so fewer parameters may vary than before.

**`xrd_data` validation hardening** *(within 0.26.0 — no version bump)*: `XRDDataModel` now
**rejects** ill-formed data instead of passing it through to GSAS-II. It requires: non-empty,
equal-length arrays; all-finite `tth`/`Itth`/`Itth_weights` (no NaN/inf — catches, e.g., a
weight of `1/esd²` with `esd=0`); strictly increasing `tth`; and weights `>= 0` with at least
one `> 0`. Negative **intensities** remain valid (background-subtracted data legitimately dips
below zero — `Itth` is checked for finiteness only, not sign). This is a validation tightening,
not a shape or `kicker.py` output change, so the version is unchanged and previously-valid
recipes are unaffected (the empty-array `example_template` skeleton is intentionally not a
runnable recipe). Rationale: PowderLine is strict and never repairs `xrd_data`; transforming
raw data (unit conversion, weight derivation, negative-intensity handling) is the job of the
file reader/parser that builds the block.

---

## Schema 0.25.4: Consistent Failure Metadata

**Status**: Superseded by 0.26.0

**Key Changes**:
- **Consistent `traceback` in all `run_refinement()` failure exits** *(behavioral change)*: failure
  exits 1 and 3 (histogram loading error, executor exception) now capture and return the Python
  traceback string under a `'traceback'` key. Exit 2 (executor returned `success=False`) forwards
  the executor's traceback if present, or `None`. Previously only the outermost unexpected-exception
  exit (exit 4) included a traceback.
- **`run()` normalizes `traceback` key**: `run()` now calls `result.setdefault('traceback', None)` so
  callers can always key into `result['traceback']` without a `KeyError`, regardless of which failure
  exit triggered.

**Rationale**:
- Consistent `traceback` across all failure exits simplifies debugging without requiring
  callers to handle multiple return shapes

**Migration**: Update `"schema_version"` from `"0.25.3"` to `"0.25.4"` in all recipe JSON files.
No other recipe changes required.

**Breaking Changes**:
- No changes to recipe format

---

## Schema 0.25.3: Refinement Failure Detection and Windows Support

**Status**: Superseded by 0.25.4

**Key Changes**:
- **Robust refinement failure detection** *(behavioral change)*: `execute_rietveld_refinement` and `execute_spf_refinement` now return `success=False` when the refinement produces no Rwp (convergence failure), instead of silently returning `success=True`. All `success=False` paths in `run_refinement()` now consistently include `'rwp': None` and forward the executor's error message rather than a generic fallback string.
- **Windows (win-64) platform support**: `pixi.toml` now targets both `linux-64` and `win-64`. Includes `flang`/`flang-rt_win-64` Fortran toolchain, an `ar.exe` wrapper (`scripts/ar_wrapper.c`) for MSVC archive compatibility, `scripts/win_activate.bat`. Shell scripts enforce LF line endings via `.gitattributes`.

**Rationale**:
- A refinement that produces no Rwp has not converged; returning `success=True` was misleading and caused silent data quality issues downstream
- Windows support is required for deployment on user workstations at partner institutions

**Migration**: Update `"schema_version"` from `"0.25.2"` to `"0.25.3"` in all recipe JSON files. No other recipe changes required.

**Breaking Changes**:
- No changes to recipe format
- Code that expected `success=True` when a refinement produced no Rwp must now handle `success=False`

---

## Schema 0.25.2: Programmatic Python API

**Status**: Superseded by 0.25.3

**Key Changes**:
- **`powderline.run()`**: Primary programmatic entry point. Accepts `dict | RecipeModel`, `output_dir: Path`, and `execution_mode` parameter (`'auto'` | `'server'` | `'subprocess'`). Eliminates the need for callers to manage raw JSON primitives or re-read output files.
- **`powderline.validate()`**: Standalone schema validation without GSAS-II execution. Returns a `RecipeModel` or raises `pydantic.ValidationError`. Enables fast CI-friendly recipe linting.
- **`run_id`**: UUID4 attached to every execution path. Enables log correlation and result tracing across concurrent or sequential runs. Not present when `validate_only=True`.
- **Structured DataFrame returns**: `run()` returns `pd.DataFrame` objects for all tabular outputs (`fit_profile`, `unit_cell_data`, `peak_list_data`, `refined_parameters`, `spf_peaks`, `spf_convergence_diagnostics`). Callers never receive raw dicts or lists. This eliminates column-index bugs that arose when callers parsed the 15-column reflection list by position.
- **DataFrame normalization boundary**: `run_refinement()` continues to return JSON-serializable primitives (required for HTTP transport). `run()` converts them to DataFrames at the public API boundary — keeping the server/client architecture intact.
- **`GSASClient` input polymorphism**: `submit_simulation()` now accepts `Path | dict | RecipeModel`. Previously only `Path` was accepted, requiring callers to serialize models back to disk before submitting.
- **`GSASClient` auto-start + retry**: Automatic server start on first call; 3-attempt exponential backoff (0.5 s → 1.0 s → 2.0 s) on `ConnectError` only. HTTP 4xx/5xx and timeout errors are not retried.
- **`SimulationResponse` (HTTP)**: FastAPI response model now carries all structured data fields (`fit_profile`, `unit_cell_data`, `peak_list_data`, `refined_parameters`, `spf_peaks`, `spf_convergence_diagnostics`), enabling rich API consumers without a separate file-read step.
- **`run(validate_only=True)`**: Fast validation path returns a slim summary dict (`success`, `schema_name`, `schema_version`, `phases`, `refinement_cycles`, `simulation_mode`) with no `run_id` and no GSAS-II invocation.

**Rationale**:
- Direct `run_refinement()` calls required callers to reconstruct results from files or parse raw primitive dicts; `run()` removes this boilerplate at the correct abstraction level
- Standalone `validate()` makes recipe linting a one-liner in CI pipelines without importing GSAS-II
- `run_id` makes it possible to correlate results, logs, and output files from batch or concurrent runs without relying on timestamps or output-directory names
- Returning DataFrames (not raw dicts) at the public API boundary is the correct place for this transformation: it keeps HTTP serialization clean while giving programmatic callers the right type immediately
- Polymorphic `GSASClient` input removes the most common adoption friction: having to write a `RecipeModel` to a file before calling `submit_simulation()`
- Auto-start + retry makes server mode transparent for new users and removes the need to manually manage server lifecycle in scripts

**Migration**: Update `"schema_version"` from `"0.25.1"` to `"0.25.2"` in all recipe JSON files. No other recipe changes required.

**Breaking Changes**:
- No changes to recipe format
- `run()` callers using the `method=` keyword argument must rename it to `execution_mode=`
- `GSASClient.submit_simulation()` callers passing `recipe` as a positional `Path` continue to work unchanged; passing as `dict` or `RecipeModel` is now also valid

**What NOT to do**:
- Do not add DataFrame conversion inside `run_refinement()` — it must remain JSON-serializable for HTTP transport
- Do not bypass `run()` for new programmatic callers; `run_refinement()` is an internal execution engine, not a public API
- Do not add retry logic for HTTP 4xx/5xx errors — these indicate recipe or server bugs that should surface immediately

---

## Schema 0.25.1: Refined Parameters Export + Extended SPF Returns

**Status**: Superseded by 0.25.2

**Key Changes**:
- **New output**: `refined_parameters.csv` with 9 columns including phase/atom associations and ESDs
- **Updated unit cell reports**: Now include `esd` column (3 total columns instead of 2)
- **Per-column float formatting**: Values use `%.6e`, ESDs use `%.8e` for appropriate precision
- **SPF DataFrames**: `powderline.run()` now returns `spf_peaks` and `spf_convergence_diagnostics` as pandas DataFrames; previously only written to files
- **Schema validator**: `GSASII_SPF` schema now validates `payload.refinement_controls.single_peak_fitting_mode` at parse time, not at runtime
- **`run_refinement()` signature**: `recipe_path` parameter removed (was accepted but never used); `method` default changed from `'direct'` to `'server'`
- **`SimulationResponse`**: Added `spf_peaks` and `spf_convergence_diagnostics` optional fields to FastAPI response model

**Rationale**:
- Comprehensive ESD reporting enables uncertainty quantification for all refined parameters
- Machine-readable CSV format enables automated downstream analysis
- Per-column formatting preserves significant figures for small ESDs while keeping values concise
- SPF DataFrame return makes single-peak-fitting results immediately usable programmatically without re-reading output files
- Schema-level SPF validation gives earlier and clearer error messages

**Migration**: Update `"schema_version"` from `"0.25"` to `"0.25.1"` in all recipe JSON files. No other recipe changes required.

**Breaking Changes**:
- `run_refinement()` callers that passed `recipe_path=` keyword argument must remove it
- `run_refinement()` callers that relied on `method='direct'` default must pass it explicitly (new default: `method='server'`)

---

## Schema 0.25: Simplification

**Status**: Superseded by 0.25.1

**Key Changes**:
- **Payload-based structure**: All recipe fields moved under `"payload"` key
- **Fixed output filenames**: `dummy.gpx`, `dummy.lst`, `<phase>_unit_cell_report.csv` (independent of sample names)
- **Removed metadata fields**: `sample_name`, `recipe_description` (manage externally)
- **Schema identification**: Required `"schema_name"` field (`"GSASII_Rietveld"` or `"GSASII_SPF"`)
- **Permanent removal of multi-strategy system**: Sequential and iterative refinement workflows removed (not planned for future development)
- **Peak broadening model field**: Added `"model"` field to support future extensibility (uniaxial, ellipsoidal)
- **Size/strain renaming**: `"size"` → `"isotropic_size"`, `"strain"` → `"isotropic_strain"`

**Rationale**:
- Payload structure separates schema metadata from recipe content
- Fixed filenames eliminate variability in testing/automation
- Multi-strategy refinement didn't work as intended; workflows now split into two schema_name options
- Sample identity belongs in external metadata (directory names, databases)

**Migration**: For recipes from schema 0.24 or earlier:
1. Add `"schema_name"` field (`"GSASII_Rietveld"` or `"GSASII_SPF"`)
2. Wrap all recipe fields in `"payload"` object
3. Remove `"sample_name"` and `"recipe_description"` fields
4. Update strategy field: remove `"strategy"` from `refinement_controls`, use `schema_name` instead

---

## Schema 0.24: Multi-Strategy Refinement System

**Status**: Deprecated (replaced by schema 0.25)

**Key Features**:
- Four refinement strategies: `peaks_only`, `structural_only`, `sequential`, `iterative`
- Single peak fitting (SPF) integrated with structural refinement
- Advanced convergence diagnostics with aphysical parameter detection
- Optional trajectory/comparison outputs

**Why Removed**: Multi-strategy system proved complex with limited benefit. Schema 0.25 simplified to two workflow types selected by `schema_name` field.

**Breaking Changes**:
- Required `refinement_controls` with `strategy` field
- Moved `refinement_cycles` from top-level to `refinement_controls`
- Renamed Gaussian parameter: `pv_gaussian_sigma` → `pv_gaussian_sigma_sq`
- Moved `use_instrument_profile` into `single_peak_fitting_mode` block

---

## Schema 0.23: Initial Multi-Strategy Exploration

**Status**: Deprecated

**Key Features**:
- Introduced basic strategy concept (peaks vs structural)
- Optional `refinement_controls` block
- Initial single peak fitting integration

**Notes**: Experimental implementation that led to full multi-strategy system in 0.24

---

## Schema 0.22: ADP Field Requirements

**Status**: Deprecated

**Key Change**: Made `ADP` (Atomic Displacement Parameter) field required in both structure and parameterization sections for all atoms.

**Rationale**: Eliminated ambiguity about which displacement parameters were active. Previously optional with complex fallback logic.

**Breaking Change**: Validation fails if `ADP` missing from atom definitions. No default value provided.

**Feature**: Structure and parameterization `ADP` values can differ (e.g., CIF provides `Uaniso`, but refinement uses `Uiso`).

---

## Schema 0.21 and Earlier

**Status**: Historical

Early development versions focused on basic Rietveld refinement capabilities:
- JSON recipe structure establishment
- GSAS-II G2scripts API integration
- Basic parameter setting/refinement
- CIF-based structure loading
- Single-phase refinements

---

## Origin (Schema 0.21–0.24)

PowderLine began as a minimal proof-of-concept to confirm that GSAS-II could be driven
from a JSON file via the G2scripts API. The initial scope established the directory
structure and documentation, a JSON schema derived from 3–5 real examples, JSON loading
and schema validation in `kicker.py`, output written to a directory, at least three
refinement types working end-to-end, and basic tests covering JSON loading and validation.

Features deferred from that initial scope (REST API, structured output parsing,
configuration system, programmatic Python API) were subsequently delivered in schema
0.25.1 and 0.25.2.

---

## Schema Design Philosophy

**Example-Driven Development**: Features are added when real refinement examples need them, not speculatively. Schema evolution is guided by committed examples in `examples/` directory.

**Schema-First Validation**: Pydantic models provide clear error messages before GSAS-II operations. Fail fast with helpful diagnostics.

**Progressive Enhancement**: Uses `ConfigDict(extra='allow')` to permit field additions without breaking existing recipes.

**Separation of Concerns**: Recipe JSON contains only refinement parameters. Sample identity, descriptions, and documentation live in external files (directory names, `DESCRIPTION.md`, databases).

---

## Migration Philosophy

- **Explicit schema versions**: Every recipe declares its `"schema_version"`. The code
  accepts one current version — **0.26.0** — and recipes written for older versions must
  be migrated to it following the per-version notes above.
- **Clean breaks with migration guidance**: Obsolete features are clearly deprecated, and
  each version entry documents the migration steps rather than maintaining indefinite
  compatibility shims.
- **Git history is source of truth**: This document provides high-level context only.
