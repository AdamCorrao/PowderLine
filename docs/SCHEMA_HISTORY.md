# Schema Evolution History

PowderLine recipes are validated against a **core schema** (the shared top level
and structural models, owned by PowderLine) plus one **engine schema** per
gateway (`gsasii`, `topas`, `easydiffraction`), each in its engine's native
conventions. Every schema has its own version. A recipe states both
(`core_schema_version`, `engine_schema_version`), and each PowderLine release
**declares** which versions it accepts. Compatibility is never inferred from the
version numbers (`powderline.support_matrix()` lists the declarations).

> **Refactor in progress (v0.2.0).** The schemas below are being introduced on
> the `refactor/multi-engine` branch. Until the 0.26.0 schema is removed (re/07),
> `powderline.run()` / `validate()` still validate recipes against the
> **unified 0.26.0 schema** documented in the archive at the bottom of this page.

---

## Core schema

### 1.0.0 — introduced in PowderLine 0.2.0 (in development)

Defined in `src/powderline/schema_core.py`. Not used by any gateway yet: the
engine schemas adopt it in re/04–06.

> **Pre-release revision (re/03b, not a new version).** Before its first
> release, core 1.0.0 replaced its structure-only phase model (`PhaseStructure`
> with plain values, plus `null`-valued "structure" parameters) with the
> **phase block** described below: one block per phase, every structural
> quantity a refinable parameter, flags and bounds checked against the symmetry,
> and `space_group` in gemmi's canonical spelling. No recipe was ever written
> against the earlier draft.

- **Recipe frame** (`CoreRecipe`): `schema_name` (`"<engine>.<workflow>"`),
  `core_schema_version`, `engine_schema_version`, `metadata`, `payload` (typed
  by the engine schema). Unknown keys are errors at every level.
- **`metadata`**: free-form and never interpreted; must be JSON-serializable and
  at most **1 MiB** as UTF-8 JSON. The cap is the constant
  `schema_core.METADATA_MAX_BYTES`; change it there and record the change here.
- **Parameters**: `[value, refine_flag]` (no bounds) or
  `[value, refine_flag, min, max]` (bounds the engine honors; `min`/`max` may
  be `null` = open). A bounded list where bounds aren't supported is an error.
  `value` is always a number and `refine_flag` a JSON boolean; neither is ever
  `null`.
- **Data and ranges**: `xrd_data` (2θ in degrees, weights 1/σ², validated as in
  0.26.0); `fit_range` `[min, max]`, with `max > min`, inside the data's 2θ range.
- **Background**: Chebyshev (`num_coefficients`, `coefficients`, `refine_flag`).
  Coefficient semantics are documented per engine.
- **Units** are fixed per field in the schema, never written in a recipe.
- **Numbers** are JSON numbers: a quoted number (`"0.25"`) or a boolean is an
  error, never converted. Integer fields (`Multiplicity`, `num_coefficients`)
  take a whole number (`4` or `4.0`), not `4.5`.
- **Accepted core versions** (this release): `==1.0.0`.

- **Phase block** (`Phase`): one block per phase, keyed by the phase name in
  the engine payload's `phases` (no `phase_name` field). Core owns
  `space_group`, `unit_cell` and `atoms`; each engine schema adds its own phase
  fields next to them (e.g. gsasii `scale`, `peak_broadening`) and may not
  redefine core's. Every structural quantity is a parameter (`[value, flag]`,
  or `[value, flag, min, max]` in engines with bounds): cell `a`–`gamma`; atom
  `x`, `y`, `z`, `occupancy`, `Uiso` or `U11`…`U23`. `element`, `ADP`,
  `Multiplicity` and `space_group` are plain values. Nothing is `null` and
  nothing has a default: `occupancy` must be stated; `Multiplicity` (derived,
  so optional) is left out when not stated, never `null`.
  **Names:** atom labels (the `atoms` keys) and phase names (the engine
  payload's `phases` keys) start with a letter and contain only ASCII letters,
  digits and `_` (`O1`, `LaB6_a`); phase names must also differ by more than
  case. This is the form every engine can use as-is: GSAS-II renames a phase
  with surrounding spaces or non-ASCII characters (and its settings were then
  skipped), easydiffraction accepts no other atom label, TOPAS writes the names
  into its input file, phase names become report file names (which ignore case
  on Windows and macOS), and problems are reported at `.`-joined paths
  (`atoms.O1.Uiso`). So `"2H-MoS2"` is written e.g. `MoS2_2H`.
  Structural interpretation is checked once, in core, so every engine gets the
  same structure. **Validation never changes what a recipe says:** the recipe
  is the record of the refinement intent. A value that has an exact decimal
  form must be written exactly (`0.5`, not `0.4999999`); otherwise validation
  fails and the message gives the value to write. A value with no exact
  decimal form (a third: 1/3, 1/6, 2/3, or a tie offset such as `x + 1/3`)
  cannot be written exactly, so **6 decimals** state it (`0.333333` means 1/3)
  and every engine receives the exact value. PowderLine never rewrites a
  recipe file; only a Python dump of a validated model shows the exact value's
  full spelling (`0.3333333333333333`), which is the same value. All problems
  are reported together, each at its own field
  (`unit_cell`, `atoms.<label>` for the position, `atoms.<label>.Multiplicity`,
  `atoms.<label>.Uaniso`, or the parameter a flag/bound rule concerns, e.g.
  `unit_cell.b`, `atoms.<label>.y`, `atoms.<label>.Uaniso.U12`); they are checked
  once the individual fields are valid:
  - **Space group**: gemmi's canonical extended Hermann–Mauguin name, exactly:
    `"P m -3 m"`, `"C 1 2/m 1"`, `"P 1 21/c 1"`, `"R -3 m:H"`, `"F d -3 m:2"`.
    Every setting has exactly one such name, and it states the setting
    (two-origin groups `:1`/`:2`, rhombohedral groups `:H`/`:R`, the monoclinic
    unique axis). Any other spelling (`"Pm-3m"`, `"C2/m"`, `"p m -3 m"`) is an
    error whose message gives the canonical name; a symbol without a setting
    (`"R -3 m"`, `"F d -3 m"`) lists both choices, and a short monoclinic
    symbol (`"P21/c"`, `"C2/m"`) lists every setting of that space group by
    unique axis, because it states neither the unique axis nor the cell choice.
    Each engine translates the name to its own convention, and may accept fewer
    settings. The accepted names are those of gemmi 0.7.5, which PowderLine pins
    exactly; the list is `tests/data/canonical_space_groups.txt`, and any change
    to it is recorded here.
  - **Unit cell**: `a`, `b`, `c` (Å, > 0), `alpha`, `beta`, `gamma` (degrees,
    0–180). In engines with bounds, a stated bound must lie in the same range
    (a length's `min` > 0; an angle's `min`/`max` strictly between 0 and 180);
    `null` means no bound. No `volume` (every engine derives it). The cell must fit the
    space group **exactly** (to floating-point precision): e.g. cubic
    `a = b = c` and all angles 90°. A mismatch is an error naming each
    parameter and its symmetric value. (GSAS-II alone would silently apply the
    symmetry only when the cell is refined.)
  - **Elements**: a bare element symbol spelled exactly (`"Fe"`, not `"FE"`).
    Charged scattering types (`"Fe3+"`) are not supported yet and are rejected,
    never reduced to the neutral atom.
  - **Occupancy**: 0 to 1 inclusive, even where an engine would accept more.
  - **Special positions:** a coordinate whose exact value has a decimal form
    is written **exactly**: `0.5`, `0.25`, `0.125`, and coupled coordinates
    satisfying their relation (`(x, 2x, 1/4)` with `y` exactly `2·x`). A
    coordinate whose exact value has none, i.e. involves a third (`1/3`, `2/3`,
    `1/6`, `1/12`, or `y = x + 1/3` on some rhombohedral sites), is written to
    **at least 6 decimals** (`0.333333`, `0.166667`) and read as the exact value.
    Within 5e-6 of a special position, anything else is an **error** whose
    message gives the values to write (each tie group's first coordinate is kept
    as stated and the others derived, e.g. `write y = 0.2468`, `write z = 0.25`).
    From 5e-6 to 2e-3 off a special position is an error too: state the
    position (as above) or move the atom off it (`0.33`, `0.3333`).
  - **Multiplicity**: optional; when stated it must equal the multiplicity
    derived from the space group.
  - **ADPs**: `ADP` selects the thermal parameter that is **required**:
    `"Uiso"` requires `Uiso` (Å²); `"Uaniso"` requires `Uaniso` with all six of
    `U11 U22 U33 U12 U13 U23` (Å²). The other one must be left out. A missing
    or extra one is an error at that field (`atoms.<label>.Uiso`,
    `atoms.<label>.Uaniso.U23`), and the JSON Schema states the same rule.
    Anisotropic ADPs must respect the site symmetry exactly; otherwise it is an
    error, which within 1e-6 Å² gives the values to write (derived from each tie
    group's first component, e.g. `write U12 = 0.0061735` for `U12 = U22/2`).
    A `Uaniso` that is not positive definite (a principal mean-square
    displacement ≤ 0, so no thermal ellipsoid exists) is **accepted with a
    structured warning** (`uaniso_not_positive_definite`), like a negative
    `Uiso`: it can show that the parameter compensates for something the model
    lacks (e.g. absorption). Engines carry it as given.
  - **Refinement intent follows the symmetry.** Parameters tied by symmetry
    are **one** parameter, and every member is stated: cubic `a, b, c`;
    tetragonal/hexagonal `a, b`; rhombohedral axes (`:R`) `a, b, c` and
    `alpha, beta, gamma`; coordinates such as `(x, 2x, 1/4)` or
    `(x, x + 1/2, z)`; Uij such as `U11 = U22`, `U12 = U22/2`, `U13 = -U23`.
    - Tied members carry the **same refine flag**; the error is reported at
      every member whose flag differs from the first member's.
    - A parameter **fixed** by symmetry (a cubic angle, `x` of an atom at the
      origin, `U12` on a mirror site) has refine flag `false` and, where the
      engine has bounds, no bounds (`null`, `null`).
    - Different groups are independent (orthorhombic `a` refined, `b` fixed).
    - **Bounds** (engines with bounds) follow the tie like the values:
      `y = k·x + c` maps x's `[min, max]` to `[k·min + c, k·max + c]`
      (swapped for negative k; an open side stays open), e.g. `(x, x + 1/2, z)`
      with x in `[0, 0.2]` needs y in `[0.5, 0.7]`. They must be the mapped
      values exactly, or to at least 6 decimals where the tie's offset is a
      third (`y = x + 1/3`); otherwise the error gives them.
    - Atoms at the same position are independent: their flags are never tied.

## gsasii engine schema

### 1.0.0 — introduced in PowderLine 0.2.0

Entries added by re/04 (`gsasii.rietveld`, `gsasii.spf`).

## topas engine schema

### 1.0.0 — introduced in PowderLine 0.2.0

Entries added by re/05 (`topas.rietveld`, `topas.spf`).

## easydiffraction engine schema

### 1.0.0 — introduced in PowderLine 0.2.0

Entries added by re/06 (`easydiffraction.rietveld`).

---

## Archive: unified-schema era (0.21–0.26)

The single schema shared by all engines up to PowderLine 0.1.1, kept verbatim
(headings moved one level down).

This document provides a concise summary of PowderLine's schema evolution for context. For detailed commit history, see the repository's git log. Current schema: **0.26.0**

---

### Schema 0.26.0: Per-Parameter Refinement Flags Honored (Current)

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

### Schema 0.25.4: Consistent Failure Metadata

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

### Schema 0.25.3: Refinement Failure Detection and Windows Support

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

### Schema 0.25.2: Programmatic Python API

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

### Schema 0.25.1: Refined Parameters Export + Extended SPF Returns

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

### Schema 0.25: Simplification

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

### Schema 0.24: Multi-Strategy Refinement System

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

### Schema 0.23: Initial Multi-Strategy Exploration

**Status**: Deprecated

**Key Features**:
- Introduced basic strategy concept (peaks vs structural)
- Optional `refinement_controls` block
- Initial single peak fitting integration

**Notes**: Experimental implementation that led to full multi-strategy system in 0.24

---

### Schema 0.22: ADP Field Requirements

**Status**: Deprecated

**Key Change**: Made `ADP` (Atomic Displacement Parameter) field required in both structure and parameterization sections for all atoms.

**Rationale**: Eliminated ambiguity about which displacement parameters were active. Previously optional with complex fallback logic.

**Breaking Change**: Validation fails if `ADP` missing from atom definitions. No default value provided.

**Feature**: Structure and parameterization `ADP` values can differ (e.g., CIF provides `Uaniso`, but refinement uses `Uiso`).

---

### Schema 0.21 and Earlier

**Status**: Historical

Early development versions focused on basic Rietveld refinement capabilities:
- JSON recipe structure establishment
- GSAS-II G2scripts API integration
- Basic parameter setting/refinement
- CIF-based structure loading
- Single-phase refinements

---

### Origin (Schema 0.21–0.24)

PowderLine began as a minimal proof-of-concept to confirm that GSAS-II could be driven
from a JSON file via the G2scripts API. The initial scope established the directory
structure and documentation, a JSON schema derived from 3–5 real examples, JSON loading
and schema validation in `kicker.py`, output written to a directory, at least three
refinement types working end-to-end, and basic tests covering JSON loading and validation.

Features deferred from that initial scope (REST API, structured output parsing,
configuration system, programmatic Python API) were subsequently delivered in schema
0.25.1 and 0.25.2.

---

### Schema Design Philosophy

**Example-Driven Development**: Features are added when real refinement examples need them, not speculatively. Schema evolution is guided by committed examples in `examples/` directory.

**Schema-First Validation**: Pydantic models provide clear error messages before GSAS-II operations. Fail fast with helpful diagnostics.

**Progressive Enhancement**: Uses `ConfigDict(extra='allow')` to permit field additions without breaking existing recipes.

**Separation of Concerns**: Recipe JSON contains only refinement parameters. Sample identity, descriptions, and documentation live in external files (directory names, `DESCRIPTION.md`, databases).

---

### Migration Philosophy

- **Explicit schema versions**: Every recipe declares its `"schema_version"`. The code
  accepts one current version — **0.26.0** — and recipes written for older versions must
  be migrated to it following the per-version notes above.
- **Clean breaks with migration guidance**: Obsolete features are clearly deprecated, and
  each version entry documents the migration steps rather than maintaining indefinite
  compatibility shims.
- **Git history is source of truth**: This document provides high-level context only.
