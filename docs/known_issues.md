# Known issues & documented failure routes

**Repo-wide register.** Curated list of behaviors we have *discovered and
understood* but are (for now) choosing not to prevent, plus quirks a developer
must keep in mind. Each has a stable `KI-NN` id that development docs, plans,
and PR bodies reference (e.g. "pass-through per KI-01"). This is a **development
doc**, not a substitute for GitHub issues — keep entries terse and durable
(breadcrumbs for when an issue recurs or a related one appears), and prune ones
that become obsolete.

**Scope.** Entries describe bugs, tripping hazards, codebase/convention
oddities, and in general issues that need a development branch (refactoring,
chores/cleanup, new features) or a standing decision. **Small transient
issues do not belong here** — e.g. doc-sweep leftovers or other nits found
during review are flagged to the user and fixed *before* the PR is prepared,
not registered (keeps the register and dev history lean).

**Conventions:**
- IDs are permanent and never reused. Add new issues with the next free
  `KI-NN`; do not renumber existing ones.
- **Status vocabulary:** `accepted` = known, intentionally not prevented now;
  `planned-fix` = slated for a named branch; `watch` = quirk to keep in mind, no
  action yet; `implemented` = addressed by a shipped change (kept for history);
  `blocker` = must be resolved before the relevant branch can merge.
- **Severity vocabulary** (impact if encountered, orthogonal to status):
  `critical` = silent wrong/empty output or data loss; `major` = breaks a real
  flow or compounds technical debt; `minor` = works but fragile /
  incorrect-in-edge-cases; `nit` = cleanup. A `nit` is registered only when
  fixing it now is genuinely out of scope — otherwise fix it pre-PR instead of
  filing it.
- Entry headings carry status and severity: `` `status · severity` ``.
- Each entry states: what, evidence (`file:line` or a probe), current decision,
  revisit trigger. Reference related issues by id.
- Reviewers file newly discovered *durable* issues here rather than fixing
  them in the branch under review; transient nits are fixed before the PR.
- Some **Evidence** lines cite the maintainers' internal design records
  (`findings §X`, `probes/*.py`) that are not part of this repository; the
  stated empirical results were verified against the pinned GSAS-II and stand
  on their own.

---

### KI-01 — Malformed fixed cell computes a wrong pattern silently `accepted · critical`

**What.** A metrically symmetry-illegal cell (e.g. cubic phase with `a≠b≠c`) that
is *not* refined is used verbatim: GSAS-II computes peak positions from the raw,
non-symmetric reciprocal metric tensor and returns a "successful" refinement with
no error or warning.

**Evidence.** findings §C.4; `probes/probe1_cell_mismatch.py` (distorted cubic,
cell-refine off → d-spacings follow the distorted metric, Rwp 65.8%, success).

**Decision.** Pass-through, do not prevent (Branch 1). Preventing symmetry-illegal
*cell values* is the future recipe builder's job. This behavior is also arguably
*useful* — e.g. deliberately simulating symmetry-breaking/strain effects with a
fixed distorted cell.

**Revisit.** When a recipe builder exists, or if a cell-vs-symmetry metric
validation option is added (design-options Fork 2/4). If added, it should be
*opt-in / warn* so the legitimate symmetry-breaking-simulation use survives.

---

### KI-02 — Oblique (monoclinic/triclinic) per-parameter cell holds are limited `implemented (0.26), limitation inherent · minor`

**What.** Holding a reciprocal-metric A-term equals holding a *direct* cell
parameter only for orthogonal cells. For monoclinic, only the unique-axis length
is independently holdable; `a`, `c`, and the oblique angle are coupled. For
triclinic, no direct parameter is individually holdable.

**Evidence.** findings §C.6; `probes/probe4_oblique_cell_coupling.py` (hold `A2`
in a monoclinic cell → `c` still moves 6.000→6.032).

**Decision (implemented in schema 0.26).** Branch 1 models the cell as
Laue-class **DOF-groups** (coupled oblique parameters form a single group) and
applies **group-OR**: a group refines if any member is requested, and only
whole unrefined groups are held (`powderline/constraints.py`,
`cell_dof_groups`). PowderLine does **not** validate or reject mixed flags
within a group — it is the engine/translator; symmetry-consistent flags are
the upstream recipe builder's responsibility. Consequence for a *malformed*
recipe: a `false` flag inside a group with a `true` member is not honored
(deterministic, documented here). Not a GSAS-II limitation we can work around:
a direct-parameter hold on an oblique cell is a nonlinear constraint outside
GSAS-II's linear constraint system.

**Revisit.** Transparency (which groups refined together) arrives with the
Fork 9 provenance work. True independent oblique direct-parameter holds would
require a fundamentally different (nonlinear) constraint approach — unlikely to
change.

---

### KI-03 — Site symmetry & multiplicity are derived, and failures are swallowed `planned-fix · major`

**What.** PowderLine derives each atom's site symmetry and multiplicity at load
via `G2spc.SytSym`, inside a bare `except Exception: pass`, defaulting to `""`
and `1` on failure. The recipe never states these, and a derivation failure is
invisible.

**Evidence.** `kicker.py:1170-1176`; findings §A.4.

**Decision.** Candidate for a dedicated branch (design-options Fork 7): make them
explicit, validated recipe fields (7A), with an immediate safety patch to stop
swallowing (7B). Improves resilience and decouples structural interpretation from
GSAS-II for future multi-engine support.

**Revisit.** Branch ordering open (Fork 7 vs general constraints). At minimum,
stop swallowing the exception when the surrounding area is next touched.

---

### KI-04 — Recipe→output provenance is implicit; no round-trip `planned-fix · major`

**What.** Outputs are CSV/txt keyed by GSAS-II parameter names. Which parameters
were *actually* varied vs held-by-user vs held-by-symmetry vs frozen-by-limit is
implicit (inferred from absence in `varyList`). No recipe-shaped output exists, so
sequential/multi-step refinement needs external output→input mapping.

**Evidence.** findings §A.5.

**Decision.** Add explicit per-parameter disposition (design-options Fork 6/9C)
and a JSON recipe-style output enabling round-trip / sequential refinement (Fork
8). Deferred to their own branch(es).

**Revisit.** When designing sequential-refinement support, or when a downstream
data-management consumer needs machine-readable provenance.

---

### KI-05 — GSAS-II refinement failures & constraint warnings don't raise `planned-fix · major`

**What.** `proj.refine()` does not raise on refinement failure (singular matrix,
etc.); constraint cascades/conflicts and frozen-variable notices go to stdout /
`Rvals['msg']` only. PowderLine currently inspects none of these, inferring
success solely from `hist.residuals['wR']` being non-None.

**Evidence.** findings §A.6, §C.5; `SCRIPT:1585-1592`, `STRMAIN:607-641`.

**Decision.** Surface diagnostics via `proj.data` structures + captured console
text (design-options Fork 9C), *without* adopting `do_refinements`. Minimal
failure-detection may land in Branch 1; fuller provenance later.

**Revisit.** Bundle with Fork 6/8 provenance work.

---

### KI-06 — GSAS-II parameter-limit (min/max) semantics are clamp-and-freeze `watch · minor`

**What.** GSAS-II enforces `parmMin/parmMaxDict` not as in-loop box constraints
but by resetting out-of-range variables to the limit *after* a refine and
freezing them from the next one. Frozen state persists in the `.gpx`. Coordinate
limits must target `Ax/Ay/Az` even though the varied variable is `dAx/dAy/dAz`.

**Evidence.** findings §B.4; `STRMAIN:570-571,1196-1228`.

**Decision.** min/max deferred (Branch 2). When implemented, document the
semantics honestly and surface frozen variables; do not imply hard bounds.

**Revisit.** Branch 2 (general constraints & limits).

---

### KI-07 — Space-group setting (hexagonal vs rhombohedral, origin choice) is unvalidated `watch · critical`

**What.** `R -3 c` (hexagonal axes) and `R -3 c R` (rhombohedral axes) are both
accepted but imply different cell parameterizations and A-term equivalence sets.
Two-origin space groups default to origin 2; a recipe with origin-1 coordinates
under such a symbol is silently wrong (no CIF symops exist to trigger GSAS-II's
auto-shift, since PowderLine is file-less).

**Evidence.** findings §D-Q2; `probes` rhombohedral check; `SPC:4195-4209`
(origin table), `CIF:677-731` (CIF-only auto-shift).

**Decision.** Not addressed in Branch 1. Candidate for the convention-validation
work (Fork 4): at minimum echo the inferred Laue class / setting back to the user.
Origin-1 detection is hard without symmetry operators and better handled in the
recipe builder.

**Revisit.** Convention-validation branch, or recipe-builder work.

---

### KI-08 — Sphinx build emits reST warnings from `RecipeModel` docstring `watch · nit`

**What.** A clean `pixi run docs` builds successfully (exit 0) but prints
**~330 warnings** (re-measured 2026-08-17 on the public-release branch), all
pre-existing or cosmetic:
- ~290 autodoc cross-reference warnings (`py:class reference target not
  found`) for pydantic internals — `PlainSerializer`, `BaseModel`,
  `ConfigDict`, and the `Annotated[..., PlainSerializer(lambda ...)]` field
  types in `schema.py` that autodoc can't resolve;
- ~18 docutils WARNING/ERROR lines from the `RecipeModel` docstring, whose
  embedded markdown-style ```json migration examples don't parse as reST
  (unexpected indentation, inline-literal start without end);
- ~19 myst warnings (header-level jumps and slug-style anchor cross-refs)
  from the markdown pages promoted into the Sphinx toctree at the public
  release (`known_issues`, `regression-tolerance`, `cross-platform-guide`);
- one `html_static_path entry '_static' does not exist` warning.

**Evidence.** `src/powderline/schema.py` (`RecipeModel` docstring and the
Annotated/PlainSerializer field types); reproduced by `pixi run docs-clean &&
pixi run docs`. Pre-existing on `main` (none of the warning sources are lines
the 0.26 branch touched; the docstring block exists verbatim at
`main:src/powderline/schema.py`).

**Decision.** Cosmetic; not a build failure. Left as-is. Fix by converting the
docstring's fenced JSON to a reST `.. code-block:: json` (or a raw literal
block), adding the missing `docs/_static/` dir, and suppressing/nitpick-
ignoring the pydantic cross-ref noise. `watch` so a reviewer does not
misattribute these warnings (or their count) to a branch under review.

**Revisit.** Any docs-hardening pass, or if the docs build is made
warnings-as-errors.

---

### KI-09 — Withdrawn

Filed during the PR #23 review; withdrawn 2026-07-16 (the reported behavior
was intentional, not a defect). Id retained so it is never reused.

---

### KI-10 — `stop_server()` force-kill escalation raises on Windows and is untested `implemented (chore/cleanup) · minor`

**What.** `stop_server()` escalates to `os.kill(pid, signal.SIGKILL)` when the
server has not exited 5 s after SIGTERM. `signal.SIGKILL` does not exist on
Windows, so the escalation branch raises `AttributeError` there instead of
force-killing. The branch is rarely reached on Windows (Python maps
`os.kill(pid, SIGTERM)` to `TerminateProcess`, a hard kill, so the graceful
loop usually succeeds), and no test on any platform drives the timeout path —
the unit tests cover only the graceful sequence
(`tests/test_gsas_server_unit.py`, `alive_states = iter([True, False])`).

**Evidence.** `src/powderline/gsas_server.py:371` (found in the chore/cleanup
pre-PR review, 2026-08-11); `tests/test_gsas_server_unit.py:157`.

**Decision.** Fixed in the same pre-PR review session on the user's explicit
instruction (overriding the Stage-3 file-don't-fix default): the escalation now
uses `getattr(signal, "SIGKILL", signal.SIGTERM)` — on Windows,
`os.kill(pid, SIGTERM)` maps to `TerminateProcess`, the correct hard kill — and
unit tests drive the escalation path (`tests/test_gsas_server_unit.py`).

**Revisit.** Closed; kept for history.

---

### KI-11 — Server discovery is split between PID file and port probe; an orphaned server is unmanageable `watch · minor`

**What.** `gsas-server status|stop` trust the PID file
(`<tempdir>/powderline_gsas_server.pid`), while `GSASClient` trusts an HTTP
`/health` probe on the port-file/default port (19471). If the PID file is lost
(crash, tempdir cleanup, or the starting process group being killed) the server
keeps serving: clients silently use it while `status` reports "not running" and
`stop` cannot stop it (there is no HTTP shutdown endpoint). A related latent
bug in the auto-start path — the child env joined `PYTHONPATH` with a
hardcoded `':'` instead of `os.pathsep` (wrong separator on Windows; benign in
the pixi env, where powderline is pip-installed) — was **fixed on
chore/cleanup** at the user's instruction; the discovery/lifecycle mismatch
below remains open.

**Evidence.** Reproduced live during the chore/cleanup pre-PR review
(2026-08-12): `/health` on 19471 answered (`uptime_seconds≈13261`) with no
`.pid`/`.port` files present and `status` reporting "not running".
`src/powderline/gsas_server.py:332-347` (`is_server_running` = PID file),
`src/powderline/gsas_client.py:40-49` (`is_server_available` = port probe),
`gsas_client.py:180` (the former hardcoded `':'`, now `os.pathsep`).

**Decision.** File for a server-lifecycle robustness pass: align `status`/
`stop` with the client's probe (e.g. `status` also checks `/health`; add an
HTTP shutdown endpoint, or have `stop` fall back to the PID reported by
`/health`).

**Revisit.** Next branch touching `gsas_server.py`/`gsas_client.py`.

---

### KI-12 — Server mode assumes a shared filesystem; output files should travel in-band `planned-fix · major`

**What.** The GSAS-II server writes output files into `output_dir` in **its
own** filesystem view and returns only result *data* over HTTP. A server
without a shared view (another cluster node; a sandbox/container with a
private `/tmp`) therefore cannot deliver the file outputs at all. The client's
output-visibility guard (stat-compare of `fit_profile.txt` before/after the
run) detects the mismatch and falls back to in-process execution — a
mitigation, not a fix, and it keeps a (cheap) filesystem check in the client.

**Evidence.** `gsas_server.py` security notes (server resolves `output_dir`
in its own view); `gsas_client.py` `_server_output_visible` /
`_submit_to_server_guarded`; `tests/test_gsas_client_visibility.py`.
mp-simulate's `.chi` export already consumes the in-band `fit_profile` dict
(fix/mp-integration-update), showing the in-band path works.

**Decision.** Planned fix in a future server-protocol branch: the server runs
in a server-local scratch directory and returns **all output artifacts
in-band** (gpx/lst as base64/text, CSVs as text) and the client materializes
the files locally. This dissolves the divergent-view failure instead of
detecting it, deletes the guard, and makes cross-node servers (HPC: client on
a login/worker node, GSAS-II server elsewhere, no shared filesystem) a
supported topology. Needs protocol versioning for already-running servers.

**Revisit.** Next branch touching the server protocol, or when the HPC
deployment work starts.
