"""Recipe -> TOPAS v7 INP + xye writer (plan §4, §6).

``render_topas`` is the pure core: a validated ``GSASII_Rietveld`` payload in,
INP text + xye text out (no I/O), so golden-file tests can exercise it directly.
``write_topas_inp`` wraps it with deterministic file output.

Every formula is implemented and cited in ``conversions.py`` (see
``powderline.topas`` for the citation notation). The package imports zero
GSAS-II code (D3).

Fidelity contract (acceptance §1.3): every recipe parameter flagged
``refine=true`` is emitted as a **named refined prm** (bare name); every other
emitted parameter is fixed (``!name`` or a bare numeric constant). Where the
plan's illustrative blueprint annotations disagree with the actual recipe flags
(e.g. the LaB6 background-peak sigma, which the recipe flags ``true``), the
recipe wins -- it is the point of truth.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from . import conversions as cv
from .errors import TopasTranslationError
from .symmetry import adp_dof, cell_constraints, site_dof

SCHEMA_RIETVELD = "GSASII_Rietveld"
SCHEMA_SPF = "GSASII_SPF"
SCHEMA_NAME_SUPPORTED = SCHEMA_RIETVELD  # back-compat alias

# GSAS-II "refinement cycles" do NOT map numerically to TOPAS iterations
# (findings §C.3): TOPAS runs to convergence. Use a large iteration ceiling and
# let chi2_convergence_criteria stop the refinement (plan §4 blueprint).
_TOPAS_ITERS = 100000

# Broadening defaults used when a parameterization value is null. These match
# kicker.py's DEFAULT_PHASE_CRYSTALLITE_SIZE (10 um -> negligible size
# broadening) and DEFAULT_PHASE_MICROSTRAIN (0 -> no strain broadening), so the
# TOPAS starting model matches the GSAS-II path's own defaults. eta defaults to
# 1.0 (pure Lorentzian), the GSAS-II Size/Mustrain LG_mix default.
_DEFAULT_SIZE_UM = 10.0
_DEFAULT_STRAIN = 0.0
_DEFAULT_ETA = 1.0

# GSAS-II and TOPAS histogram scale factors differ in absolute magnitude: for the
# synchrotron demo data a GSAS-II scale of ~1 corresponds to a TOPAS scale of
# ~1e-6 (the difference absorbs data normalisation, LP and structure-factor
# conventions). Since scale is refined this only sets a better-conditioned
# starting value. Documented approximation, pending maintainer ratification (S5).
_GSASII_TO_TOPAS_SCALE = 1e-6

# TOPAS cell keyword <- recipe cell name
_ANGLE_KEYWORD = {"alpha": "al", "beta": "be", "gamma": "ga"}


@dataclass
class RenderedTopas:
    """Result of rendering a recipe (pure, no I/O)."""

    inp_text: str
    xye_text: str
    base_name: str
    dropped_points: int
    refined_names: frozenset[str]
    warnings: tuple[str, ...] = ()
    # name -> metadata for each refined prm (category, descriptive_name,
    # phase_name, phase_idx, atom_name, transform); the authoritative index the
    # round-trip parser uses to interpret the _results.csv (see roundtrip.py).
    param_index: dict = field(default_factory=dict)


# --- small emission helpers -------------------------------------------------


def _unpack(param, default=None):
    """Return (value, refine_flag) from a 4-list, applying a default value."""
    if param is None:
        return default, False
    value = param[0]
    flag = bool(param[1]) if len(param) > 1 and param[1] is not None else False
    return (default if value is None else value), flag


def _named_token(name: str, value, refined: bool, bounds: str = "") -> str:
    """A value token on a TOPAS keyword: ``name val`` refined, ``!name val`` fixed."""
    prefix = "" if refined else "!"
    tok = f"{prefix}{name} {cv.fmt(value)}"
    return f"{tok} {bounds}".rstrip() if bounds else tok


def _prm_line(name: str, value, refined: bool, bounds: str = "") -> str:
    """A standalone ``prm`` definition line."""
    return "    " + "prm " + _named_token(name, value, refined, bounds)


# --- main entry points ------------------------------------------------------


def _payload_of(recipe) -> dict:
    """Accept a validated RecipeModel or a raw recipe dict; return the payload + meta."""
    if hasattr(recipe, "model_dump"):
        recipe = recipe.model_dump()
    if not isinstance(recipe, dict):
        raise TopasTranslationError("recipe must be a dict or a RecipeModel")
    return recipe


def _check_translatable(payload: dict, schema_name: str) -> None:
    """Fail fast with a clear message on a skeleton/incomplete recipe.

    The shipped `example_template` recipe is a fill-in-the-blanks stub with null
    and empty required fields; without this guard it would crash deep in emission
    with a raw AttributeError. Reject it (and any similarly incomplete recipe)
    up front with a TopasTranslationError naming what is missing.
    """
    missing = []
    tth = (payload.get("xrd_data") or {}).get("tth")
    if not tth:
        missing.append("payload.xrd_data.tth is empty (no pattern to model)")
    if schema_name == SCHEMA_RIETVELD:
        if (payload.get("instrument") or {}).get("parameterization") is None:
            missing.append("payload.instrument.parameterization is null")
        if not (payload.get("phases") or {}):
            missing.append("payload.phases is empty")
    if missing:
        raise TopasTranslationError(
            "recipe is incomplete / a template skeleton and cannot be translated: "
            + "; ".join(missing)
        )


def render_topas(recipe, base_name: str) -> RenderedTopas:
    """Render a recipe to INP + xye text (pure). See module docstring."""
    recipe = _payload_of(recipe)
    schema_name = recipe.get("schema_name")
    if schema_name not in (SCHEMA_RIETVELD, SCHEMA_SPF):
        raise TopasTranslationError(
            f"unsupported schema_name {schema_name!r}; supported: "
            f"{SCHEMA_RIETVELD!r}, {SCHEMA_SPF!r}"
        )
    schema_version = recipe.get("schema_version", "?")
    payload = recipe.get("payload", {})
    _check_translatable(payload, schema_name)

    ctx = _Context(base_name=base_name)
    xye_text, dropped = _render_xye(payload)

    # Render the body first so the header knows whether anything is refined:
    # a recipe with zero refined parameters is a calculate-only "simulation"
    # (GSAS-II refinement_cycles=1 with every flag locked), which TOPAS runs with
    # `iters 0` (no refinement loop, no do_errors) -- see findings D.15.
    body: list[str] = []
    if schema_name == SCHEMA_SPF:
        _emit_spf_xdd(body, payload, base_name, ctx)
    else:
        _emit_globals(body, payload, ctx)
        _emit_xdd(body, payload, base_name, dropped, ctx)

    simulate = not ctx.refined_names
    lines: list[str] = []
    _emit_header(lines, payload, base_name, schema_version, dropped, simulate)
    lines.extend(body)

    inp_text = "\n".join(lines) + "\n"
    return RenderedTopas(
        inp_text=inp_text,
        xye_text=xye_text,
        base_name=base_name,
        dropped_points=dropped,
        refined_names=frozenset(ctx.refined_names),
        warnings=tuple(ctx.warnings),
        param_index=dict(ctx.param_index),
    )


def write_topas_inp(recipe, output_dir: str, base_name: str | None = None) -> dict:
    """Render ``recipe`` and write ``<base>.inp`` + ``<base>.xye`` into ``output_dir``.

    ``base_name`` defaults to the basename of ``output_dir``'s parent example
    directory when derivable, else ``"topas"``. Returns a dict of the written
    paths and render statistics.
    """
    if base_name is None:
        base_name = _default_base_name(output_dir)
    rendered = render_topas(recipe, base_name)

    os.makedirs(output_dir, exist_ok=True)
    inp_path = os.path.join(output_dir, f"{base_name}.inp")
    xye_path = os.path.join(output_dir, f"{base_name}.xye")
    # Explicit ASCII/UTF-8: the INP text is ASCII, so bytes are identical on
    # every platform (Windows would otherwise default to cp1252).
    with open(inp_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(rendered.inp_text)
    with open(xye_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(rendered.xye_text)

    # Warnings (e.g. multiplicity mismatches) are returned, not raised, so the
    # caller (CLI) decides how to surface them without a doubled UserWarning.
    return {
        "inp_path": inp_path,
        "xye_path": xye_path,
        "base_name": base_name,
        "dropped_points": rendered.dropped_points,
        "refined_names": sorted(rendered.refined_names),
        "warnings": list(rendered.warnings),
        "param_index": rendered.param_index,
    }


def _default_base_name(output_dir: str) -> str:
    # output_dir is typically examples/<name>/output/topas -> use <name>
    parts = os.path.normpath(output_dir).split(os.sep)
    if "output" in parts:
        i = parts.index("output")
        if i > 0:
            return parts[i - 1]
    return "topas"


# --- render context ---------------------------------------------------------


@dataclass
class _Context:
    base_name: str
    refined_names: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    param_index: dict = field(default_factory=dict)
    # Per-phase cell-report rows (name, descriptive, phase_name, phase_idx, expr)
    # buffered by ``_emit_cell`` and emitted by ``_emit_results_export`` only when
    # the recipe actually refines something (a simulation writes no results.csv).
    cell_reports: list = field(default_factory=list)

    def note_refined(
        self,
        name: str,
        category: str = "",
        descriptive: str = "",
        phase_name: str = "",
        phase_idx="",
        atom_name: str = "",
        transform: str = "",
    ) -> None:
        """Record a refined prm name and the metadata the round-trip needs.

        ``transform`` names a units conversion the parser applies to recover the
        GSAS-II-native value: ``"beq_to_uiso"`` (Uiso = beq / 8pi^2) or
        ``"topas_scale"`` (a note that the value is a TOPAS-magnitude scale).
        """
        self.refined_names.add(name)
        self.param_index[name] = {
            "category": category,
            "descriptive_name": descriptive or name,
            "phase_name": phase_name,
            "phase_idx": phase_idx,
            "atom_name": atom_name,
            "transform": transform,
        }

    def note_cell_report(self, name, descriptive, phase_name, phase_idx) -> None:
        """Register a TOPAS-authoritative cell-report row in the param index.

        Unlike :meth:`note_refined` this does **not** add to ``refined_names``:
        cell-report rows are derived exports (each phase's ``cell_a..cell_gamma``
        + ``cell_volume`` with TOPAS ESDs), not free refined prms. The round-trip
        parser consumes them via ``category="cell_report"`` (skipped from
        ``refined_parameters``, fed into ``<phase>_unit_cell_report.csv``).
        """
        self.param_index[name] = {
            "category": "cell_report",
            "descriptive_name": descriptive,
            "phase_name": phase_name,
            "phase_idx": phase_idx,
            "atom_name": "",
            "transform": "",
        }

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


# --- header + execution controls --------------------------------------------


def _resolve_instrument_value(payload: dict, key: str) -> float:
    """Broadening/emission value: parameterization value if non-null else Iparm1[key][1]."""
    init = payload.get("instrument", {}).get("initialization", [{}])
    iparm1 = init[0] if init else {}
    fallback = None
    if key in iparm1 and isinstance(iparm1[key], list) and len(iparm1[key]) > 1:
        fallback = iparm1[key][1]
    return fallback


def _emit_header(lines, payload, base_name, schema_version, dropped, simulate=False) -> None:
    instr = payload.get("instrument", {})
    polariz = _resolve_instrument_value(payload, "Polariz.")
    shl = _resolve_instrument_value(payload, "SH/L")
    lines.append(
        f"' Generated by PowderLine topas-kicker from {base_name}/input.json "
        f"(schema {schema_version})"
    )
    lines.append(
        "' Documented approximations: Chebyshev basis transfer (D11); "
        f"Polariz. {cv.fmt(polariz)} -> LP_Factor(90) (D5); "
        f"SH/L {cv.fmt(shl)} omitted, no axial model (D4)."
    )
    # "non-positive weight" is over-broad: negative weights now raise in _render_xye, so
    # only zero-weight (excluded) points can be dropped. The wording is kept because it is
    # baked into the .inp/.out goldens, which can only be regenerated on a Windows/tc.exe
    # run — reword it the next time the goldens are regenerated.
    lines.append(
        f"' Data: {base_name}.xye carries the full pattern; "
        f"{dropped} point(s) with non-positive weight dropped."
    )
    if simulate:
        # Calculate-only: no free parameters, so no refinement loop and no error
        # calculation. TOPAS still computes Ycalc (the simulated pattern); Yobs in
        # the .xye is irrelevant here. Absolute intensities are NOT comparable to
        # GSAS-II (the scale conventions differ) -- only positions and relative
        # intensities are meaningful. See findings D.15.
        lines.append("' simulation (calculate-only): 0 refined prms -> iters 0, no do_errors")
        lines.append("iters 0")
        lines.append("")
        return
    if payload.get("phases"):  # Rietveld-only caveat (no correlated profile terms in SPF)
        lines.append(
            "' Note: when a recipe refines multiple correlated profile terms "
            "(instrument U..Z; size<->strain), those individual values are not "
            "uniquely determined (D6/findings D.6); cell, scale and occupancy are reliable."
        )
    lines.append(f"iters {_TOPAS_ITERS}")
    lines.append("do_errors")
    lines.append("chi2_convergence_criteria 0.001")
    lines.append("")
    lines.append("r_wp 0 r_exp 0 gof 0")
    lines.append("")


# --- global prm block (instrument profile) ----------------------------------

_INSTR_PARAMS = ("U", "V", "W", "X", "Y", "Z")


def _emit_globals(lines, payload, ctx) -> None:
    broadening = payload.get("instrument", {}).get("parameterization", {}).get("broadening", {}) or {}
    lines.append("' --- global instrument profile prms (GSAS-II units: U,V,W centideg^2 variance; X,Y,Z centideg FWHM) ---")
    for key in _INSTR_PARAMS:
        value, refined = _unpack(broadening.get(key), default=_resolve_instrument_value(payload, key))
        name = f"inst_{key}"
        if refined:
            ctx.note_refined(
                name,
                category="instrument_broadening",
                descriptive=f"instrument_broadening_{key}",
            )
        lines.append("prm " + _named_token(name, value, refined))
    lines.append("")


# --- xdd block --------------------------------------------------------------


def _emit_xdd(lines, payload, base_name, dropped, ctx) -> None:
    fit_range = payload.get("fit_range", [None, None])
    lines.append(f'xdd "{base_name}.xye" xye_format')
    if fit_range[0] is not None:
        lines.append(f"    start_X {cv.fmt(fit_range[0])}")
    if fit_range[1] is not None:
        lines.append(f"    finish_X {cv.fmt(fit_range[1])}")
    step = cv.x_calculation_step(payload["xrd_data"]["tth"])
    lines.append(f"    x_calculation_step {cv.fmt(step)}")

    _emit_background(lines, payload, ctx)
    _emit_emission(lines, payload, ctx)
    _emit_single_peaks(lines, payload, ctx)

    for phase_idx, (phase_name, phase) in enumerate((payload.get("phases") or {}).items()):
        _emit_str(lines, phase_name, phase, payload, ctx, phase_idx)

    _emit_fit_profile(lines, base_name)
    # <base>_results.csv is the single round-trip source (no out_prm_vals_on_convergence).
    _emit_results_export(lines, base_name, ctx)


#: Intermediate TOPAS output filenames (headerless numeric; PowderLine's
#: roundtrip.py post-processes them into the GSAS-II-matching final files).
PROFILE_SUFFIX = "_topas_profile.txt"
PEAKS_SUFFIX = "_topas_peaks.txt"


def _emit_fit_profile(lines, base_name) -> None:
    """Emit the whole-pattern fit profile as a headerless numeric table (Stage 2).

    Columns X, Yobs, 1/SigmaYobs^2, Ycalc, Yobs-Ycalc, Get(bkg) — the inputs to
    the GSAS-II `fit_profile.txt` (roundtrip.py computes q/d and adds the header).
    Grounded in the maintainer's confirmed `Out_X_Yobs_Ycalc_Ydiff_Ybkg` /
    `Out_X_Yobs_Ycalc_Ydiff` macros; `Get(bkg)` is the Chebyshev background only
    (xo_Is single-peak contributions land in Ycalc, unlike GSAS-II's y_bkg).
    """
    lines.append("")
    lines.append("' --- fit profile (headerless: X Yobs 1/sig^2 Ycalc Yobs-Ycalc bkg) ---")
    lines.append(f'xdd_out "{base_name}{PROFILE_SUFFIX}" load out_record out_fmt out_eqn')
    lines.append("{")
    lines.append('    " %.8g" = X;')
    lines.append('    " %.8g" = Yobs;')
    lines.append('    " %.8g" = 1 / SigmaYobs^2;')
    lines.append('    " %.8g" = Ycalc;')
    lines.append('    " %.8g" = Yobs - Ycalc;')
    lines.append('    " %.8g\\n" = Get(bkg);')
    lines.append("}")


def _emit_peak_lists(lines, phase_name) -> None:
    """Emit this phase's reflection list as a headerless numeric table (Stage 2b).

    Columns H K L M D_spacing 2theta(=2 Rad Th) I_no_scale_pks I_after_scale_pks
    Iobs_no_scale_pks A01 B01 A11 B11 — all confirmed valid in a `str` `phase_out`
    (findings §D.11 via probe_peaklist.inp). roundtrip.py derives, per reflection,
    F_calc² = A01²+B01²+A11²+B11² and F_obs² = F_calc² · Iobs/I_no_scale_pks for
    the GSAS-II `<phase>_peak_list_report.csv` (peak widths have no faithful
    per-peak TOPAS variable → omitted, not fabricated).
    """
    lines.append(f'        phase_out "{phase_name}{PEAKS_SUFFIX}" load out_record out_fmt out_eqn')
    lines.append("        {")
    lines.append('            " %.0f" = H;')
    lines.append('            " %.0f" = K;')
    lines.append('            " %.0f" = L;')
    lines.append('            " %.0f" = M;')
    lines.append('            " %.8g" = D_spacing;')
    lines.append('            " %.8g" = 2 Rad Th;')
    lines.append('            " %.8g" = I_no_scale_pks;')
    lines.append('            " %.8g" = I_after_scale_pks;')
    lines.append('            " %.8g" = Iobs_no_scale_pks;')
    lines.append('            " %.8g" = A01;')
    lines.append('            " %.8g" = B01;')
    lines.append('            " %.8g" = A11;')
    lines.append('            " %.8g\\n" = B11;')
    lines.append("        }")


#: Filename suffix for the structured, comment-free results file the INP writes.
RESULTS_SUFFIX = "_results.csv"

#: Fit-quality scalars exported (value only; no ESD).
_FIT_STATS = ("r_wp", "r_exp", "gof")


#: Numeric format for the exported value/ESD columns (%g keeps them compact).
_RESULTS_FMT = "%11.8g"


def _emit_results_export(lines, base_name, ctx) -> None:
    """Emit a TOPAS ``out`` block writing a ``parameter,value,esd`` CSV.

    The round-trip source of truth (maintainer, 2026-07-27): a comment-free,
    machine-readable file with every refined parameter's value + ESD, so the
    parser never has to read the ``.OUT``.

    Uses the **raw** ``out_record``/``out_fmt``/``out_eqn``/``out_fmt_err``
    keywords rather than the ``Out``/``Out_String`` macros. Those macros split
    their arguments on commas, and our format strings contain commas (CSV) — that
    mismatch crashed TOPAS with "Number of arguments 3" (findings §D.9). The raw
    keywords are not macros, so embedded commas are literal. ``out_fmt_err`` fills
    the ESD from ``do_errors``; fit scalars (no ESD) omit it and keep a trailing
    comma so every row stays three columns. Order is param_index insertion order.
    """
    if not ctx.param_index:
        return
    lines.append("")
    lines.append("' --- structured results export (comment-free CSV; parsed by roundtrip.py) ---")
    lines.append("' raw out_record primitives (NOT the Out/Out_String macros: those comma-split)")
    lines.append(f'out "{base_name}{RESULTS_SUFFIX}"')
    lines.append(r'    out_record out_fmt "parameter,value,esd\n"')
    for stat in _FIT_STATS:
        # value-only rows (fit scalars have no ESD); trailing comma keeps 3 columns
        lines.append(f'    out_record out_eqn = Get({stat}); out_fmt "{stat},{_RESULTS_FMT},\\n"')
    # snapshot the real refined prms before cell-report rows are registered below
    for name in list(ctx.param_index):
        # value via out_fmt, ESD via out_fmt_err (do_errors); one row per prm
        lines.append(
            f'    out_record out_eqn = {name}; '
            f'out_fmt "{name},{_RESULTS_FMT}," out_fmt_err "{_RESULTS_FMT}\\n"'
        )
    # TOPAS-authoritative per-phase cell report (D10-update): each phase's six cell
    # params + volume with TOPAS values/ESDs, so a cubic phase reports a=b=c and the
    # volume ESD is covariance-propagated. Emitted only here (a simulation returns
    # above with an empty param_index -> no results.csv, recipe-value fallback).
    if ctx.cell_reports:
        lines.append("' per-phase cell report (a,b,c,al,be,ga,volume with TOPAS ESDs)")
    for name, descriptive, phase_name, phase_idx, expr in ctx.cell_reports:
        ctx.note_cell_report(name, descriptive, phase_name, phase_idx)
        lines.append(
            f'    out_record out_eqn = {expr}; '
            f'out_fmt "{name},{_RESULTS_FMT}," out_fmt_err "{_RESULTS_FMT}\\n"'
        )


#: Base name for the Chebyshev background. TOPAS auto-names the coefficients
#: ``{BKG_NAME}_bkg0__``, ``{BKG_NAME}_bkg1__`` … so the round-trip can find them
#: by name (maintainer, 2026-07-27) instead of the opaque ``@`` hash names.
BKG_NAME = "chebyshev_bkg"


def _emit_background(lines, payload, ctx) -> None:
    cheb = payload.get("background", {}).get("chebyshev")
    if not cheb:
        return
    coeffs = cheb.get("coefficients", [])
    refined = bool(cheb.get("refine_flag"))
    # Named Chebyshev block (D11): one name drives all coefficients; '!' fixes it.
    name_token = BKG_NAME if refined else f"!{BKG_NAME}"
    tokens = " ".join(cv.fmt(c) for c in coeffs)
    lines.append("")
    lines.append(
        f"    ' background: {len(coeffs)}-term Chebyshev (D11); TOPAS names the "
        f"coefficients {BKG_NAME}_bkg0__ .. {BKG_NAME}_bkg{len(coeffs) - 1}__"
    )
    lines.append(f"    bkg {name_token} {tokens}")
    if refined:
        # `chebyshev_bkg` is the name that appears in the INP (invariant check);
        # the per-coefficient auto-names are recorded for the round-trip parser.
        ctx.refined_names.add(BKG_NAME)
        for i in range(len(coeffs)):
            ctx.param_index[f"{BKG_NAME}_bkg{i}__"] = {
                "category": "background",
                "descriptive_name": f"background_coefficient_{i}",
                "phase_name": "",
                "phase_idx": "",
                "atom_name": "",
                "transform": "",
            }


def _emit_emission(lines, payload, ctx) -> None:
    param = payload.get("instrument", {}).get("parameterization", {})
    wavelength, refined = _unpack(
        param.get("wavelength"), default=_resolve_instrument_value(payload, "Lam")
    )
    lines.append("")
    lines.append("    ' emission: single synchrotron line (findings B.6); lh is a delta-function width")
    lam_token = _named_token("inst_lam", wavelength, refined) if refined else cv.fmt(wavelength)
    if refined:
        ctx.note_refined("inst_lam", category="wavelength", descriptive="wavelength")
    lines.append(f"    lam ymin_on_ymax 0.001 la 1 lo {lam_token} lh 1e-5")


# --- single-peak-fitting (GSASII_SPF) ---------------------------------------


def _emit_spf_xdd(lines, payload, base_name, ctx) -> None:
    """Emit the xdd body for a pure single-peak-fit recipe (no ``str`` phases).

    The top-level ``single_peaks`` are the whole model: one ``xo_Is`` per peak,
    each carrying its own Gaussian variance (sigma^2) and Lorentzian gamma. With
    ``use_instrument_profile=false`` (the only case in the demo recipes) the peaks
    are not convolved with the instrument profile.
    """
    mode = (payload.get("refinement_controls", {}) or {}).get("single_peak_fitting_mode", {}) or {}
    if mode.get("use_instrument_profile"):
        raise TopasTranslationError(
            "SPF use_instrument_profile=true is not yet supported on the TOPAS path"
        )

    fit_range = payload.get("fit_range", [None, None])
    lines.append(f'xdd "{base_name}.xye" xye_format')
    if fit_range[0] is not None:
        lines.append(f"    start_X {cv.fmt(fit_range[0])}")
    if fit_range[1] is not None:
        lines.append(f"    finish_X {cv.fmt(fit_range[1])}")
    lines.append(f"    x_calculation_step {cv.fmt(cv.x_calculation_step(payload['xrd_data']['tth']))}")

    _emit_background(lines, payload, ctx)
    _emit_emission(lines, payload, ctx)
    _emit_spf_peaks(lines, payload, ctx)
    _emit_fit_profile(lines, base_name)
    _emit_results_export(lines, base_name, ctx)


def _emit_spf_peaks(lines, payload, ctx) -> None:
    sp = payload.get("single_peaks") or {}
    positions = sp.get("positions") or []
    intensities = sp.get("intensities") or []
    sigsqs = sp.get("pv_gaussian_sigma_sq") or []
    gammas = sp.get("pv_lorentzian_gamma") or []
    n = len(positions)
    if not (len(intensities) == len(sigsqs) == len(gammas) == n):
        raise TopasTranslationError("single_peaks lists have inconsistent lengths")
    if n == 0:
        raise TopasTranslationError("GSASII_SPF recipe has no single_peaks to fit")

    lines.append("")
    lines.append(f"    ' --- single-peak fit: {n} pseudo-Voigt peak(s); sigma is variance (sigma^2) ---")
    for i in range(n):
        pos_v, pos_r = _unpack(positions[i])
        int_v, int_r = _unpack(intensities[i])
        sigsq_v, sigsq_r = _unpack(sigsqs[i])
        gam_v, gam_r = _unpack(gammas[i])
        base = f"spfpk{i}"
        pos_n, int_n = f"{base}_pos", f"{base}_int"
        sigsq_n, gam_n = f"{base}_sigsq", f"{base}_gam"
        for nm, r, field_desc in (
            (pos_n, pos_r, "position"),
            (int_n, int_r, "intensity"),
            (sigsq_n, sigsq_r, "sigma_squared"),
            (gam_n, gam_r, "gamma"),
        ):
            if r:
                ctx.note_refined(nm, category="spf_peak", descriptive=f"spf_peak_{i}_{field_desc}")
        lines.append("    xo_Is")
        lines.append("        peak_type pv pv_lor 0 pv_fwhm 0.0001")
        lines.append("        xo " + _named_token(pos_n, pos_v, pos_r))
        lines.append("        I " + _named_token(int_n, int_v, int_r))
        lines.append("        prm " + _named_token(sigsq_n, sigsq_v, sigsq_r, "min 1e-6" if sigsq_r else ""))
        lines.append("        prm " + _named_token(gam_n, gam_v, gam_r, "min 1e-6" if gam_r else ""))
        lines.append(f"        gauss_fwhm = {cv.spf_gauss_fwhm_eq(sigsq_n)};")
        lines.append(f"        lor_fwhm   = {cv.bkg_lor_fwhm_eq(gam_n)};")


def _emit_single_peaks(lines, payload, ctx) -> None:
    sp = payload.get("background", {}).get("single_peaks")
    if not sp:
        return
    positions = sp.get("positions") or []
    intensities = sp.get("intensities") or []
    sigmas = sp.get("pv_gaussian_sigma") or []
    gammas = sp.get("pv_lorentzian_gamma") or []
    n = len(positions)
    if not (len(intensities) == len(sigmas) == len(gammas) == n):
        raise TopasTranslationError("single-peak background lists have inconsistent lengths")

    for i in range(n):
        # Skip peaks that carry any null (mirrors kicker.set_single_peak_background).
        group = [positions[i], intensities[i], sigmas[i], gammas[i]]
        if any(p is None or p[0] is None or p[1] is None for p in group):
            continue
        pos_v, pos_r = _unpack(positions[i])
        int_v, int_r = _unpack(intensities[i])
        sig_v, sig_r = _unpack(sigmas[i])
        gam_v, gam_r = _unpack(gammas[i])
        base = f"bkgpk{i}"
        pos_n, int_n = f"{base}_pos", f"{base}_int"
        sig_n, gam_n = f"{base}_sig", f"{base}_gam"
        for nm, r, field_desc in (
            (pos_n, pos_r, "position"),
            (int_n, int_r, "intensity"),
            (sig_n, sig_r, "sigma"),
            (gam_n, gam_r, "gamma"),
        ):
            if r:
                ctx.note_refined(
                    nm,
                    category="background_peak",
                    descriptive=f"background_peak_{i}_{field_desc}",
                )
        lines.append("")
        lines.append(f"    ' --- background single peak {i}: sigma/gamma are THE prms (D2) ---")
        lines.append("    xo_Is")
        lines.append("        peak_type pv pv_lor 0 pv_fwhm 0.0001")
        lines.append("        xo " + _named_token(pos_n, pos_v, pos_r))
        lines.append("        I " + _named_token(int_n, int_v, int_r))
        # sigma (centideg std-dev) and gamma (centideg FWHM): positivity guard when refined
        lines.append("        " + "prm " + _named_token(sig_n, sig_v, sig_r, "min 1e-6" if sig_r else ""))
        lines.append("        " + "prm " + _named_token(gam_n, gam_v, gam_r, "min 1e-6" if gam_r else ""))
        lines.append(f"        gauss_fwhm = {cv.bkg_gauss_fwhm_eq(sig_n)};")
        lines.append(f"        lor_fwhm   = {cv.bkg_lor_fwhm_eq(gam_n)};")


# --- str (phase) block ------------------------------------------------------


def _emit_str(lines, phase_name, phase, payload, ctx, phase_idx=0) -> None:
    structure = phase.get("structure", {})
    param = phase.get("parameterization", {})
    pname = cv.sanitize(phase_name)

    lines.append("")
    lines.append("    str")
    lines.append(f'        phase_name "{phase_name}"')
    lines.append(f'        space_group "{structure.get("space_group")}"')

    scale_v, scale_r = _unpack(param.get("scale"), default=1.0)
    scale_start = scale_v * _GSASII_TO_TOPAS_SCALE
    scale_name = f"{pname}_scale"
    if scale_r:
        ctx.note_refined(
            scale_name,
            category="scale",
            descriptive=f"phase_{phase_idx}_scale_factor",
            phase_name=phase_name,
            phase_idx=phase_idx,
            transform="topas_scale",
        )
    lines.append(
        f"        ' scale start = recipe scale {cv.fmt(scale_v)} x {cv.fmt(_GSASII_TO_TOPAS_SCALE)} "
        "(GSAS-II->TOPAS magnitude; refined)"
    )
    lines.append("        scale " + _named_token(scale_name, scale_start, scale_r))
    lines.append("        r_bragg 0")

    _emit_cell(lines, phase_name, pname, structure, param, ctx, phase_idx)
    _emit_sites(lines, phase_name, pname, structure, param, ctx, phase_idx)
    _emit_instrument_convolutions(lines, ctx)
    _emit_sample_broadening(lines, pname, param, ctx, phase_name, phase_idx)
    _emit_peak_lists(lines, phase_name)


def _emit_cell(lines, phase_name, pname, structure, param, ctx, phase_idx=0) -> None:
    rules = cell_constraints(structure.get("space_group"))
    ucell = structure.get("unit_cell", {})
    pflags = param.get("unit_cell", {}) or {}

    # TOPAS expression used to export each cell param into <base>_results.csv: the
    # shared length prm name (value + ESD, identical across an equality group so a
    # cubic phase reports a=b=c) or a symmetry-fixed angle's literal (value only).
    cell_exprs: dict[str, str] = {}

    # lengths: one shared prm per equality group, refined iff any member refines
    length_tokens = []
    for group in rules.length_groups:
        group_refined = any(_unpack(pflags.get(name))[1] for name in group)
        gname = f"{pname}_{group[0]}"
        if group_refined:
            ctx.note_refined(
                gname,
                category="cell",
                descriptive=f"cell_{group[0]}",
                phase_name=phase_name,
                phase_idx=phase_idx,
            )
        for name in group:
            value = ucell.get(name)
            length_tokens.append(f"{name} " + _named_token(gname, value, group_refined))
            # every length carries a named prm (refined or fixed !name), so the
            # cell-report export references it directly -> group members share it.
            cell_exprs[name] = gname
    lines.append("        " + "   ".join(length_tokens))

    # angles: symmetry-fixed -> bare constant unless the recipe flags it (then
    # refine + warn, permissive); free angles (monoclinic beta / triclinic) ->
    # named prm per flag.
    angle_tokens = []
    for angle in ("alpha", "beta", "gamma"):
        kw = _ANGLE_KEYWORD[angle]
        value = ucell.get(angle)
        _, flag = _unpack(pflags.get(angle))
        fixed_by_symmetry = angle in rules.fixed_angles
        if fixed_by_symmetry and not flag:
            angle_tokens.append(f"{kw} {cv.fmt(value)}")
            cell_exprs[angle] = cv.fmt(value)  # bare constant -> literal in the export
            continue
        if fixed_by_symmetry and flag:
            ctx.warn(
                f"phase {phase_name!r}: refining symmetry-fixed angle {angle!r} "
                f"({rules.crystal_system}) breaks cell symmetry"
            )
        aname = f"{pname}_{angle}"
        if flag:
            ctx.note_refined(
                aname, category="cell", descriptive=f"cell_{angle}",
                phase_name=phase_name, phase_idx=phase_idx,
            )
        angle_tokens.append(f"{kw} " + _named_token(aname, value, flag))
        cell_exprs[angle] = aname  # named prm (refined or fixed !name)
    lines.append("        " + "   ".join(angle_tokens))

    # Buffer this phase's cell-report rows; _emit_results_export emits them (and
    # registers them in the param index) only when the recipe is a real refinement.
    _buffer_cell_report(ctx, phase_name, phase_idx, pname, cell_exprs)


def _buffer_cell_report(ctx, phase_name, phase_idx, pname, cell_exprs) -> None:
    """Queue the six cell params + volume as results-export rows for one phase."""
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        expr = cell_exprs.get(key)
        if expr is None:
            continue
        ctx.cell_reports.append(
            (f"{pname}_cell_{key}", f"cell_{key}", phase_name, phase_idx, expr)
        )
    ctx.cell_reports.append(
        (f"{pname}_cell_volume", "cell_volume", phase_name, phase_idx,
         _cell_volume_expr(cell_exprs))
    )


def _cell_volume_expr(cell_exprs: dict[str, str]) -> str:
    """Triclinic cell-volume TOPAS equation over the phase's cell prms/constants.

    V = a b c sqrt(1 - cos^2 al - cos^2 be - cos^2 ga + 2 cos al cos be cos ga).
    TOPAS equation trig is in radians, so degree angles are converted inline
    (``* Pi / 180``). When the angle is a literal 90 the cosine collapses to 0.
    Under ``do_errors`` TOPAS propagates the covariance ESD through this equation,
    so ``out_fmt_err`` yields a covariance-correct volume ESD (matching GSAS-II).
    """
    a, b, c = cell_exprs["a"], cell_exprs["b"], cell_exprs["c"]
    ca = f"Cos(({cell_exprs['alpha']}) * Pi / 180)"
    cb = f"Cos(({cell_exprs['beta']}) * Pi / 180)"
    cg = f"Cos(({cell_exprs['gamma']}) * Pi / 180)"
    return (
        f"({a}) * ({b}) * ({c}) * "
        f"Sqrt(1 - {ca}^2 - {cb}^2 - {cg}^2 + 2 * {ca} * {cb} * {cg})"
    )



def _emit_sites(lines, phase_name, pname, structure, param, ctx, phase_idx=0) -> None:
    atoms = structure.get("atoms", {})
    patoms = param.get("atoms", {}) or {}
    space_group = structure.get("space_group")

    # Group coordinate-identical atoms into one shared site (D7); preserve order.
    groups: list[tuple[tuple[float, float, float], list[str]]] = []
    for label, atom in atoms.items():
        key = (float(atom["x"]), float(atom["y"]), float(atom["z"]))
        for gkey, members in groups:
            if gkey == key:
                members.append(label)
                break
        else:
            groups.append((key, [label]))

    for xyz, members in groups:
        _emit_one_site(lines, phase_name, pname, space_group, xyz, members, atoms, patoms, ctx, phase_idx)


def _emit_one_site(lines, phase_name, pname, space_group, xyz, members, atoms, patoms, ctx, phase_idx=0) -> None:
    first = members[0]
    site_label = cv.sanitize(first)

    # Multiplicity cross-check: warn (never error) on orbit-size mismatch (plan §5(5)).
    dof = site_dof(space_group, xyz)
    for label in members:
        mult = atoms[label].get("Multiplicity")
        if mult is not None and int(mult) != dof.orbit_size:
            ctx.warn(
                f"phase {phase_name!r} site {first!r} at {tuple(cv.fmt(c) for c in xyz)}: "
                f"recipe Multiplicity {mult} != symmetry orbit {dof.orbit_size} "
                f"(atom {label!r}); check for a rounded special-position coordinate"
            )

    # Per-axis coordinate emission: fixed -> bare constant; refined -> named prm.
    # TOPAS is permissive (it will refine a symmetry-restricted coordinate and
    # break site symmetry); PowderLine translates faithfully and only WARNS on
    # such a choice -- it does not arbitrate recipe correctness.
    coord_tokens = []
    for axis, value in zip(("x", "y", "z"), xyz):
        # a coordinate is refined if ANY grouped atom flags it (group-OR)
        refined = any(_unpack((patoms.get(label) or {}).get(axis))[1] for label in members)
        if not refined:
            coord_tokens.append(f"{axis} {cv.fmt(value)}")
            continue
        cls = dof.classification(axis)
        if cls != "FREE":
            reason = "symmetry-fixed" if cls == "FIXED" else "symmetry-coupled"
            ctx.warn(
                f"phase {phase_name!r}: refining {reason} coordinate {axis!r} of "
                f"site {first!r} at {tuple(cv.fmt(c) for c in xyz)} "
                f"(space group {space_group!r}) breaks site symmetry"
            )
        cname = f"{pname}_{site_label}_{axis}"
        ctx.note_refined(
            cname,
            category="atom_coordinate",
            descriptive=f"{first}_{axis}",
            phase_name=phase_name,
            phase_idx=phase_idx,
            atom_name=first,
        )
        coord_tokens.append(f"{axis} " + _named_token(cname, value, True))

    # occupancy + ADP per member (shared site => multiple occ entries, D7).
    # Uiso emits beq (=8pi^2 U); Uaniso emits the site-level u11..u23 tensor.
    occ_tokens = []
    aniso_tokens: list[str] = []
    for label in members:
        atom = atoms[label]
        patom = patoms.get(label) or {}
        element = atom.get("element", label)
        occ_v, occ_r = _unpack(patom.get("occupancy"), default=atom.get("occupancy"))
        occ_name = f"{pname}_{cv.sanitize(label)}_occ"
        if occ_r:
            ctx.note_refined(
                occ_name, category="occupancy", descriptive=f"{label}_occupancy",
                phase_name=phase_name, phase_idx=phase_idx, atom_name=label,
            )
        occ_field = _named_token(occ_name, occ_v, occ_r) if occ_r else cv.fmt(occ_v)

        adp = str(patom.get("ADP", atom.get("ADP", "Uiso"))).lower()
        if adp == "uaniso":
            aniso_tokens = _uaniso_tokens(
                pname, label, atom, patom, space_group, xyz, ctx, phase_name, phase_idx
            )
            occ_tokens.append(f"occ {element} {occ_field}")
        else:
            beq_name = f"{pname}_{cv.sanitize(label)}_beq"
            uiso_v, uiso_r = _unpack(patom.get("Uiso"), default=atom.get("Uiso"))
            beq_v = cv.beq_from_uiso(uiso_v)
            if uiso_r:
                # reported back as Uiso via the beq_to_uiso transform (Uiso = beq / 8pi^2)
                ctx.note_refined(
                    beq_name, category="atom_adp", descriptive=f"{label}_Uiso",
                    phase_name=phase_name, phase_idx=phase_idx, atom_name=label,
                    transform="beq_to_uiso",
                )
            occ_tokens.append(f"occ {element} {occ_field} beq {_named_token(beq_name, beq_v, uiso_r)}")

    if len(members) > 1:
        lines.append(f"        ' merged shared site (D7): {', '.join(members)}")
    site_line = "site " + site_label + " " + " ".join(coord_tokens) + " " + " ".join(occ_tokens)
    if aniso_tokens:
        site_line += " " + " ".join(aniso_tokens)
    lines.append("        " + site_line)


#: TOPAS u_ij keyword <- recipe Uaniso key.
_UANISO_KEYS = (("u11", "U11"), ("u22", "U22"), ("u33", "U33"),
                ("u12", "U12"), ("u13", "U13"), ("u23", "U23"))


def _uaniso_tokens(pname, label, atom, patom, space_group, xyz, ctx, phase_name, phase_idx) -> list[str]:
    """Emit the six ``u_ij`` for an anisotropic ADP atom (permissive; warns on a
    refine flag on a symmetry-restricted component but does not enforce it)."""
    aniso = patom.get("Uaniso") or {}
    dof = adp_dof(space_group, xyz)
    tokens = []
    for tkey, rkey in _UANISO_KEYS:
        entry = aniso.get(rkey)
        value, flag = _unpack(entry)
        if value is None:  # isotropic-U fallback for a missing component
            value = atom.get("Uiso") if tkey in ("u11", "u22", "u33") else 0.0
        name = f"{pname}_{cv.sanitize(label)}_{rkey}"
        if flag:
            ctx.note_refined(
                name, category="atom_adp", descriptive=f"{label}_{rkey}",
                phase_name=phase_name, phase_idx=phase_idx, atom_name=label,
            )
            if dof.classification(tkey) != "FREE":
                ctx.warn(
                    f"phase {phase_name!r}: refining symmetry-restricted ADP {rkey} "
                    f"({dof.classification(tkey)}) of atom {label!r} at "
                    f"{tuple(cv.fmt(c) for c in xyz)} breaks site symmetry"
                )
        tokens.append(f"{tkey} " + _named_token(name, value, flag))
    return tokens


def _emit_instrument_convolutions(lines, ctx) -> None:
    lines.append("        ' instrument profile as literal GSAS-II convolutions (D4); prms are global")
    # Gaussian arg is already Max-guarded inside the Sqrt; guard the Lorentzian
    # too so a GSAS-II X/Y/Z refining negative can't drive lor_fwhm <= 0 (S5).
    lines.append(f"        gauss_fwhm = {cv.instrument_gauss_fwhm_eq('inst_U', 'inst_V', 'inst_W')};")
    lines.append(f"        lor_fwhm   = {cv.guard_fwhm(cv.instrument_lor_fwhm_eq('inst_X', 'inst_Y', 'inst_Z'))};")
    lines.append("        LP_Factor(90)")


def _lor_share_is_fixed_zero(mag_v, mag_r, eta_v, eta_r) -> bool:
    """Lorentzian convolution is identically zero and fixed (skip it)."""
    return (not mag_r and mag_v == 0) or (not eta_r and eta_v == 0)


def _gauss_share_is_fixed_zero(mag_v, mag_r, eta_v, eta_r) -> bool:
    """Gaussian convolution (the ``1-eta`` share) is identically zero and fixed (skip it)."""
    return (not mag_r and mag_v == 0) or (not eta_r and eta_v == 1)


def _emit_sample_broadening(lines, pname, param, ctx, phase_name="", phase_idx=0) -> None:
    pb = param.get("peak_broadening", {}) or {}
    size = pb.get("size_broadening", {}) or {}
    strain = pb.get("strain_broadening", {}) or {}

    size_v, size_r = _unpack(size.get("isotropic_size"), default=_DEFAULT_SIZE_UM)
    size_eta_v, size_eta_r = _unpack(size.get("LG_eta"), default=_DEFAULT_ETA)
    strain_v, strain_r = _unpack(strain.get("isotropic_strain"), default=_DEFAULT_STRAIN)
    strain_eta_v, strain_eta_r = _unpack(strain.get("LG_eta"), default=_DEFAULT_ETA)

    size_n = f"{pname}_size_um"
    size_eta_n = f"{pname}_size_eta"
    strain_n = f"{pname}_strain"
    strain_eta_n = f"{pname}_strain_eta"

    # Decide which convolutions survive. A convolution that is fixed at exactly
    # zero (strain=0 by default, or a Gaussian share with eta fixed at 1) is
    # skipped entirely rather than emitted as a Max(0, 1e-9) no-op. Live
    # convolutions are Max-guarded so a refining eta hitting a bound (share -> 0)
    # or a magnitude excursion can't yield a non-positive FWHM (S5).
    size_lor = not _lor_share_is_fixed_zero(size_v, size_r, size_eta_v, size_eta_r)
    size_gauss = not _gauss_share_is_fixed_zero(size_v, size_r, size_eta_v, size_eta_r)
    strain_lor = not _lor_share_is_fixed_zero(strain_v, strain_r, strain_eta_v, strain_eta_r)
    strain_gauss = not _gauss_share_is_fixed_zero(strain_v, strain_r, strain_eta_v, strain_eta_r)

    lines.append("        ' sample broadening: literal GSAS-II parameterization (D2; findings C.1)")
    if not (size_lor or size_gauss or strain_lor or strain_gauss):
        lines.append("        ' (no size/strain broadening: all convolutions fixed at zero)")

    # Emit a prm only if a surviving convolution references it.
    if size_lor or size_gauss:
        if size_r:
            ctx.note_refined(size_n, category="size_broadening", descriptive="size_isotropic_um",
                             phase_name=phase_name, phase_idx=phase_idx)
        if size_eta_r:
            ctx.note_refined(size_eta_n, category="size_broadening", descriptive="size_LG_eta",
                             phase_name=phase_name, phase_idx=phase_idx)
        lines.append("        prm " + _named_token(size_n, size_v, size_r, "min 1e-6" if size_r else ""))
        lines.append("        prm " + _named_token(size_eta_n, size_eta_v, size_eta_r, "min 0 max 1"))
    if strain_lor or strain_gauss:
        if strain_r:
            ctx.note_refined(strain_n, category="strain_broadening", descriptive="strain_isotropic",
                             phase_name=phase_name, phase_idx=phase_idx)
        if strain_eta_r:
            ctx.note_refined(strain_eta_n, category="strain_broadening", descriptive="strain_LG_eta",
                             phase_name=phase_name, phase_idx=phase_idx)
        lines.append("        prm " + _named_token(strain_n, strain_v, strain_r, "min 1e-6" if strain_r else ""))
        lines.append("        prm " + _named_token(strain_eta_n, strain_eta_v, strain_eta_r, "min 0 max 1"))

    if size_lor:
        lines.append(f"        lor_fwhm   = {cv.guard_fwhm(cv.size_lor_fwhm_eq(size_eta_n, size_n))};")
    if size_gauss:
        lines.append(f"        gauss_fwhm = {cv.guard_fwhm(cv.size_gauss_fwhm_eq(size_eta_n, size_n))};")
    if strain_lor:
        lines.append(f"        lor_fwhm   = {cv.guard_fwhm(cv.strain_lor_fwhm_eq(strain_eta_n, strain_n))};")
    if strain_gauss:
        lines.append(f"        gauss_fwhm = {cv.guard_fwhm(cv.strain_gauss_fwhm_eq(strain_eta_n, strain_n))};")


# --- .xye writer ------------------------------------------------------------


def _render_xye(payload: dict) -> tuple[str, int]:
    """Three columns tth / Itth / sigma=1/sqrt(w) (D8).

    Strict, non-mutating export consistent with PowderLine's xrd_data policy:

    * Intensities pass through **unchanged** — negative intensities (background subtraction)
      are valid data and are preserved, never zeroed.
    * **Malformed** data is REJECTED, not silently dropped: a non-finite tth/Itth, a
      non-strictly-increasing tth, or a weight that is non-finite or ``< 0``, raises
      ``TopasTranslationError``. (The recipe schema enforces the same rules, but this writer
      is also reachable with raw dicts — e.g. the ``topas-kicker`` CLI — and the TOPAS
      package deliberately imports no pydantic, so it must stand on its own.)
    * A weight of **exactly 0** is a legitimate "exclude this point" marker (schema-valid) that
      the TOPAS ``xye_format`` cannot represent (sigma ``1/sqrt(0)`` is infinite); such points
      are omitted from the .xye and counted (returned drop count, reported in the header) —
      the one unavoidable, transparent exclusion. Everything with a positive weight is written.
      If *every* weight is 0 there is no pattern left to fit, which is rejected too.
    """
    xrd = payload.get("xrd_data", {})
    tth = xrd.get("tth", [])
    itth = xrd.get("Itth", [])
    weights = xrd.get("Itth_weights", [])
    if not (len(tth) == len(itth) == len(weights)):
        raise TopasTranslationError("xrd_data arrays (tth/Itth/Itth_weights) length mismatch")
    if len(tth) == 0:
        raise TopasTranslationError("xrd_data is empty; no pattern to write to the .xye")

    def _first_where(vals, ok) -> int:
        for i, v in enumerate(vals):
            if v is None or not ok(v):
                return i
        return -1

    i = _first_where(tth, math.isfinite)
    if i >= 0:
        raise TopasTranslationError(f"xrd_data.tth[{i}]={tth[i]!r} is non-finite; cannot write .xye")
    i = _first_where(itth, math.isfinite)  # negative intensities allowed; only NaN/inf rejected
    if i >= 0:
        raise TopasTranslationError(f"xrd_data.Itth[{i}]={itth[i]!r} is non-finite; cannot write .xye")
    for i in range(1, len(tth)):
        if tth[i] <= tth[i - 1]:
            raise TopasTranslationError(
                f"xrd_data.tth must be strictly increasing; tth[{i}]={tth[i]!r} <= "
                f"tth[{i - 1}]={tth[i - 1]!r}. Sort/deduplicate the pattern upstream."
            )
    i = _first_where(weights, lambda w: math.isfinite(w) and w >= 0.0)
    if i >= 0:
        raise TopasTranslationError(
            f"xrd_data.Itth_weights[{i}]={weights[i]!r} is negative or non-finite. Weights are "
            "1/sigma**2 and cannot be negative; PowderLine rejects malformed data rather than "
            "silently dropping it. Fix or reweight this point upstream."
        )

    rows = []
    dropped = 0
    for x, y, w in zip(tth, itth, weights):
        if w == 0.0:  # legitimate exclusion; sigma = 1/sqrt(0) is not representable in xye_format
            dropped += 1
            continue
        rows.append(f"{cv.fmt(x)} {cv.fmt(y)} {cv.fmt(1.0 / math.sqrt(w))}")
    if not rows:
        raise TopasTranslationError(
            "every xrd_data point has weight 0 (all excluded) — nothing to fit; "
            "cannot write a usable .xye"
        )
    return ("\n".join(rows) + "\n", dropped)
