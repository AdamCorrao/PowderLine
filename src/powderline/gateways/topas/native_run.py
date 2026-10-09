"""Run a native ``topas.*`` recipe and build the standard result (re/05; draft rev 3 §9-§11, A145-A148).

Steps: write the INP and ``.xye`` (:mod:`.native_writer`), run ``tc.exe`` with
the TOPAS folder as working directory, read the files the INP asked TOPAS to
write, and build the result dict every gateway returns.

TOPAS 6 facts this module is built on (devkit ``re05-topas6-compat.md``):

- the exit code says nothing: 3221226324 on success too (C13), so success is
  judged from **fresh expected output files** (written during this run);
- TOPAS's messages go to the console window only (C14), so everything
  reported comes from those files;
- ``Get(bkg)`` is 0 (C17): the background column is computed here from the
  refined terms (C18 basis; background peaks per A147 Q-B3);
- ``_LIMIT_`` markers are unreliable (C21): limit hits are checked here
  against the fixed limits the writer recorded (A147 Q-L4);
- a non-finite term silently zeroes a phase (C23): a phase whose calculated
  intensities are all zero gets an informative warning, never an error
  (A147 Q-Z1).
"""

from __future__ import annotations

import csv
import math
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from powderline import reports
from powderline.exceptions import EngineExecutionError, StructuredWarning
from powderline.fitstats import compute_fit_statistics
from powderline.gateways.topas.native_writer import CHEBYSHEV_NAME, NativeInput, render_native
from powderline.gateways.topas.roundtrip import PEAK_LIST_COLUMNS, parse_peak_list, parse_results_csv
from powderline.gateways.topas.runner import discover_tc_exe
from powderline.gateways.topas.schema import TopasRietveldRecipe
from powderline.schema_core import collect_warnings, parameters_requested

#: Columns of ``spf_peaks`` for ``topas.spf`` (A147 D13): core's shared columns, then TOPAS's.
SPF_COLUMNS = ("position_2theta", "position_2theta_esd", "intensity", "intensity_esd",
               "gauss_fwhm", "gauss_fwhm_esd", "lor_fwhm", "lor_fwhm_esd",
               "fwhm", "fwhm_esd", "integral_breadth", "integral_breadth_esd")
SPF_DIAG_COLUMNS = ("peak_index", "position_2theta", "status", "notes")
SPF_REPORT = "single_peaks_report.csv"
#: A final value within this relative distance of a fixed limit is a limit hit (A147 Q-L4).
LIMIT_HIT_RTOL = 1e-6


def default_base_name(output_dir) -> str:
    """``examples/<name>/output`` -> ``<name>``; otherwise ``topas`` (as the 0.26.0 path)."""
    out = Path(output_dir)
    return out.parent.name if out.name == "output" and out.parent.name else "topas"


def write_input(model, output_dir, base_name: str | None = None) -> NativeInput:
    """Render and write ``<base>.inp`` and ``<base>.xye`` (UTF-8, LF); return the writer's map."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    native = render_native(model, base_name or default_base_name(out))
    base = native.results_file[: -len("_results.csv")]
    (out / f"{base}.inp").write_text(native.inp_text, encoding="utf-8", newline="\n")
    (out / f"{base}.xye").write_text(native.xye_text, encoding="utf-8", newline="\n")
    return native


def run_tc(tc: Path, inp: Path, runner=subprocess.run) -> None:
    """Run ``tc "<inp>"`` in the TOPAS folder. The exit code is not used (C13)."""
    runner([str(tc), str(inp)], cwd=str(tc.parent), capture_output=True)


def _fresh(path: Path, started: float) -> bool:
    return path.is_file() and path.stat().st_mtime >= started - 1.0


def run_native(model, output_dir, *, base_name: str | None = None, topas_dir=None, topas_version=None,
               engine_version_check=None, runner=subprocess.run, warnings_from=None) -> dict:
    """Generate, run and read back one native recipe; return the standard result dict.

    ``warnings_from`` is the model validated from the user's recipe (A115); by
    default ``model`` itself. ``engine_version_check(tc_path)`` is called only
    when TOPAS will actually run (A145); without ``tc.exe`` the run is
    generate-only, as on the 0.26.0 path.
    """
    started = time.time()
    out = Path(output_dir)
    native = write_input(model, out, base_name)
    base = native.results_file[: -len("_results.csv")]
    validated_warnings = collect_warnings(warnings_from if warnings_from is not None else model)

    tc = discover_tc_exe(topas_dir, topas_version)
    if tc is None:
        return _empty_result(out, success=False, method="topas_generate_only", elapsed=time.time() - started,
                             simulation=native.simulation, warnings=validated_warnings,
                             error=f"tc.exe not found: wrote {base}.inp and {base}.xye only (set topas_dir or "
                                   ".powderline_config.yaml topas.dir to run TOPAS)")
    if engine_version_check is not None:
        engine_version_check(tc)
    run_started = time.time()
    run_tc(tc, out / f"{base}.inp", runner=runner)
    missing = [f for f in (native.results_file, native.profile_file, *native.phase_files.values())
               if not _fresh(out / f, run_started)]
    if missing:
        raise EngineExecutionError(
            f"TOPAS stopped without writing its results ({', '.join(missing)} missing or not updated in {out}). "
            "TOPAS reports the reason only in its console window; run the INP in a terminal to see it. The "
            f"expanded input is in {tc.parent / 'tc.log'}.")
    return build_result(model, native, out, validated_warnings, elapsed=time.time() - started)


# --- reading TOPAS's files --------------------------------------------------------------


def _read_profile(path: Path) -> np.ndarray:
    rows = [[float(tok) for tok in line.split()] for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    arr = np.array(rows, dtype=float).reshape(-1, 3)
    return arr


def _diverged(what: str) -> EngineExecutionError:
    return EngineExecutionError(f"the refinement diverged: {what} is not finite (A130)")


def build_result(model, native: NativeInput, out: Path, validated_warnings: list, *, elapsed: float) -> dict:
    """The result dict from the files TOPAS wrote (tested without TOPAS from recorded files)."""
    p = model.payload
    stats, values = parse_results_csv((out / native.results_file).read_text(encoding="utf-8"))
    for name, (v, e) in values.items():
        if v is None:
            raise _diverged(f"{name} in {native.results_file}")
    profile = _read_profile(out / native.profile_file)
    if not np.isfinite(profile).all():
        raise _diverged(f"the calculated pattern in {native.profile_file}")

    # standard statistics over exactly the stated window (A131), the recipe's weights, TOPAS's Ycalc
    tth = np.asarray(p.xrd_data.tth, dtype=float)
    yobs = np.asarray(p.xrd_data.Itth, dtype=float)
    w = np.asarray(p.xrd_data.Itth_weights, dtype=float)
    lo, hi = p.window()
    mask = (tth >= lo) & (tth <= hi)
    ycalc = np.full_like(tth, np.nan)
    idx = np.searchsorted(tth, profile[:, 0])
    idx = np.clip(idx, 0, len(tth) - 1)
    near = np.abs(tth[idx] - profile[:, 0]) <= 1e-6 * np.maximum(np.abs(tth[idx]), 1.0)
    ycalc[idx[near]] = profile[near, 2]
    lacking = mask & (w > 0) & ~np.isfinite(ycalc)
    if lacking.any():
        raise EngineExecutionError(f"TOPAS did not calculate {int(lacking.sum())} weighted point(s) of the stated "
                                   f"fit window [{lo}, {hi}] (first at {tth[lacking][0]})")
    ycalc_used = np.where(np.isfinite(ycalc), ycalc, 0.0)
    varied = int(round(values.get("number_independent_parameters", (0, None))[0]))
    fs = compute_fit_statistics(yobs, ycalc_used, w, varied, mask=mask)
    undefined = lambda x: None if x is None or not math.isfinite(x) else float(x)  # noqa: E731
    requested = parameters_requested(model)

    warnings = list(validated_warnings)
    if varied < requested:
        warnings.append(StructuredWarning(
            code="topas_parameters_not_varied",
            message=(f"the recipe requests {requested} independently refined parameter(s) but TOPAS varied "
                     f"{varied}: a flagged parameter that does not affect the pattern is not refined (C22)"),
            field_path=None))
    warnings += _limit_hits(native, values)

    refined_rows = _refined_rows(model, native, values)
    unit_cells, peak_lists = {}, {}
    if isinstance(model, TopasRietveldRecipe):
        for i, name in enumerate(p.phases, start=1):
            unit_cells[name] = _cell_rows(model, i, name, values)
            peak_lists[name] = parse_peak_list((out / native.phase_files[name]).read_text(encoding="utf-8"), name)
            if not any(r["I_after_scale_pks"] != 0 for r in peak_lists[name]):
                warnings.append(StructuredWarning(
                    code="topas_phase_contributes_nothing",
                    message=(f"phase {name!r} contributes nothing to the calculated pattern (all its calculated "
                             "intensities are zero): its scale is 0, or TOPAS discarded a calculation that was "
                             "not finite (TOPAS does not report that, C23)"),
                    field_path=f"payload.phases.{name}"))
    spf_rows, spf_diag = _spf_rows(model, native, values) if not isinstance(model, TopasRietveldRecipe) else ([], [])

    wavelength = _final(native, values, "lam_lo", p.instrument.radiation.lo.value)
    bkg = _background(model, native, values, profile[:, 0])
    profile_rows = []
    for x, yo, yc, yb in zip(profile[:, 0], profile[:, 1], profile[:, 2], bkg):
        i = int(np.searchsorted(tth, x))
        weight = w[min(i, len(w) - 1)]
        s = math.sin(math.radians(x / 2))
        profile_rows.append({"two_theta": x, "y_obs": yo, "y_weights": weight, "y_calc": yc, "y_diff": yo - yc,
                             "y_bkg": yb, "q_values": 4 * math.pi * s / wavelength,
                             "d_spacings": wavelength / (2 * s) if s else math.inf})

    reports.write_refined_parameters(out, refined_rows)
    reports.write_fit_profile(out, profile_rows)
    for name, rows in unit_cells.items():
        reports.write_unit_cell_report(out, name, rows)
    for name, rows in peak_lists.items():
        reports.write_peak_list_report(out, name, rows, PEAK_LIST_COLUMNS)
    if spf_rows:
        with open(out / SPF_REPORT, "w", encoding="utf-8", newline="\n") as fh:
            writer = csv.DictWriter(fh, fieldnames=SPF_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(spf_rows)

    return {
        "success": True,
        "run_id": str(uuid.uuid4()),
        "rwp": undefined(fs.rwp),
        "r_exp": undefined(fs.rexp),
        "gof": undefined(fs.gof),
        "chi2_red": undefined(fs.chi2_red),
        "simulation_mode": native.simulation,
        "engine_details": {
            "engine": "TOPAS",
            "engine_version": model.engine_version,
            "rwp": stats.get("r_wp"),
            "r_exp": stats.get("r_exp"),
            "gof": stats.get("gof"),
            "n_points": fs.n_points,
            "parameters_requested": requested,
            "parameters_varied": varied,
        },
        "warnings": warnings,
        "elapsed_time": elapsed,
        "method": "topas_simulation" if native.simulation else "topas",
        "output_files": sorted(str(f) for f in out.glob("*") if f.is_file()),
        "fit_profile": pd.DataFrame(profile_rows, columns=list(reports.FIT_PROFILE_COLUMNS)),
        "unit_cell_data": {k: pd.DataFrame(v, columns=list(reports.UNIT_CELL_COLUMNS)) for k, v in unit_cells.items()},
        "peak_list_data": {k: pd.DataFrame(v, columns=list(PEAK_LIST_COLUMNS)) for k, v in peak_lists.items()},
        "refined_parameters": pd.DataFrame(refined_rows, columns=list(reports.REFINED_PARAMETERS_COLUMNS)),
        "spf_peaks": pd.DataFrame(spf_rows, columns=list(SPF_COLUMNS)) if spf_rows else pd.DataFrame(),
        "spf_convergence_diagnostics": (pd.DataFrame(spf_diag, columns=list(SPF_DIAG_COLUMNS)) if spf_diag
                                        else pd.DataFrame()),
        "error": None,
        "traceback": None,
    }


def _empty_result(out: Path, *, success: bool, method: str, elapsed: float, simulation: bool, warnings: list,
                  error=None) -> dict:
    return {
        "success": success, "run_id": str(uuid.uuid4()), "rwp": None, "r_exp": None, "gof": None,
        "chi2_red": None, "simulation_mode": simulation, "engine_details": {}, "warnings": warnings,
        "elapsed_time": elapsed, "method": method,
        "output_files": sorted(str(f) for f in out.glob("*") if f.is_file()),
        "fit_profile": pd.DataFrame(), "unit_cell_data": {}, "peak_list_data": {},
        "refined_parameters": pd.DataFrame(columns=list(reports.REFINED_PARAMETERS_COLUMNS)),
        "spf_peaks": pd.DataFrame(), "spf_convergence_diagnostics": pd.DataFrame(),
        "error": error, "traceback": None,
    }


def validate_only_result(model, validated_warnings: list) -> dict:
    """The ``validate_only`` summary (the 0.26.0 shape plus native keys)."""
    p = model.payload
    return {
        "success": True, "rwp": None, "elapsed_time": 0.0, "method": "validate_only",
        "schema_name": model.schema_name, "engine_schema_version": model.engine_schema_version,
        "engine_version": model.engine_version,
        "phases": len(p.phases) if isinstance(model, TopasRietveldRecipe) else 0,
        "iters": p.refinement_controls.iters, "simulation_mode": p.refinement_controls.iters == 0,
        "warnings": validated_warnings,
    }


# --- result pieces ------------------------------------------------------------------------


def _final(native: NativeInput, values: dict, name: str, fallback: float) -> float:
    """A parameter's value after the run (TOPAS units): refined value, else the recipe's."""
    return values[name][0] if name in values else fallback


def _limit_hits(native: NativeInput, values: dict) -> list:
    found = []
    for name, info in native.params.items():
        if not info.refined or name not in values:
            continue
        v = values[name][0]
        for side, lim, source in zip(("min", "max"), info.limits, info.limit_source):
            if lim is None:
                continue
            tol = LIMIT_HIT_RTOL * abs(lim) if lim != 0 else 1e-12
            if abs(v - lim) <= tol:
                whose = "the recipe's stated bound" if source == "stated" else "TOPAS's default limit"
                found.append(StructuredWarning(
                    code="topas_parameter_at_limit",
                    message=(f"{info.descriptive} ended at its {side} limit {lim:g} ({whose}): the refinement "
                             "may have been stopped by the limit"),
                    field_path=info.path))
    return found


def _refined_rows(model, native: NativeInput, values: dict) -> list[dict]:
    phases = list(model.payload.phases) if isinstance(model, TopasRietveldRecipe) else []
    rows = []
    for name, info in list(native.params.items()) + list(native.derived.items()):
        if info.category in ("cell_volume",) or name not in values:
            continue
        if info.category != "derived" and not info.refined:
            continue
        v, e = values[name]
        atoms = list(model.payload.phases[info.phase].atoms) if info.phase in phases else []
        rows.append({
            "parameter_name": name, "descriptive_name": info.descriptive, "phase_name": info.phase,
            "phase_idx": phases.index(info.phase) if info.phase in phases else None,
            "atom_name": info.atom, "atom_idx": atoms.index(info.atom) if info.atom in atoms else None,
            "value": v / info.scale, "esd": None if e is None else e / info.scale, "category": info.category,
        })
    return rows


def _cell_rows(model, i: int, name: str, values: dict) -> list[dict]:
    """a..gamma and volume: tied members take their group's refined value and ESD, fixed ones the model's."""
    from powderline.symmetry import cell_tie_groups

    phase = model.payload.phases[name]
    ties = cell_tie_groups(phase.space_group)
    rep = {m: g.members[0] for g in ties.groups for m in g.members}
    rows = []
    for member in ("a", "b", "c", "alpha", "beta", "gamma"):
        key = f"p{i}_cell_{rep.get(member, member)}"
        if member not in ties.fixed and key in values:
            v, e = values[key]
        else:
            v, e = getattr(phase.unit_cell, member).value, 0.0
        rows.append({"parameter": member, "value": v, "esd": e})
    vol = values.get(f"p{i}_volume")
    rows.append({"parameter": "volume", "value": None if vol is None else vol[0],
                 "esd": None if vol is None else vol[1]})
    return rows


def _spf_rows(model, native: NativeInput, values: dict) -> tuple[list[dict], list[dict]]:
    rows, diag = [], []
    for i, peak in enumerate(model.payload.single_peaks, start=1):
        row = {}
        for col, field_, recipe in (("position_2theta", "xo", peak.xo), ("intensity", "I", peak.I),
                                    ("gauss_fwhm", "gauss_fwhm", peak.gauss_fwhm),
                                    ("lor_fwhm", "lor_fwhm", peak.lor_fwhm),
                                    ("fwhm", "fwhm", None), ("integral_breadth", "integral_breadth", None)):
            key = f"spf{i}_{field_}"
            if key in values:
                row[col], row[f"{col}_esd"] = values[key]
            elif recipe is not None:
                row[col], row[f"{col}_esd"] = recipe.value, None
            else:
                row[col], row[f"{col}_esd"] = None, None
        rows.append(row)
        hits = [w for w in _limit_hits(native, values) if w["field_path"].startswith(f"payload.single_peaks.{i - 1}.")]
        if hits:
            diag.append({"peak_index": i - 1, "position_2theta": row["position_2theta"], "status": "at_limit",
                         "notes": "; ".join(w["message"] for w in hits)})
    return rows, diag


# --- background column (A147 Q-B3; C17, C18) ------------------------------------------------


def _background(model, native: NativeInput, values: dict, x: np.ndarray) -> np.ndarray:
    """Chebyshev (T_n over the calculated window, C18) + One_on_X + background peaks at the run's values.

    Background peaks are unit-area pv / split-pv / split-PVII shapes times ``I``
    at ``xo`` + zero error; the xdd-level axial and capillary convolutions are
    not applied to this column (the fit itself has them).
    """
    bg = model.payload.background
    y = np.zeros_like(x)
    if bg is None or len(x) == 0:
        return y
    if bg.chebyshev is not None:
        coeffs = [_final(native, values, f"{CHEBYSHEV_NAME}_bkg{n}__", c) for n, c in enumerate(bg.chebyshev.coefficients)]
        t = (2 * x - (x[0] + x[-1])) / (x[-1] - x[0]) if x[-1] != x[0] else np.zeros_like(x)
        y += np.polynomial.chebyshev.chebval(t, coeffs)
    if bg.One_on_X is not None:
        y += _final(native, values, "oox", bg.One_on_X.value) / x
    corr = model.payload.instrument.corrections
    zero = 0.0
    if corr is not None and corr.Zero_Error is not None:
        zero = _final(native, values, "ze_th2_offset", corr.Zero_Error.th2_offset.value)
    for i, pk in enumerate(bg.peaks or [], start=1):
        v = lambda n, rec: _final(native, values, f"bkpk{i}_{n}", rec.value)  # noqa: E731
        pos = v("xo", pk.xo) + zero
        area = v("I", pk.I)
        prm = {n: v(n, rec) for n, rec in pk.parameters.items()}
        y += area * _peak_shape(pk.peak_type, prm, x - pos)
    return y


def _pv(dx, fwhm, eta):
    g = (2 / fwhm) * math.sqrt(math.log(2) / math.pi) * np.exp(-4 * math.log(2) * dx ** 2 / fwhm ** 2)
    lor = (2 / (math.pi * fwhm)) / (1 + 4 * dx ** 2 / fwhm ** 2)
    return eta * lor + (1 - eta) * g


def _pvii_unit_height(dx, fwhm, m):
    return (1 + 4 * (2 ** (1 / m) - 1) * dx ** 2 / fwhm ** 2) ** (-m)


def _pvii_half_area(fwhm, m):
    """Area of one half of a unit-height Pearson VII."""
    return 0.5 * fwhm * math.sqrt(math.pi) * math.gamma(m - 0.5) / (2 * math.sqrt(2 ** (1 / m) - 1) * math.gamma(m))


def _peak_shape(kind: str, prm: dict, dx: np.ndarray) -> np.ndarray:
    """Unit-area peak (TR p. 39, Tables 6.1/6.2): split halves meet at the peak (one height)."""
    if kind == "pv":
        return _pv(dx, prm["pv_fwhm"], prm["pv_lor"])
    if kind == "spv":
        f1, f2 = 2 * prm["spv_h1"], 2 * prm["spv_h2"]
        h1, h2 = _pv(0.0, f1, prm["spv_l1"]), _pv(0.0, f2, prm["spv_l2"])
        shape = np.where(dx < 0, _pv(dx, f1, prm["spv_l1"]) / h1, _pv(dx, f2, prm["spv_l2"]) / h2)
        return shape / (0.5 / h1 + 0.5 / h2)
    f1, f2 = 2 * prm["h1"], 2 * prm["h2"]
    shape = np.where(dx < 0, _pvii_unit_height(dx, f1, prm["m1"]), _pvii_unit_height(dx, f2, prm["m2"]))
    return shape / (_pvii_half_area(f1, prm["m1"]) + _pvii_half_area(f2, prm["m2"]))


def run_native_validate(model, warnings_from=None) -> dict:
    """``validate_only`` for a native recipe."""
    return validate_only_result(model, collect_warnings(warnings_from if warnings_from is not None else model))
