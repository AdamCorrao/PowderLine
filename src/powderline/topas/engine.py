"""GSAS-II-free TOPAS engine adapter for ``powderline.run(engine="topas")``.

Composes the existing pieces — :func:`writer.write_topas_inp` →
:func:`runner.run_topas` → :mod:`roundtrip` — and returns the **same dict shape**
the GSAS-II engine returns (locked by ``tests/test_api.py``): DataFrame
``fit_profile`` / ``refined_parameters``, ``dict{phase: DataFrame}``
``unit_cell_data`` / ``peak_list_data``, and every structured key normalised to an
empty DataFrame on failure. Additionally returns the fit stats ``rwp``/``r_exp``/
``gof`` (maintainer request). Imports pandas but **no GSAS-II**.

``write_standardized_outputs`` writes the GSAS-II-matching bare-named report files
(``refined_parameters.csv``, ``fit_profile.txt``, ``<phase>_*.csv``) and is shared
with the ``topas-kicker`` CLI so there is one naming scheme.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pandas as pd

from . import roundtrip as rt
from .errors import TopasTranslationError
from .runner import run_topas
from .writer import (
    PEAKS_SUFFIX,
    PROFILE_SUFFIX,
    RESULTS_SUFFIX,
    _default_base_name,
    render_topas,
    write_topas_inp,
)


def _as_dict(recipe):
    return recipe.model_dump() if hasattr(recipe, "model_dump") else recipe


def _empty_df(columns=None) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns)) if columns else pd.DataFrame()


# --- shared report writer (bare GSAS-II filenames) --------------------------


def write_standardized_outputs(recipe, base_name, results_text, output_dir, source_dir=None,
                               require_params=True) -> dict:
    """Parse TOPAS outputs -> GSAS-II-matching report files; return parsed data.

    Writes ``refined_parameters.csv``, ``<phase>_unit_cell_report.csv``, and (if
    the headerless intermediates exist in ``source_dir``) ``fit_profile.txt`` +
    ``<phase>_peak_list_report.csv``. Fit stats (``r_wp``/``r_exp``/``gof``) live
    in the returned dict, not a bespoke file GSAS-II does not write. Raises
    ``ValueError`` on an empty/unusable results file (loud failure), unless
    ``require_params`` is False -- a **simulation** (calculate-only, zero refined
    parameters) legitimately produces no ``_results.csv``, only the calculated
    profile + peak lists.
    """
    recipe = _as_dict(recipe)
    result = rt.build_roundtrip(recipe, results_text, base_name)
    if require_params and not result.refined_parameters and not result.fit:
        raise ValueError(
            "results file has no parseable parameters or fit stats "
            "(expected a 'parameter,value,esd' table from the TOPAS run)"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    p = output_dir / "refined_parameters.csv"
    p.write_text(rt.refined_parameters_csv(result))
    written.append(p)
    for phase_name, rows in result.unit_cell.items():
        p = output_dir / f"{phase_name}_unit_cell_report.csv"
        p.write_text(rt.unit_cell_report_csv(rows))
        written.append(p)

    src = Path(source_dir) if source_dir else output_dir
    fit_profile_rows: list[dict] = []
    profile_in = src / f"{base_name}{PROFILE_SUFFIX}"
    if profile_in.exists():
        fit_profile_rows = rt.parse_fit_profile(profile_in.read_text(), rt.recipe_wavelength(recipe))
        p = output_dir / "fit_profile.txt"
        p.write_text(rt.fit_profile_txt(fit_profile_rows))
        written.append(p)

    peak_list: dict[str, list] = {}
    for phase_name in (recipe.get("payload", {}).get("phases", {}) or {}):
        peaks_in = src / f"{phase_name}{PEAKS_SUFFIX}"
        if peaks_in.exists():
            rows = rt.parse_peak_list(peaks_in.read_text(), phase_name)
            if rows:
                peak_list[phase_name] = rows
                p = output_dir / f"{phase_name}_peak_list_report.csv"
                p.write_text(rt.peak_list_report_csv(rows))
                written.append(p)

    spf_peaks, spf_diag = [], []
    if recipe.get("schema_name") == "GSASII_SPF":
        spf_peaks, spf_diag = rt.parse_spf_peaks(recipe, results_text)
        if spf_peaks:
            p = output_dir / "single_peaks_report.csv"
            p.write_text(rt.spf_peaks_report_csv(spf_peaks))
            written.append(p)

    return {
        "result": result, "fit_profile": fit_profile_rows, "peak_list": peak_list,
        "spf_peaks": spf_peaks, "spf_diag": spf_diag, "written": written,
    }


# --- run() adapter ----------------------------------------------------------


def _result_dict(*, success, method, output_dir, outputs=None, elapsed=0.0, error=None) -> dict:
    """Assemble the locked run() dict shape from parsed TOPAS outputs (or empties)."""
    rtres = outputs["result"] if outputs else None
    fit = rtres.fit if rtres else {}
    refined = (
        pd.DataFrame(rtres.refined_parameters)
        if (rtres and rtres.refined_parameters)
        else _empty_df(rt.REFINED_COLUMNS)
    )
    fit_profile = (
        pd.DataFrame(outputs["fit_profile"]) if (outputs and outputs["fit_profile"]) else _empty_df()
    )
    unit_cell_data = (
        {ph: pd.DataFrame(rows) for ph, rows in rtres.unit_cell.items()} if rtres else {}
    )
    peak_list_data = (
        {ph: pd.DataFrame(rows) for ph, rows in outputs["peak_list"].items()} if outputs else {}
    )
    spf_peaks = (
        pd.DataFrame(outputs["spf_peaks"], columns=list(rt.SPF_PEAK_COLUMNS))
        if (outputs and outputs["spf_peaks"])
        else _empty_df()  # GSAS-II parity: empty spf_peaks is a column-less DataFrame
    )
    spf_diag = (
        pd.DataFrame(outputs["spf_diag"], columns=list(rt.SPF_DIAG_COLUMNS))
        if (outputs and outputs["spf_diag"])
        else _empty_df()  # GSAS-II parity: empty diagnostics is a column-less DataFrame
    )
    return {
        "success": success,
        "run_id": str(uuid.uuid4()),
        "rwp": fit.get("r_wp"),
        "r_exp": fit.get("r_exp"),
        "gof": fit.get("gof"),
        "elapsed_time": elapsed,
        "method": method,
        "output_files": sorted(str(f) for f in Path(output_dir).glob("*") if f.is_file()),
        "fit_profile": fit_profile,
        "unit_cell_data": unit_cell_data,
        "peak_list_data": peak_list_data,
        "refined_parameters": refined,
        "spf_peaks": spf_peaks,
        "spf_convergence_diagnostics": spf_diag,
        "error": error,
        "traceback": None,
    }


def _validate_only_result(recipe) -> dict:
    """Slim summary matching the GSAS-II ``validate_only`` shape (no run_id)."""
    payload = recipe.get("payload", {})
    controls = payload.get("refinement_controls", {}) or {}
    cycles = controls.get("refinement_cycles")
    return {
        "success": True,
        "rwp": None,
        "elapsed_time": 0.0,
        "method": "validate_only",
        "schema_name": recipe.get("schema_name"),
        "schema_version": recipe.get("schema_version"),
        "phases": len(payload.get("phases") or {}),
        "refinement_cycles": cycles,
        "simulation_mode": cycles == 1,
    }


def run_topas_recipe(
    recipe, output_dir, *, verbose=False, validate_only=False, topas_dir=None, topas_version=None
) -> dict:
    """Drive the TOPAS engine and return the standardized ``run()`` dict.

    ``validate_only`` renders (validating translatability; raises
    ``TopasTranslationError`` on unsupported features) and returns the slim
    summary. Otherwise: generate INP+xye → run ``tc`` if discoverable → parse into
    the standardized tables + files. When ``tc`` is absent (e.g. this Linux box)
    the INP+xye are still written and ``success`` is ``False`` with
    ``method="topas_generate_only"`` and a clear ``error`` (never a crash).
    """
    recipe = _as_dict(recipe)
    output_dir = Path(output_dir)

    if validate_only:
        render_topas(recipe, _default_base_name(str(output_dir)))  # validates; may raise
        return _validate_only_result(recipe)

    gen = write_topas_inp(recipe, str(output_dir), base_name=None)  # may raise TopasTranslationError
    base_name = gen["base_name"]
    if verbose:
        print(f"[topas] wrote {gen['inp_path']}")
        for w in gen["warnings"]:
            print(f"[topas] warning: {w}")

    t0 = time.time()
    run_result = run_topas(gen["inp_path"], topas_dir, topas_version)
    elapsed = time.time() - t0

    if run_result.status == "tc_not_found":
        return _result_dict(
            success=False, method="topas_generate_only", output_dir=output_dir,
            error="TOPAS (tc.exe) not found; INP + xye generated only. "
            "Configure topas.dir/topas.version or pass topas_dir/topas_version.",
        )
    if run_result.status == "failed":
        return _result_dict(
            success=False, method="topas", output_dir=output_dir, elapsed=elapsed,
            error=f"TOPAS exited with code {run_result.returncode}",
        )

    # A simulation (calculate-only) has zero refined prms and emits no
    # _results.csv -- only the calculated profile + peak lists. That is success,
    # not a missing-output failure.
    is_simulation = not gen["refined_names"]
    results_path = output_dir / f"{base_name}{RESULTS_SUFFIX}"
    if not results_path.exists():
        if not is_simulation:
            return _result_dict(
                success=False, method="topas", output_dir=output_dir, elapsed=elapsed,
                error=f"TOPAS ran (rc 0) but {results_path.name} was not produced",
            )
        results_text = ""  # no refined parameters to export
    else:
        results_text = results_path.read_text()
    try:
        outputs = write_standardized_outputs(
            recipe, base_name, results_text, output_dir, require_params=not is_simulation
        )
    except ValueError as exc:
        return _result_dict(
            success=False, method="topas", output_dir=output_dir, elapsed=elapsed, error=str(exc)
        )
    method = "topas_simulation" if is_simulation else "topas"
    return _result_dict(
        success=True, method=method, output_dir=output_dir, outputs=outputs, elapsed=elapsed
    )
