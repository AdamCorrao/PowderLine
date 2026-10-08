"""Native ``gsasii.*`` recipes run end to end (re/04): results, warnings, GSAS-II site check, SPF failure message.

Recipes come from the committed examples through the 0.26.0 converter
(``scripts/convert_recipe_026.py``). The runs use GSAS-II in-process
(``execution_mode='subprocess'``); skipped where GSAS-II is not installed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("GSASII.GSASIIscriptable")

import powderline  # noqa: E402
from powderline.exceptions import EngineExecutionError  # noqa: E402
from powderline.gateways.gsasii.executors import execute_spf_refinement  # noqa: E402
from powderline.symmetry import analyze_site, orbit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("convert_recipe_026", ROOT / "scripts" / "convert_recipe_026.py")
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def _lab6() -> dict:
    """The converted LaB6 example (its background peak raised to GSAS-II's floor first)."""
    old = json.loads((ROOT / "examples" / "example_LaB6" / "input.json").read_text(encoding="utf-8"))
    old["payload"]["background"]["single_peaks"]["pv_lorentzian_gamma"][0][0] = 0.1
    return conv.convert(old)[0]


@pytest.fixture(scope="module")
def lab6_result(tmp_path_factory):
    return powderline.run(_lab6(), tmp_path_factory.mktemp("lab6"), execution_mode="subprocess")


def test_standard_fit_statistics_and_engine_details(lab6_result):
    r = lab6_result
    assert r["success"], r.get("error")
    for key in ("rwp", "r_exp", "gof", "chi2_red"):  # A125
        assert isinstance(r[key], float)
    d = r["engine_details"]
    assert d["engine"] == "GSAS-II"
    # scale, cell (cubic a = b = c is one parameter, A94), 6 Chebyshev, U..Z, 4 background-peak terms
    assert d["parameters_requested"] == 17
    assert d["parameters_varied"] == 17  # GSAS-II's Nvars: counts the 4 peak terms it then drops (EB-33)
    assert abs(r["rwp"] - d["rwp"]) < 0.05  # core Rwp vs GSAS-II's wR (same data; A23 documents differences)
    assert abs(r["gof"] - d["gof"]) / d["gof"] < 0.01  # Rietveld: GSAS-II's 'GOF' is sqrt(reduced chi^2) (EB-52)
    assert r["gof"] == pytest.approx(r["chi2_red"] ** 0.5)
    json.dumps(d)  # crosses the server boundary


def test_warnings_carried_with_full_paths(lab6_result):
    codes = {w["code"]: w for w in lab6_result["warnings"]}
    assert lab6_result["warnings"][0]["field_path"] == "payload.phases.LaB6.peak_broadening.size_broadening"
    assert "gsasii_broadening_default_applied" in codes  # A107, from validating the user's recipe
    msg = codes["gsasii_refinement_message"]  # EB-32: GSAS-II's message, EB-33: dropped count
    assert msg["field_path"] is None and "4 parameter(s) dropped as singular" in msg["message"]


def test_validate_only_summary():
    r = powderline.run(_lab6(), "unused", validate_only=True)
    assert (r["schema_name"], r["core_schema_version"], r["engine_schema_version"], r["phases"]) == \
        ("gsasii.rietveld", "1.0.0", "1.0.0", 1)
    assert "schema_version" not in r and len(r["warnings"]) == 2


def test_refined_corrections_are_varied(tmp_path):
    """zero_shift, polarization and axial_divergence flags reach GSAS-II (A103, review S1)."""
    recipe = _lab6()
    recipe["payload"]["background"].pop("single_peaks")
    for p in recipe["payload"]["instrument"]["corrections"].values():
        p[1] = True
    r = powderline.run(recipe, tmp_path, execution_mode="subprocess")
    assert r["success"], r.get("error")
    assert {":0:Zero", ":0:Polariz.", ":0:SH/L"} <= set(r["refined_parameters"]["parameter_name"])


def _single_atom(tmp_path, sg, cell, xyz, flags=(False, False, False)):
    recipe = _lab6()
    recipe["payload"]["background"].pop("single_peaks")
    p = lambda v, f=False: [v, f]
    recipe["payload"]["phases"] = {"P": {
        "space_group": sg,
        "unit_cell": {k: p(v) for k, v in zip(("a", "b", "c", "alpha", "beta", "gamma"), cell)},
        "atoms": {"Fe": {"element": "Fe", **{k: p(v, f) for k, v, f in zip("xyz", xyz, flags)},
                         "occupancy": p(1.0), "ADP": "Uiso", "Uiso": p(0.01)}},
        "scale": p(1.0)}}
    return powderline.run(recipe, tmp_path, execution_mode="subprocess")


RHOMB = ("R -3 m:H", (5.0, 5.0, 12.0, 90.0, 90.0, 120.0))
CUBIC = ("P 4 3 2", (5.0, 5.0, 5.0, 90.0, 90.0, 90.0))


def test_site_gsasii_misreads_is_an_error_naming_an_equivalent_position(tmp_path):
    """EB-40: GSAS-II doubles the multiplicity at (0, 1/3, 1/6); 4x the intensity, so always an error (A117)."""
    r = _single_atom(tmp_path / "a", *RHOMB, (0.0, 0.333333, 0.166667))
    assert not r["success"]
    assert ("phases.P.atoms.Fe at (0.0, 0.333333, 0.166667): GSAS-II reads it as site '1' with multiplicity 36, "
            "but its multiplicity is 18") in r["error"]
    assert "state the equivalent position (0.666667, 0.666667, 0.166667) instead" in r["error"]
    assert _single_atom(tmp_path / "b", *RHOMB, (0.666667, 0.666667, 0.166667))["success"]


def test_site_gsasii_cannot_name_is_an_error_only_when_refined(tmp_path):
    """EB-05: 'sp' at (0, 1/24, 0) in P432 crashes GSAS-II only when a coordinate is refined (A117)."""
    assert _single_atom(tmp_path / "fixed", *CUBIC, (0.0, 0.041667, 0.0))["success"]
    r = _single_atom(tmp_path / "refined", *CUBIC, (0.0, 0.041667, 0.0), (False, True, False))
    assert not r["success"]
    assert "GSAS-II cannot name its site symmetry ('sp')" in r["error"]
    assert "state the equivalent position (0.958333, 0.0, 0.0) instead" in r["error"]


@pytest.mark.parametrize("sg, point", [("R -3 m:H", (0, 8, 4)), ("R 3 2:H", (16, 0, 8)), ("P 4 3 2", (0, 1, 0)),
                                       ("F m -3 c", (6, 1, 18)), ("P 63/m m c", (8, 16, 6)), ("P 1", (1, 2, 3))])
def test_orbit_has_the_multiplicity(sg, point):
    xyz = tuple(v / 24 for v in point)
    members = orbit(sg, xyz)
    assert len(members) == analyze_site(sg, xyz).multiplicity
    assert all(0.0 <= v < 1.0 for m in members for v in m)
    assert members == orbit(sg, xyz)  # a fixed order


def test_spf_peak_fit_failure_is_a_clear_engine_error():
    """EB-38: refine_peaks raises a bare TypeError when GSAS-II's DoPeakFit fails."""
    hist = MagicMock()
    hist.refine_peaks.side_effect = TypeError("'NoneType' object is not subscriptable")
    recipe = MagicMock()
    recipe.payload.refinement_controls.single_peak_fitting_mode.use_instrument_profile = False
    with pytest.raises(EngineExecutionError, match="GSAS-II's peak fit failed.*EB-38"):
        execute_spf_refinement(MagicMock(), hist, recipe, verbose=False)


def test_spf_runs_with_standard_statistics(tmp_path):
    old = json.loads((ROOT / "examples" / "example_LaB6_singlepeakfit" / "input.json").read_text(encoding="utf-8"))
    r = powderline.run(conv.convert(old)[0], tmp_path, execution_mode="subprocess")
    assert r["success"], r.get("error")
    assert r["engine_details"]["parameters_varied"] > 0 and isinstance(r["r_exp"], float)
    assert not r["spf_peaks"].empty
