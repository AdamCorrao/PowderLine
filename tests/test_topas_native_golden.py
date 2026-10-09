"""Native ``topas.*`` fixtures against TOPAS 6 runs (re/05 goldens; A143, A149, A152).

``tests/data/topas/native/golden/`` holds what TOPAS-64 Version 6 wrote for each
fixture in ``tests/data/topas/native/`` (devkit probe packs 3 and 5, base name
``<fixture>__snapped``): the INP the writer produced, TOPAS's results CSV,
profile and reflection lists, and the background run (INP and profile).
Without TOPAS these tests check that the writer still emits exactly those INPs
and that the result built from TOPAS's files is right.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from powderline.gateways.topas import native_run, native_writer, schema
from powderline.gateways.topas.roundtrip import parse_results_csv

NATIVE = Path(__file__).parent / "data" / "topas" / "native"
GOLDEN = NATIVE / "golden"
FIXTURES = ("drx33_rietveld", "lab6_corrections", "lab6_pv_full_axial", "lab6_rietveld", "lab6_spf",
            "ties_simulation")
#: Warnings each TOPAS 6 run gives (beyond the recipe's own validation warnings).
RUN_WARNINGS = {
    "drx33_rietveld": set(),
    "lab6_corrections": set(),
    "lab6_pv_full_axial": {"topas_parameter_at_limit"},
    "lab6_rietveld": {"topas_parameter_at_limit"},
    "lab6_spf": set(),
    "ties_simulation": set(),
}


def _model(fixture):
    return schema.validate_recipe(json.loads((NATIVE / f"{fixture}.json").read_text(encoding="utf-8")))


def _base(fixture):
    return f"{fixture}__snapped"


def _values(fixture):
    _, values = parse_results_csv((GOLDEN / f"{_base(fixture)}_results.csv").read_text(encoding="utf-8"))
    return values


@pytest.fixture(scope="module", params=FIXTURES)
def golden_result(request, tmp_path_factory):
    fixture = request.param
    out = tmp_path_factory.mktemp(fixture)
    for f in GOLDEN.glob(f"{_base(fixture)}*"):
        if not f.name.endswith(".inp"):
            shutil.copy(f, out / f.name)
    model = _model(fixture)
    native = native_writer.render_native(model, _base(fixture))
    return fixture, model, native_run.build_result(model, native, out, [], elapsed=0.0)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_writer_emits_the_inp_topas_ran(fixture):
    model = _model(fixture)
    assert native_writer.render_native(model, _base(fixture)).inp_text == \
        (GOLDEN / f"{_base(fixture)}.inp").read_text(encoding="utf-8")


@pytest.mark.parametrize("fixture", FIXTURES)
def test_background_run_inp_is_the_one_topas_ran(fixture):
    """A152: the background INP from the golden refined values is the one TOPAS 6 ran (pack 5)."""
    values = {name: v for name, (v, _) in _values(fixture).items()}
    assert native_writer.render_background(_model(fixture), _base(fixture), values).inp_text == \
        (GOLDEN / f"{_base(fixture)}_bkg.inp").read_text(encoding="utf-8")


def test_core_rwp_equals_topas_rwp(golden_result):
    """A149: with the fit limits on data points TOPAS fits exactly the window core scores."""
    fixture, model, r = golden_result
    d = r["engine_details"]
    assert r["rwp"] == pytest.approx(d["rwp"], rel=1e-6)
    assert d["n_points"] == 3767
    assert d["parameters_varied"] == d["parameters_requested"]


def test_background_column_is_topas_background_run(golden_result):
    """A152: y_bkg is the background run's Ycalc; TOPAS's Ycalc never falls below it."""
    fixture, model, r = golden_result
    bkg = np.loadtxt(GOLDEN / f"{_base(fixture)}_bkg_profile.txt")
    profile = r["fit_profile"]
    np.testing.assert_array_equal(profile["two_theta"].to_numpy(), bkg[:, 0])
    np.testing.assert_array_equal(profile["y_bkg"].to_numpy(), bkg[:, 2])
    assert (profile["y_calc"] - profile["y_bkg"] > -1e-6).all()


def test_run_warnings(golden_result):
    fixture, model, r = golden_result
    assert {w["code"] for w in r["warnings"]} == RUN_WARNINGS[fixture]


def test_refined_values_are_topas_values(golden_result):
    """Every refined parameter row carries TOPAS's value (recipe units: beq / 8 pi^2)."""
    fixture, model, r = golden_result
    values = _values(fixture)
    native = native_writer.render_native(model, _base(fixture))
    rows = r["refined_parameters"].set_index("parameter_name")
    for name, info in native.params.items():
        if info.refined:
            assert rows.loc[name, "value"] == pytest.approx(values[name][0] / info.scale, rel=1e-12)


def test_drx33_cells(golden_result):
    fixture, model, r = golden_result
    if fixture != "drx33_rietveld":
        pytest.skip("DRX_33 only")
    cells = {name: df.set_index("parameter")["value"] for name, df in r["unit_cell_data"].items()}
    values = _values(fixture)
    (cubic, c1), (mono, c2) = cells.items()
    assert c1["a"] == c1["b"] == c1["c"] == pytest.approx(values["p1_cell_a"][0], rel=1e-12)
    for axis in ("a", "b", "c"):
        assert c2[axis] == pytest.approx(values[f"p2_cell_{axis}"][0], rel=1e-12)
    assert c2["beta"] == pytest.approx(values["p2_cell_beta"][0], rel=1e-12)
    assert c1["volume"] == pytest.approx(values["p1_volume"][0], rel=1e-12)
