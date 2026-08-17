"""Integration tests: per-parameter refine flags honored end-to-end (schema 0.26).

A recipe that requests a subset of a lumped GSAS-II parameter class must
refine exactly that subset — verified against the actual post-refinement
varyList and the held values in the saved project.

These tests run real GSAS-II refinements on small synthetic patterns; the fit
quality is irrelevant (the data are featureless), only the constraint
machinery is under test.
"""
import numpy as np
import pytest

from GSASII import GSASIIscriptable as G2
from powderline.kicker import run_refinement
from powderline.schema import RecipeModel, EXPECTED_SCHEMA_VERSION

ON = [None, True, None, None]
OFF = [None, False, None, None]


def make_recipe(space_group, cell, atoms_structure, parameterization):
    """Minimal valid GSASII_Rietveld recipe around a synthetic pattern."""
    tth = np.arange(5.0, 90.0, 0.05)
    y = 100.0 + 10.0 * np.sin(tth / 5.0)
    w = 1.0 / np.sqrt(np.maximum(y, 1.0))
    a, b, c, alpha, beta, gamma = cell
    return {
        "schema_name": "GSASII_Rietveld",
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "payload": {
            "xrd_data": {
                "tth": tth.tolist(),
                "Itth": y.tolist(),
                "Itth_weights": w.tolist(),
            },
            "instrument": {
                "description": "synthetic CW instrument",
                "initialization": [
                    {
                        "Type": "PXC", "Bank": 1, "Lam": 1.5406, "Zero": 0.0,
                        "Polariz.": 0.7, "Azimuth": 0.0,
                        "U": 2.0, "V": -2.0, "W": 5.0,
                        "X": 0.0, "Y": 0.0, "Z": 0.0, "SH/L": 0.002,
                    },
                    {},
                ],
            },
            "phases": {
                "P": {
                    "structure": {
                        "phase_name": "P",
                        "space_group": space_group,
                        "unit_cell": {
                            "a": a, "b": b, "c": c,
                            "alpha": alpha, "beta": beta, "gamma": gamma,
                        },
                        "atoms": atoms_structure,
                    },
                    "parameterization": parameterization,
                }
            },
            "refinement_controls": {"refinement_cycles": 2},
        },
    }


def run(recipe_dict, tmp_path):
    model = RecipeModel.model_validate(recipe_dict)
    result = run_refinement(model, tmp_path, verbose=False, method="subprocess")
    assert result["success"], f"refinement failed: {result.get('error')}"
    gpx = G2.G2Project(str(tmp_path / "dummy.gpx"))
    vary = list(gpx.data["Covariance"]["data"].get("varyList", []))
    return result, gpx, vary


def test_tetragonal_refine_a_hold_c(tmp_path):
    """Recipe-level probe2: a=true, c=false -> A0 varies, A2 held, c unchanged."""
    atoms = {"Ba": {"element": "Ba", "x": 0.0, "y": 0.0, "z": 0.0,
                    "occupancy": 1.0, "ADP": "Uiso", "Uiso": 0.01}}
    param = {"unit_cell": {"a": ON, "b": OFF, "c": OFF,
                           "alpha": OFF, "beta": OFF, "gamma": OFF}}
    recipe = make_recipe("P 4/m m m", (3.90, 3.90, 4.50, 90, 90, 90), atoms, param)
    result, gpx, vary = run(recipe, tmp_path)

    assert "0::A0" in vary
    assert "0::A2" not in vary
    final_cell = gpx.data["Phases"]["P"]["General"]["Cell"]
    assert final_cell[3] == pytest.approx(4.50, abs=1e-9)   # c held exactly
    assert final_cell[1] != pytest.approx(3.90, abs=1e-12)  # a actually refined


def test_tetragonal_refine_c_hold_a(tmp_path):
    """Holding the equivalence-independent A0 cascades to A1; only c refines."""
    atoms = {"Ba": {"element": "Ba", "x": 0.0, "y": 0.0, "z": 0.0,
                    "occupancy": 1.0, "ADP": "Uiso", "Uiso": 0.01}}
    param = {"unit_cell": {"a": OFF, "b": OFF, "c": ON,
                           "alpha": OFF, "beta": OFF, "gamma": OFF}}
    recipe = make_recipe("P 4/m m m", (3.90, 3.90, 4.50, 90, 90, 90), atoms, param)
    result, gpx, vary = run(recipe, tmp_path)

    assert "0::A2" in vary
    assert "0::A0" not in vary
    final_cell = gpx.data["Phases"]["P"]["General"]["Cell"]
    assert final_cell[1] == pytest.approx(3.90, abs=1e-9)   # a held exactly


def test_p1_refine_x_only(tmp_path):
    """Recipe-level probe3: x=true, y/z=false -> only dAx varies; y,z unchanged."""
    atoms = {
        "Na": {"element": "Na", "x": 0.111, "y": 0.222, "z": 0.333,
               "occupancy": 1.0, "ADP": "Uiso", "Uiso": 0.01},
        "Cl": {"element": "Cl", "x": 0.611, "y": 0.522, "z": 0.733,
               "occupancy": 1.0, "ADP": "Uiso", "Uiso": 0.01},
    }
    param = {"atoms": {"Na": {"x": ON, "y": OFF, "z": OFF, "ADP": "Uiso"}}}
    recipe = make_recipe("P 1", (5.0, 6.0, 7.0, 88.0, 92.0, 95.0), atoms, param)
    result, gpx, vary = run(recipe, tmp_path)

    assert "0::dAx:0" in vary
    assert "0::dAy:0" not in vary
    assert "0::dAz:0" not in vary
    na = gpx.data["Phases"]["P"]["Atoms"][0]
    assert na[4] == pytest.approx(0.222, abs=1e-12)  # y held
    assert na[5] == pytest.approx(0.333, abs=1e-12)  # z held


def test_p1_refine_single_uij_component(tmp_path):
    """Recipe-level probe5: U11=true, others=false -> only AU11 varies."""
    uaniso_vals = {"U11": 0.011, "U22": 0.012, "U33": 0.013,
                   "U12": 0.001, "U13": 0.002, "U23": 0.003}
    atoms = {
        "W": {"element": "W", "x": 0.2, "y": 0.3, "z": 0.4,
              "occupancy": 1.0, "ADP": "Uaniso", "Uiso": None,
              "Uaniso": uaniso_vals},
    }
    param = {"atoms": {"W": {
        "ADP": "Uaniso",
        "Uaniso": {"U11": [0.011, True, None, None],
                   "U22": [0.012, False, None, None],
                   "U33": [0.013, False, None, None],
                   "U12": [0.001, False, None, None],
                   "U13": [0.002, False, None, None],
                   "U23": [0.003, False, None, None]},
    }}}
    recipe = make_recipe("P 1", (5.0, 6.0, 7.0, 88.0, 92.0, 95.0), atoms, param)
    result, gpx, vary = run(recipe, tmp_path)

    au_varied = [v for v in vary if "::AU" in v]
    assert au_varied == ["0::AU11:0"]
    w = gpx.data["Phases"]["P"]["Atoms"][0]
    # indices 12..16 = U22, U33, U12, U13, U23 — all held at input values
    assert w[12] == pytest.approx(0.012, abs=1e-12)
    assert w[13] == pytest.approx(0.013, abs=1e-12)
    assert w[14] == pytest.approx(0.001, abs=1e-12)
    assert w[15] == pytest.approx(0.002, abs=1e-12)
    assert w[16] == pytest.approx(0.003, abs=1e-12)


def test_uniform_flags_unchanged_behavior(tmp_path):
    """All symmetry-linked flags true -> whole cell refines, no holds (legacy path)."""
    atoms = {"Ba": {"element": "Ba", "x": 0.0, "y": 0.0, "z": 0.0,
                    "occupancy": 1.0, "ADP": "Uiso", "Uiso": 0.01}}
    param = {"unit_cell": {"a": ON, "b": ON, "c": ON,
                           "alpha": OFF, "beta": OFF, "gamma": OFF}}
    recipe = make_recipe("P 4/m m m", (3.90, 3.90, 4.50, 90, 90, 90), atoms, param)
    result, gpx, vary = run(recipe, tmp_path)

    assert "0::A0" in vary and "0::A2" in vary
    assert gpx.data["Constraints"]["data"]["Phase"] == []  # no holds emitted
