"""Recipe -> easydiffraction Project translation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

ed = pytest.importorskip("easydiffraction")

from powderline.easydiff.builder import BuildResult, ManifestEntry, build_project
from powderline.easydiff.conversions import GAUSS_CDEG2_TO_DEG2

from test_easydiff_policy import base_recipe  # same tests/ dir; pytest adds it to sys.path


def rich_recipe():
    r = base_recipe()
    tth = np.linspace(1.0, 15.0, 200)
    r["payload"]["xrd_data"] = {
        "tth": tth.tolist(),
        "Itth": (100 + 10 * np.exp(-((tth - 5) ** 2))).tolist(),
        "Itth_weights": (np.ones_like(tth) / 100.0).tolist(),
    }
    pz = r["payload"]["phases"]["LaB6"]["parameterization"]
    pz["unit_cell"]["a"] = [None, True, None, None]
    bz = r["payload"]["instrument"]["parameterization"]["broadening"]
    bz["U"] = [None, True, 0.0, 100.0]
    return r


def test_build_project_returns_fit_ready_result(tmp_path):
    br = build_project(rich_recipe(), tmp_path)
    assert isinstance(br, BuildResult)
    assert br.phase_slugs == {"LaB6": "lab6"}
    assert (tmp_path / "easydiff_data.xye").exists()
    assert br.wavelength == pytest.approx(0.1665)
    assert br.mask.sum() == 200  # all points inside [1, 15] with positive weights
    # structure landed with Uiso convention and cubic symmetry
    model = br.project.structures["lab6"]
    assert model.cell.length_a.value == pytest.approx(4.15682)


def test_manifest_covers_freed_parameters(tmp_path):
    br = build_project(rich_recipe(), tmp_path)
    names = {m.parameter_name for m in br.manifest}
    # scale + cheb 0/1 + U + cell a  (V..Y flags are False in the fixture)
    assert names == {"0:0:Scale", ":0:Back;0", ":0:Back;1", ":0:U", "0::a"}
    u = next(m for m in br.manifest if m.parameter_name == ":0:U")
    assert u.category == "instrument_broadening"
    assert u.scale_to_recipe == pytest.approx(1.0 / GAUSS_CDEG2_TO_DEG2)
    assert u.parameter.free is True
    # bounds converted into easydiffraction units
    assert u.parameter.fit_min == pytest.approx(0.0)
    assert u.parameter.fit_max == pytest.approx(100.0 * GAUSS_CDEG2_TO_DEG2)
    cell = next(m for m in br.manifest if m.parameter_name == "0::a")
    assert cell.phase_name == "LaB6" and cell.category == "unit_cell"


def test_initial_values_converted(tmp_path):
    br = build_project(rich_recipe(), tmp_path)
    expt = br.experiment
    assert expt.peak.broad_gauss_u.value == pytest.approx(18.7 * GAUSS_CDEG2_TO_DEG2)
    assert expt.peak.broad_lorentz_x.value == pytest.approx(0.0028, abs=1e-4)
    assert expt.background["0"].coef.value == pytest.approx(10.0)


def test_symmetry_constrained_axis_not_in_manifest(tmp_path):
    r = rich_recipe()
    r["payload"]["phases"]["LaB6"]["parameterization"]["unit_cell"]["b"] = [None, True, None, None]
    br = build_project(r, tmp_path)
    names = {m.parameter_name for m in br.manifest}
    assert "0::b" not in names  # cubic: b is symmetry-tied to a
    assert "0::a" in names  # flag folded into representative
    assert any("tied" in w for w in br.warnings)  # warning about tied axis


def test_tetragonal_symmetry(tmp_path):
    r = base_recipe()
    # Create tetragonal phase (P 42/m n m)
    r["payload"]["phases"]["Tetragonal"] = {
        "structure": {
            "phase_name": "Tetragonal",
            "space_group": "P 42/m n m",
            "unit_cell": {"a": 4.59, "b": 4.59, "c": 2.96,
                          "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "atoms": {"Ti": {"element": "Ti", "x": 0.0, "y": 0.0, "z": 0.0,
                             "occupancy": 1.0, "Uiso": 0.01, "ADP": "Uiso"}}
        },
        "parameterization": {
            "atoms": {"Ti": {k: [None, False, None, None]
                             for k in ("x", "y", "z", "occupancy", "Uiso")}},
            "scale": [1, True, None, None],
            "unit_cell": {
                "a": [None, True, None, None],
                "b": [None, False, None, None],
                "c": [None, True, None, None],
                "alpha": [None, False, None, None],
                "beta": [None, False, None, None],
                "gamma": [None, True, None, None]  # fixed by symmetry
            },
        },
    }
    tth = np.linspace(1.0, 15.0, 200)
    r["payload"]["xrd_data"] = {
        "tth": tth.tolist(),
        "Itth": (100 + 10 * np.exp(-((tth - 5) ** 2))).tolist(),
        "Itth_weights": (np.ones_like(tth) / 100.0).tolist(),
    }
    # Remove LaB6 to avoid confusion
    del r["payload"]["phases"]["LaB6"]

    br = build_project(r, tmp_path)
    names = {m.parameter_name for m in br.manifest}
    # Tetragonal: a=b (group), c independent; all angles fixed at 90
    assert "0::a" in names  # representative for (a,b) group
    assert "0::c" in names  # independent
    assert "0::b" not in names  # tied to a
    assert "0::gamma" not in names  # fixed by symmetry
    assert any("fixed by symmetry" in w for w in br.warnings)  # gamma warning


def test_slug_collision_with_three_phases(tmp_path):
    """Three phases with identical base slugs get distinct suffixes."""
    r = base_recipe()
    # Create three phases that all map to "lab6"
    for name in ["Lab6", "LAB6"]:
        r["payload"]["phases"][name] = {
            "structure": {
                "phase_name": name,
                "space_group": "P m -3 m",
                "unit_cell": {"a": 4.15, "b": 4.15, "c": 4.15,
                              "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
                "atoms": {"La": {"element": "La", "x": 0.0, "y": 0.0, "z": 0.0,
                                 "occupancy": 1.0, "Uiso": 0.01, "ADP": "Uiso"}}
            },
            "parameterization": {
                "atoms": {"La": {k: [None, False, None, None]
                                 for k in ("x", "y", "z", "occupancy", "Uiso")}},
                "scale": [1, True, None, None],
                "unit_cell": {k: [None, False, None, None]
                              for k in ("a", "b", "c", "alpha", "beta", "gamma")},
            },
        }
    tth = np.linspace(1.0, 15.0, 200)
    r["payload"]["xrd_data"] = {
        "tth": tth.tolist(),
        "Itth": (100 + 10 * np.exp(-((tth - 5) ** 2))).tolist(),
        "Itth_weights": (np.ones_like(tth) / 100.0).tolist(),
    }

    br = build_project(r, tmp_path)
    slugs = set(br.phase_slugs.values())
    # All three should have distinct slugs
    assert slugs == {"lab6", "lab6_2", "lab6_3"}
    assert len(br.phase_slugs) == 3


def test_bounds_from_non_representative_axis(tmp_path):
    """Bounds from a non-representative axis are used when it's the only flagged one."""
    r = rich_recipe()
    # Flag only 'b' with bounds, not 'a' (cubic: b tied to a)
    r["payload"]["phases"]["LaB6"]["parameterization"]["unit_cell"]["a"] = [None, False, None, None]
    r["payload"]["phases"]["LaB6"]["parameterization"]["unit_cell"]["b"] = [None, True, 4.0, 5.0]

    br = build_project(r, tmp_path)
    names = {m.parameter_name for m in br.manifest}
    # Should free "a" (representative) with bounds from "b"
    assert "0::a" in names
    assert "0::b" not in names
    a_entry = next(m for m in br.manifest if m.parameter_name == "0::a")
    assert a_entry.parameter.fit_min == pytest.approx(4.0)
    assert a_entry.parameter.fit_max == pytest.approx(5.0)


def test_data_outside_fit_range(tmp_path):
    """Data extending outside fit_range is correctly masked in .xye file."""
    r = base_recipe()
    # Data extends beyond fit_range
    tth = np.linspace(0.5, 16.0, 300)
    r["payload"]["xrd_data"] = {
        "tth": tth.tolist(),
        "Itth": (100 + 10 * np.exp(-((tth - 5) ** 2))).tolist(),
        "Itth_weights": (np.ones_like(tth) / 100.0).tolist(),
    }
    r["payload"]["fit_range"] = [1.0, 15.0]

    br = build_project(r, tmp_path)
    # Verify mask is partial
    assert br.mask.sum() < 300
    # Verify .xye file has exactly mask.sum() rows
    xye_data = np.loadtxt(tmp_path / "easydiff_data.xye")
    assert xye_data.shape[0] == br.mask.sum()
