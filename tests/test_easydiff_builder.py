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
    assert "0::b" not in names  # cubic: b is symmetry-tied to a; library forces free=False
