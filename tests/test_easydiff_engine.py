"""Result-contract tests for the easydiffraction engine adapter."""
import numpy as np
import pandas as pd
import pytest

ed = pytest.importorskip("easydiffraction")

from powderline.easydiff.engine import run_easydiffraction_recipe

from test_easydiff_builder import rich_recipe
from test_easydiff_policy import base_recipe

RESULT_KEYS = {
    "success", "run_id", "rwp", "r_exp", "gof", "elapsed_time", "method",
    "output_files", "fit_profile", "unit_cell_data", "peak_list_data",
    "refined_parameters", "spf_peaks", "spf_convergence_diagnostics",
    "error", "traceback",
}
REFINED_COLUMNS = ["parameter_name", "descriptive_name", "phase_name", "phase_idx",
                   "atom_name", "atom_idx", "value", "esd", "category"]
PROFILE_COLUMNS = ["two_theta", "y_obs", "y_weights", "y_calc", "y_diff", "y_bkg",
                   "q_values", "d_spacings"]


def test_validate_only_slim_shape(tmp_path):
    result = run_easydiffraction_recipe(base_recipe(), tmp_path, validate_only=True)
    assert result["success"] is True
    assert result["method"] == "validate_only"
    assert result["rwp"] is None
    assert "run_id" not in result
    assert result["phases"] == 1
    assert result["simulation_mode"] is False  # cycles = 5


def test_synthetic_fit_result_contract(tmp_path):
    result = run_easydiffraction_recipe(rich_recipe(), tmp_path)
    assert set(result) == RESULT_KEYS
    assert result["success"] is True, result["error"]
    assert result["method"] == "easydiffraction"
    assert result["rwp"] is not None and 0 < result["rwp"] < 100  # percent
    assert list(result["refined_parameters"].columns) == REFINED_COLUMNS
    assert len(result["refined_parameters"]) == 5  # scale, 2 cheb, U, cell a
    assert list(result["fit_profile"].columns) == PROFILE_COLUMNS
    assert len(result["fit_profile"]) == 200  # full grid
    assert "LaB6" in result["unit_cell_data"]
    assert isinstance(result["spf_peaks"], pd.DataFrame) and result["spf_peaks"].empty
    assert result["error"] is None
    out_names = {p.rsplit("/", 1)[-1] for p in result["output_files"]}
    assert {"refined_parameters.csv", "fit_profile.txt",
            "LaB6_unit_cell_report.csv"} <= out_names


def test_simulation_mode_when_nothing_refined(tmp_path):
    r = rich_recipe()
    pz = r["payload"]["phases"]["LaB6"]["parameterization"]
    pz["scale"] = [1, False, None, None]
    pz["unit_cell"]["a"] = [None, False, None, None]
    r["payload"]["instrument"]["parameterization"]["broadening"]["U"] = [None, False, None, None]
    r["payload"]["background"]["chebyshev"]["refine_flag"] = False
    result = run_easydiffraction_recipe(r, tmp_path)
    assert result["success"] is True
    assert result["method"] == "easydiffraction_simulation"
    assert result["rwp"] is None
    assert result["refined_parameters"].empty
    assert list(result["refined_parameters"].columns) == REFINED_COLUMNS
    assert (result["fit_profile"]["y_calc"] != 0).any()  # pattern actually calculated


def test_unique_run_ids(tmp_path):
    import uuid
    r1 = run_easydiffraction_recipe(rich_recipe(), tmp_path / "a")
    r2 = run_easydiffraction_recipe(rich_recipe(), tmp_path / "b")
    assert uuid.UUID(r1["run_id"]) != uuid.UUID(r2["run_id"])
