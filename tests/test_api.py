"""API-based tests for the powderline public programmatic interface.

These tests complement the CLI-based regression suite by exercising the
``powderline.run()``, ``powderline.validate()``, and ``powderline.GSASClient``
entry points directly, confirming that the API and CLI paths produce
equivalent outputs.
"""

import json
import uuid
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from pathlib import Path
from pydantic import ValidationError

import powderline
from powderline import RecipeModel
from subprocess_utils import unit_cell_match, extract_rwp_from_lst, REFINED_PARAM_COLUMNS


# ─── Constants (must match test_example_LaB6_regression.py tolerances) ────────

RWP_TOLERANCE = 0.1          # percent — same as CLI regression test
REFERENCE_DIR = Path("examples/example_LaB6/output")


# ─── Shared LaB6 subprocess run ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def lab6_subprocess_run(tmp_path_factory):
    """Run the LaB6 subprocess refinement ONCE and share it across read-only tests.

    Returns ``(result, output_dir)``. Tests that only inspect a successful run's
    result dict / output files consume this instead of each launching their own
    (slow) subprocess refinement. Tests that need a fresh or second independent
    run (e.g. run_id uniqueness) still call powderline.run() themselves.
    """
    recipe = json.loads(Path("examples/example_LaB6/input.json").read_text())
    output_dir = tmp_path_factory.mktemp("lab6_shared_run")
    result = powderline.run(recipe, output_dir, execution_mode="subprocess")
    return result, output_dir


# ─── validate() tests ─────────────────────────────────────────────────────────

def test_validate_valid_recipe(recipe_LaB6_dict):
    """validate() accepts a valid dict and returns a RecipeModel."""
    result = powderline.validate(recipe_LaB6_dict)
    assert isinstance(result, RecipeModel)
    assert result.schema_name == "GSASII_Rietveld"


def test_validate_returns_recipemodel_from_model(recipe_LaB6_model):
    """validate() is idempotent — a RecipeModel passes through unchanged."""
    result = powderline.validate(recipe_LaB6_model)
    assert isinstance(result, RecipeModel)


def test_validate_invalid_recipe_raises():
    """validate() raises ValidationError for a recipe missing required fields."""
    bad = {"schema_name": "GSASII_Rietveld", "schema_version": "0.26.0"}
    with pytest.raises(ValidationError):
        powderline.validate(bad)


def test_validate_empty_dict_raises():
    """validate() raises ValidationError for an empty dict."""
    with pytest.raises(ValidationError):
        powderline.validate({})


# ─── run() validate_only tests ────────────────────────────────────────────────

def test_run_validate_only_returns_summary(recipe_LaB6_dict, tmp_path):
    """run(validate_only=True) returns a summary dict without executing a refinement."""
    result = powderline.run(recipe_LaB6_dict, tmp_path / "out", validate_only=True)

    assert result["success"] is True
    assert result["method"] == "validate_only"
    assert result["schema_name"] == "GSASII_Rietveld"
    assert isinstance(result["phases"], int)
    assert result["rwp"] is None

    # No run was executed, so no run_id should be present.
    assert "run_id" not in result, (
        "validate_only mode must not include run_id — no refinement was executed"
    )

    # No output files should have been written
    gpx = tmp_path / "out" / "dummy.gpx"
    assert not gpx.exists(), "dummy.gpx must not be created in validate_only mode"


def test_run_validate_only_bad_recipe_raises(tmp_path):
    """run(validate_only=True) still raises ValidationError for a bad recipe."""
    with pytest.raises(ValidationError):
        powderline.run({"schema_name": "GSASII_Rietveld"}, tmp_path / "out", validate_only=True)


# ─── run() subprocess mode tests ──────────────────────────────────────────────


def test_run_result_has_run_id(lab6_subprocess_run):
    """run() result includes a valid UUID4 run_id for every actual refinement."""
    result, _ = lab6_subprocess_run
    assert result["success"] is True, f"Refinement failed: {result.get('error')}"
    assert "run_id" in result, "result dict must contain 'run_id'"
    # Raises ValueError if not a valid UUID4
    parsed = uuid.UUID(result["run_id"], version=4)
    assert str(parsed) == result["run_id"], "run_id must be a canonical UUID4 string"


def test_run_result_run_id_is_unique(recipe_LaB6_dict, tmp_path):
    """Each call to run() produces a distinct run_id."""
    result_a = powderline.run(
        recipe_LaB6_dict,
        tmp_path / "out_a",
        execution_mode="subprocess",
    )
    result_b = powderline.run(
        recipe_LaB6_dict,
        tmp_path / "out_b",
        execution_mode="subprocess",
    )
    assert result_a["success"] is True
    assert result_b["success"] is True
    assert result_a["run_id"] != result_b["run_id"], (
        "Two separate runs must produce different run_ids"
    )

def test_run_subprocess_succeeds(lab6_subprocess_run):
    """run(execution_mode='subprocess') completes successfully."""
    result, _ = lab6_subprocess_run
    assert result["success"] is True, f"Refinement failed: {result.get('error')}"
    assert result["method"] == "subprocess"
    assert result["elapsed_time"] > 0


def test_run_subprocess_matches_cli_rwp(lab6_subprocess_run):
    """run() subprocess Rwp matches the committed reference within tolerance."""
    result, _ = lab6_subprocess_run
    assert result["success"] is True, f"Refinement failed: {result.get('error')}"
    assert result.get("rwp") is not None, "No Rwp returned from subprocess run"

    ref_rwp = extract_rwp_from_lst(REFERENCE_DIR / "dummy.lst")
    assert abs(result["rwp"] - ref_rwp) < RWP_TOLERANCE, (
        f"API Rwp ({result['rwp']:.3f}%) differs from CLI reference "
        f"({ref_rwp:.3f}%) by more than {RWP_TOLERANCE}%"
    )


def test_run_subprocess_output_files_exist(lab6_subprocess_run):
    """run() subprocess creates all expected output files."""
    result, out = lab6_subprocess_run
    assert result["success"] is True

    for fname in ["dummy.gpx", "dummy.lst", "LaB6_unit_cell_report.csv",
                  "LaB6_peak_list_report.csv", "fit_profile.txt"]:
        assert (out / fname).exists(), f"Expected output file missing: {fname}"


def test_run_subprocess_unit_cell_matches_reference(lab6_subprocess_run):
    """run() subprocess unit_cell CSV matches the committed reference."""
    result, out = lab6_subprocess_run
    assert result["success"] is True

    test_df = pd.read_csv(out / "LaB6_unit_cell_report.csv")
    ref_df = pd.read_csv(REFERENCE_DIR / "LaB6_unit_cell_report.csv")
    assert unit_cell_match(test_df, ref_df), (
        f"Unit cell report mismatch:\nTest:\n{test_df}\nReference:\n{ref_df}"
    )


# ─── run() structured output tests (all modes return identical structure) ──────

def test_run_result_has_structured_data_keys(lab6_subprocess_run):
    """run() result dict always contains the expected structured-data keys."""
    result, _ = lab6_subprocess_run
    assert result["success"] is True
    for key in ("fit_profile", "unit_cell_data", "peak_list_data", "refined_parameters"):
        assert key in result, f"Expected key '{key}' missing from result dict"


def test_run_subprocess_structured_data_is_dataframes(lab6_subprocess_run):
    """run(execution_mode='subprocess') returns non-empty DataFrames in result."""
    result, _ = lab6_subprocess_run
    assert result["success"] is True

    assert isinstance(result["fit_profile"], pd.DataFrame)
    assert not result["fit_profile"].empty, "fit_profile DataFrame must not be empty"

    assert isinstance(result["unit_cell_data"], dict)
    assert len(result["unit_cell_data"]) > 0, "unit_cell_data dict must not be empty"
    for phase, df in result["unit_cell_data"].items():
        assert isinstance(df, pd.DataFrame), f"unit_cell_data['{phase}'] must be a DataFrame"
        assert not df.empty

    assert isinstance(result["peak_list_data"], dict)
    assert len(result["peak_list_data"]) > 0, "peak_list_data dict must not be empty"
    for phase, df in result["peak_list_data"].items():
        assert isinstance(df, pd.DataFrame), f"peak_list_data['{phase}'] must be a DataFrame"
        assert not df.empty

    assert isinstance(result["refined_parameters"], pd.DataFrame)
    assert not result["refined_parameters"].empty, "refined_parameters DataFrame must not be empty"


def test_run_refined_parameters_has_nine_columns(lab6_subprocess_run):
    """refined_parameters DataFrame contains all 9 expected columns in all modes."""
    expected_cols = set(REFINED_PARAM_COLUMNS)
    result, _ = lab6_subprocess_run
    assert result["success"] is True
    actual_cols = set(result["refined_parameters"].columns)
    assert expected_cols == actual_cols, (
        f"refined_parameters columns mismatch.\nExpected: {sorted(expected_cols)}\nGot: {sorted(actual_cols)}"
    )


def test_run_auto_fallback_returns_full_structured_data(recipe_LaB6_dict, tmp_path):
    """GSASClient fallback (server down) returns full structured DataFrames identical to subprocess mode."""
    client = powderline.GSASClient(fallback_to_subprocess=True)
    result = client.submit_simulation(
        recipe=recipe_LaB6_dict,
        output_dir=tmp_path / "out",
        auto_start_server=False,  # force fallback path
    )
    # After run() DataFrame conversion the result should have the same structure
    # as any other mode — convert manually since submit_simulation() bypasses run()
    result["fit_profile"] = pd.DataFrame(result.get("fit_profile") or {})
    result["unit_cell_data"] = {
        k: pd.DataFrame(v) for k, v in result.get("unit_cell_data", {}).items()
    }
    result["peak_list_data"] = {
        k: pd.DataFrame(v) for k, v in result.get("peak_list_data", {}).items()
    }
    refined_raw = result.get("refined_parameters")
    result["refined_parameters"] = (
        pd.DataFrame(refined_raw) if refined_raw
        else pd.DataFrame(columns=list(REFINED_PARAM_COLUMNS))
    )

    assert result["success"] is True
    assert result.get("rwp") is not None, "rwp must be present in fallback result"
    assert isinstance(result["fit_profile"], pd.DataFrame)
    assert not result["fit_profile"].empty
    assert isinstance(result["unit_cell_data"], dict) and len(result["unit_cell_data"]) > 0
    assert isinstance(result["refined_parameters"], pd.DataFrame)
    assert not result["refined_parameters"].empty


# ─── GSASClient direct dict/model input tests ─────────────────────────────────

def test_gsasclient_accepts_dict(recipe_LaB6_dict, tmp_path):
    """GSASClient.submit_simulation() accepts a plain dict (no file required)."""
    client = powderline.GSASClient(fallback_to_subprocess=True)
    result = client.submit_simulation(
        recipe=recipe_LaB6_dict,
        output_dir=tmp_path / "out",
        auto_start_server=False,  # force subprocess for determinism in CI
    )
    assert result["success"] is True, f"GSASClient dict submission failed: {result.get('error')}"


def test_gsasclient_accepts_recipemodel(recipe_LaB6_model, tmp_path):
    """GSASClient.submit_simulation() accepts a RecipeModel instance."""
    client = powderline.GSASClient(fallback_to_subprocess=True)
    result = client.submit_simulation(
        recipe=recipe_LaB6_model,
        output_dir=tmp_path / "out",
        auto_start_server=False,
    )
    assert result["success"] is True, f"GSASClient model submission failed: {result.get('error')}"


def test_gsasclient_accepts_path(tmp_path):
    """GSASClient.submit_simulation() still accepts a Path (backward compat)."""
    client = powderline.GSASClient(fallback_to_subprocess=True)
    result = client.submit_simulation(
        recipe=Path("examples/example_LaB6/input.json"),
        output_dir=tmp_path / "out",
        auto_start_server=False,
    )
    assert result["success"] is True, f"GSASClient path submission failed: {result.get('error')}"


# ─── load_recipe_asset tests ──────────────────────────────────────────────────

def test_load_recipe_asset_json():
    """load_recipe_asset() loads a JSON file and returns a dict."""
    recipe = powderline.load_recipe_asset(Path("examples/example_LaB6/input.json"))
    assert isinstance(recipe, dict)
    assert "schema_name" in recipe
    assert "payload" in recipe


def test_load_recipe_asset_unsupported_format(tmp_path):
    """load_recipe_asset() raises ValueError for unsupported file formats."""
    bad_file = tmp_path / "recipe.xyz"
    bad_file.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported recipe format"):
        powderline.load_recipe_asset(bad_file)


# ─── error key contract tests ──────────────────────────────────────────────────

def test_run_success_result_has_error_key_none(lab6_subprocess_run):
    """run() on success always includes 'error': None so callers can use result['error']."""
    result, _ = lab6_subprocess_run
    assert result["success"] is True
    assert "error" in result, "'error' key must always be present in run() result"
    assert result["error"] is None, "On success, result['error'] must be None"


def test_run_failure_result_normalization(recipe_LaB6_dict, tmp_path):
    """run() correctly normalises an early-exit failure from run_refinement().

    When run_refinement() returns an early-exit dict (e.g. histogram load fails)
    with only a subset of keys, run() must not raise a KeyError and must still
    produce all expected output keys as empty DataFrames.
    """
    minimal_failure = {
        "success": False,
        "run_id": "test-run-id",
        "error": "Simulated histogram load failure",
        "elapsed_time": 0.5,
        "method": "subprocess",
    }

    with patch("powderline.kicker.run_refinement", return_value=minimal_failure):
        result = powderline.run(
            recipe_LaB6_dict,
            tmp_path / "out",
            execution_mode="subprocess",
        )

    assert result["success"] is False
    assert result["error"] == "Simulated histogram load failure"

    # All DataFrame fields must be present and be valid (empty) DataFrames
    assert isinstance(result["fit_profile"], pd.DataFrame)
    assert isinstance(result["unit_cell_data"], dict)
    assert isinstance(result["peak_list_data"], dict)
    assert isinstance(result["refined_parameters"], pd.DataFrame)
    assert isinstance(result["spf_peaks"], pd.DataFrame)
    assert isinstance(result["spf_convergence_diagnostics"], pd.DataFrame)

    # refined_parameters empty DataFrame must preserve the 9-column schema
    expected_cols = set(REFINED_PARAM_COLUMNS)
    assert set(result["refined_parameters"].columns) == expected_cols


# ─── execution_mode='auto' fallback tests ──────────────────────────────────────

def test_run_auto_mode_falls_back_to_subprocess_when_server_unavailable(
    recipe_LaB6_dict, tmp_path
):
    """run(execution_mode='auto') falls back to subprocess when server cannot be started."""
    from powderline import gsas_client as _gc

    with patch.object(_gc.GSASClient, "is_server_available", return_value=False), \
         patch.object(_gc.GSASClient, "_start_server_background", return_value=False):
        result = powderline.run(
            recipe_LaB6_dict,
            tmp_path / "out",
            execution_mode="auto",
        )

    assert result["success"] is True, f"Refinement failed: {result.get('error')}"
    assert result["method"] == "subprocess", (
        f"Expected 'subprocess' fallback but got method='{result['method']}'"
    )


# ─── execution_mode='server' failure normalization ────────────────────────────

def test_run_server_mode_failure_normalizes_result(recipe_LaB6_dict, tmp_path):
    """run(execution_mode='server') failure returns all DataFrame keys as empty DataFrames.

    When the server is unavailable and fallback is disabled, the error result
    must still pass through DataFrame normalization so callers can safely
    access result['fit_profile'], result['refined_parameters'], etc.
    """
    from powderline import gsas_client as _gc

    with patch.object(_gc.GSASClient, "is_server_available", return_value=False), \
         patch.object(_gc.GSASClient, "_start_server_background", return_value=False):
        result = powderline.run(
            recipe_LaB6_dict,
            tmp_path / "out",
            execution_mode="server",
        )

    assert result["success"] is False
    assert result["error"] is not None

    # All DataFrame fields must be present and be valid (empty) DataFrames
    assert isinstance(result["fit_profile"], pd.DataFrame)
    assert isinstance(result["unit_cell_data"], dict)
    assert isinstance(result["peak_list_data"], dict)
    assert isinstance(result["refined_parameters"], pd.DataFrame)
    assert isinstance(result["spf_peaks"], pd.DataFrame)
    assert isinstance(result["spf_convergence_diagnostics"], pd.DataFrame)

    # refined_parameters empty DataFrame must preserve the 9-column schema
    expected_cols = set(REFINED_PARAM_COLUMNS)
    assert set(result["refined_parameters"].columns) == expected_cols


# ─── HTTP retry logic tests ────────────────────────────────────────────────────

def test_submit_to_server_retries_on_connect_error(recipe_LaB6_dict, tmp_path):
    """_submit_to_server() retries up to 3 times on ConnectError before raising."""
    import httpx
    from powderline.gsas_client import GSASClient

    client = GSASClient()
    connect_error = httpx.ConnectError("Simulated connection refused")

    with patch("httpx.post", side_effect=connect_error) as mock_post:
        with pytest.raises(httpx.ConnectError):
            client._submit_to_server(recipe_LaB6_dict, tmp_path / "out")

    assert mock_post.call_count == 3, (
        f"Expected 3 retry attempts on ConnectError, got {mock_post.call_count}"
    )


def test_submit_to_server_does_not_retry_on_http_status_error(
    recipe_LaB6_dict, tmp_path
):
    """_submit_to_server() raises immediately on HTTP status errors (no retry)."""
    import httpx
    from powderline.gsas_client import GSASClient

    client = GSASClient()
    mock_response = MagicMock()
    mock_response.status_code = 500
    http_error = httpx.HTTPStatusError(
        "500 Server Error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.post", side_effect=http_error) as mock_post:
        with pytest.raises(httpx.HTTPStatusError):
            client._submit_to_server(recipe_LaB6_dict, tmp_path / "out")

    assert mock_post.call_count == 1, (
        f"Expected exactly 1 attempt on HTTPStatusError (no retry), got {mock_post.call_count}"
    )


# ─── execution_mode validation tests ─────────────────────────────────────────

def test_run_invalid_execution_mode_raises(recipe_LaB6_dict, tmp_path):
    """run() raises ValueError immediately for an invalid execution_mode."""
    with pytest.raises(ValueError, match="Invalid execution_mode"):
        powderline.run(
            recipe_LaB6_dict,
            tmp_path / "out",
            execution_mode="sever",  # typo — should be 'server'
        )


# ─── SPF Rwp via programmatic API ────────────────────────────────────────────

def test_run_spf_returns_rwp_directly(recipe_LaB6_singlepeakfit_dict, tmp_path):
    """run() for GSASII_SPF returns Rwp in the result dict (not only in stdout).

    This closes the gap where the CLI regression test could not assert Rwp when
    running via the server (stdout is captured server-side). The API always
    returns result['rwp'] regardless of execution backend.
    """
    result = powderline.run(
        recipe_LaB6_singlepeakfit_dict,
        tmp_path / "out",
        execution_mode="subprocess",
    )
    assert result["success"] is True, f"SPF refinement failed: {result.get('error')}"
    assert result.get("rwp") is not None, (
        "GSASII_SPF run must return result['rwp'] from the programmatic API"
    )
    # Observed Rwp 1.805% on the pinned GSAS-II (v5.7.9); band is the same
    # ~±11% convention the original DRX-based test used ([40, 50] around 45.26),
    # absorbing cross-build/BLAS drift without accepting a wrong result.
    assert 1.6 < result["rwp"] < 2.0, (
        f"SPF Rwp ({result['rwp']:.3f}%) outside expected range [1.6, 2.0]"
    )
