"""Native ``easydiffraction.rietveld`` schema tests (re/06).

Tests the easydiffraction gateway schema against the rules documented in
src/powderline/gateways/easydiffraction/schema.py module docstring.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from powderline.gateways.easydiffraction.schema import (
    easydiffraction_space_group,
    is_native_recipe,
    validate_recipe,
)
from powderline.schema_core import parameters_requested

DATA = Path(__file__).resolve().parent / "data"


def P(value, flag=False, lo=None, hi=None):
    """Bounded refinable parameter [value, refine_flag, min, max]."""
    return [value, flag, lo, hi]


def _errors(exc) -> list[tuple]:
    return [(e["loc"], e["type"]) for e in exc.value.errors()]


def _fails(recipe, loc, kind=None):
    """Assert that validation fails with an error at the given location and optionally of a given type."""
    with pytest.raises(ValidationError) as exc:
        validate_recipe(recipe)
    errors = _errors(exc)
    assert any(e[0] == loc and (kind is None or e[1] == kind) for e in errors), errors
    return errors


def _load_fixture(name: str) -> dict:
    """Load a fixture from tests/data/easydiffraction/native/."""
    with open(DATA / "easydiffraction" / "native" / name, encoding="utf-8") as f:
        return json.load(f)


# --- valid recipes, round-trip, simulation mode (test 1) ---------------------------


@pytest.mark.parametrize("fixture_name,is_simulation", [
    ("lab6_crysfml_tch.json", False),
    ("lab6_cryspy_pv.json", False),
    ("lab6_slots_simulation.json", True),
])
def test_fixture_validates_and_round_trips(fixture_name, is_simulation):
    """Each fixture validates; model_dump(mode='json') round-trips; simulation_mode matches expectation."""
    recipe = _load_fixture(fixture_name)
    model = validate_recipe(recipe)
    assert validate_recipe(model) is model  # idempotent
    dumped = model.model_dump(mode="json")
    assert validate_recipe(dumped) == model
    assert model.payload.simulation_mode() == is_simulation


# --- is_native_recipe routing (test 2) ----------------------------------------------


def test_is_native_recipe_any_easydiffraction_name():
    assert is_native_recipe({"schema_name": "easydiffraction.rietveld"})
    assert is_native_recipe({"schema_name": "easydiffraction.rietvel"})  # typo counts as native
    assert not is_native_recipe({"schema_name": "gsasii.rietveld"})
    assert not is_native_recipe({"schema_name": "topas.rietveld"})
    assert not is_native_recipe({"schema_name": "0.26.0"})


def test_unknown_easydiffraction_workflow_reported_at_schema_name():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["schema_name"] = "easydiffraction.rietvel"
    _fails(r, ("schema_name",), "schema_name")


# --- space groups (test 3) ----------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("C 1 2/m 1", ("C 2/m", "b1")),
    ("R -3 m:R", ("R -3 m", "r")),
    ("F d -3 m:1", ("F d -3 m", "1")),
])
def test_easydiffraction_space_group_table(name, expected):
    assert easydiffraction_space_group(name) == expected


@pytest.mark.parametrize("bad_setting", ["A 1", "C 4 2 2"])
def test_space_group_not_in_table_rejected(bad_setting):
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["phases"]["LaB6"]["space_group"] = bad_setting
    errors = _fails(r, ("payload", "phases", "LaB6", "space_group"))
    # Should mention EB-42 in the message
    exc_info = None
    with pytest.raises(ValidationError) as exc_info:
        validate_recipe(r)
    error_messages = [e["msg"] for e in exc_info.value.errors()]
    assert any("EB-42" in msg for msg in error_messages)


# --- physical limits (test 4) -------------------------------------------------------


def test_cell_length_outside_physical_limit():
    r = _load_fixture("lab6_crysfml_tch.json")
    # Cubic tie: set all three to 35 so core's cubic constraint is satisfied
    for axis in ["a", "b", "c"]:
        r["payload"]["phases"]["LaB6"]["unit_cell"][axis] = P(35, True)
    _fails(r, ("payload", "phases", "LaB6", "unit_cell", "a"), "easydiffraction_physical_limit")


def test_uiso_below_physical_limit():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["phases"]["LaB6"]["atoms"]["La"]["Uiso"] = P(-0.001, True)
    _fails(r, ("payload", "phases", "LaB6", "atoms", "La", "Uiso"), "easydiffraction_physical_limit")


def test_uiso_above_physical_limit():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["phases"]["LaB6"]["atoms"]["La"]["Uiso"] = P(10.5, True)
    _fails(r, ("payload", "phases", "LaB6", "atoms", "La", "Uiso"), "easydiffraction_physical_limit")


def test_occupancy_above_one_rejected():
    """Occupancy > 1 rejected by core (test that core's check works)."""
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["phases"]["LaB6"]["atoms"]["La"]["occupancy"] = P(1.2, True)
    # Core rejects this, location somewhere in atoms
    with pytest.raises(ValidationError):
        validate_recipe(r)


def test_scale_negative_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["phases"]["LaB6"]["scale"] = P(-1, True)
    _fails(r, ("payload", "phases", "LaB6", "scale"), "easydiffraction_physical_limit")


def test_scale_zero_accepted():
    """scale = 0 is accepted (A161): a phase may contribute nothing."""
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["phases"]["LaB6"]["scale"] = P(0, False)
    validate_recipe(r)


def test_mu_r_negative_rejected():
    r = _load_fixture("lab6_slots_simulation.json")
    r["payload"]["instrument"]["absorption"]["mu_r"] = P(-0.1, False)
    _fails(r, ("payload", "instrument", "absorption", "mu_r"), "easydiffraction_physical_limit")


# --- wavelength and doublet (test 5) ------------------------------------------------


def test_wavelength_zero_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength"] = P(0.0, False)
    _fails(r, ("payload", "instrument", "radiation", "setup_wavelength"), "easydiffraction_positive")


def test_wavelength_negative_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength"] = P(-1.5, False)
    _fails(r, ("payload", "instrument", "radiation", "setup_wavelength"), "easydiffraction_positive")


def test_ka2_wavelength_without_ratio_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2"] = 1.544426
    # Missing setup_wavelength_2_to_1_ratio
    _fails(r, ("payload", "instrument", "radiation", "setup_wavelength_2_to_1_ratio"), "easydiffraction_doublet")


def test_ka2_ratio_without_wavelength_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2_to_1_ratio"] = 0.5
    # Missing setup_wavelength_2
    _fails(r, ("payload", "instrument", "radiation", "setup_wavelength_2"), "easydiffraction_doublet")


def test_ka2_both_present_accepted():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2"] = 1.544426
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2_to_1_ratio"] = 0.5
    validate_recipe(r)


def test_ka2_ratio_zero_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2"] = 1.544426
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2_to_1_ratio"] = 0.0
    with pytest.raises(ValidationError):
        validate_recipe(r)


def test_ka2_ratio_above_one_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2"] = 1.544426
    r["payload"]["instrument"]["radiation"]["setup_wavelength_2_to_1_ratio"] = 1.5
    with pytest.raises(ValidationError):
        validate_recipe(r)


# --- weights (test 6) ---------------------------------------------------------------


def test_heavy_weight_inside_window_rejected():
    """Weight > 1e8 on a point INSIDE fit_range rejected."""
    r = _load_fixture("lab6_crysfml_tch.json")
    # Find a point inside the window (fit_range is [1.0, 15.0])
    tth = r["payload"]["xrd_data"]["tth"]
    idx = None
    for i, t in enumerate(tth):
        if 5.0 <= t <= 10.0:
            idx = i
            break
    assert idx is not None

    r["payload"]["xrd_data"]["Itth_weights"][idx] = 2e8
    _fails(r, ("payload", "xrd_data", "Itth_weights", idx), "easydiffraction_weight")


def test_heavy_weight_outside_window_accepted():
    """Weight > 1e8 on a point OUTSIDE fit_range accepted."""
    r = _load_fixture("lab6_crysfml_tch.json")
    # Find a point outside the window (fit_range is [1.0, 15.0])
    tth = r["payload"]["xrd_data"]["tth"]
    idx = None
    for i, t in enumerate(tth):
        if t > 15.5:
            idx = i
            break
    assert idx is not None

    r["payload"]["xrd_data"]["Itth_weights"][idx] = 2e8
    validate_recipe(r)


def test_weight_exactly_1e8_accepted():
    """Weight exactly 1e8 is accepted."""
    r = _load_fixture("lab6_crysfml_tch.json")
    tth = r["payload"]["xrd_data"]["tth"]
    idx = None
    for i, t in enumerate(tth):
        if 5.0 <= t <= 10.0:
            idx = i
            break
    assert idx is not None

    r["payload"]["xrd_data"]["Itth_weights"][idx] = 1e8
    validate_recipe(r)


# --- peak type vs calculator (test 7) -----------------------------------------------


def test_tch_with_cryspy_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["refinement_controls"]["calculator"] = "cryspy"
    # TCH (cwl-thompson-cox-hastings) not computed by cryspy
    _fails(r, ("payload", "instrument", "broadening", "peak_type"), "easydiffraction_peak_type")


def test_berar_baldinozzi_with_crysfml_rejected():
    r = _load_fixture("lab6_cryspy_pv.json")
    r["payload"]["instrument"]["broadening"]["peak_type"] = "cwl-pseudo-voigt-berar-baldinozzi-asymmetry"
    r["payload"]["instrument"]["broadening"]["parameters"] = {
        "broad_gauss_u": P(0.01), "broad_gauss_v": P(0.01), "broad_gauss_w": P(0.01),
        "broad_lorentz_x": P(0.01), "broad_lorentz_y": P(0.01),
        "asym_beba_a0": P(0.0), "asym_beba_b0": P(0.0),
        "asym_beba_a1": P(0.0), "asym_beba_b1": P(0.0),
    }
    r["payload"]["refinement_controls"]["calculator"] = "crysfml"
    # Bérar-Baldinozzi not computed by crysfml
    _fails(r, ("payload", "instrument", "broadening", "peak_type"), "easydiffraction_peak_type")


def test_pseudo_voigt_with_both_calculators_accepted():
    # cryspy
    r = _load_fixture("lab6_cryspy_pv.json")
    assert r["payload"]["instrument"]["broadening"]["peak_type"] == "cwl-pseudo-voigt"
    assert r["payload"]["refinement_controls"]["calculator"] == "cryspy"
    validate_recipe(r)

    # crysfml
    r2 = copy.deepcopy(r)
    r2["payload"]["refinement_controls"]["calculator"] = "crysfml"
    validate_recipe(r2)


def test_cutoff_fwhm_with_crysfml_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["instrument"]["broadening"]["cutoff_fwhm"] = 10.0
    _fails(r, ("payload", "instrument", "broadening", "cutoff_fwhm"), "easydiffraction_cutoff")


def test_cutoff_fwhm_with_cryspy_accepted():
    r = _load_fixture("lab6_cryspy_pv.json")
    r["payload"]["instrument"]["broadening"]["cutoff_fwhm"] = 10.0
    validate_recipe(r)


def _b_uaniso(r: dict, u11: float = 0.009) -> dict:
    """LaB6's B (6f, x 1/2 1/2) with an anisotropic tensor respecting its site symmetry (U22 = U33, no cross terms)."""
    b = r["payload"]["phases"]["LaB6"]["atoms"]["B"]
    b.pop("Uiso")
    b["ADP"] = "Uaniso"
    b["Uaniso"] = {"U11": P(u11), "U22": P(0.009), "U33": P(0.009), "U12": P(0.0), "U13": P(0.0), "U23": P(0.0)}
    return r


def test_uaniso_with_crysfml_rejected():
    """CrysFML is handed B = 8 pi^2 Ueq only, so the tensor would have no effect (EB-23)."""
    r = _b_uaniso(_load_fixture("lab6_crysfml_tch.json"))
    errors = _fails(r, ("payload", "phases", "LaB6", "atoms", "B", "Uaniso"), "easydiffraction_uaniso_crysfml")
    assert len(errors) == 1


def test_uaniso_with_cryspy_accepted():
    validate_recipe(_b_uaniso(_load_fixture("lab6_cryspy_pv.json"), u11=0.03))


# --- per-type parameter sets (test 8) -----------------------------------------------


def test_tch_missing_asym_fcj_2_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    del r["payload"]["instrument"]["broadening"]["parameters"]["asym_fcj_2"]
    with pytest.raises(ValidationError) as exc:
        validate_recipe(r)
    # Should be missing field error
    errors = _errors(exc)
    assert any("asym_fcj_2" in str(loc) for loc, _ in errors)


def test_pseudo_voigt_with_extra_asym_fcj_1_rejected():
    r = _load_fixture("lab6_cryspy_pv.json")
    r["payload"]["instrument"]["broadening"]["parameters"]["asym_fcj_1"] = P(0.0)
    # extra='forbid' should catch this
    with pytest.raises(ValidationError) as exc:
        validate_recipe(r)
    errors = _errors(exc)
    assert any("asym_fcj_1" in str(loc) for loc, _ in errors)


# --- optional blocks never null (test 9) --------------------------------------------


@pytest.mark.parametrize("path", [
    ("payload", "instrument", "polarization"),
    ("payload", "instrument", "absorption"),
    ("payload", "instrument", "description"),
    ("payload", "background"),
    ("payload", "fit_range"),
    ("payload", "instrument", "corrections", "calib_sample_displacement"),
    ("payload", "instrument", "radiation", "setup_wavelength_2"),
    ("payload", "instrument", "broadening", "cutoff_fwhm"),
])
def test_optional_block_never_null(path):
    r = _load_fixture("lab6_crysfml_tch.json")
    node = r
    for key in path[:-1]:
        if key not in node:
            # Need to add the parent structure
            if path[-1] == "cutoff_fwhm":
                # Already at broadening level
                pass
        node = node.setdefault(key, {}) if isinstance(node, dict) else node

    # Navigate to parent
    parent = r
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = None

    with pytest.raises(ValidationError, match="must not be null; leave it out instead"):
        validate_recipe(r)


# --- refinement_controls (test 10) --------------------------------------------------


def test_minimizer_other_than_lmfit_leastsq_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["refinement_controls"]["minimizer"] = "scipy.optimize.minimize"
    with pytest.raises(ValidationError):
        validate_recipe(r)


def test_max_iterations_zero_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["refinement_controls"]["max_iterations"] = 0
    with pytest.raises(ValidationError):
        validate_recipe(r)


def test_chi_square_change_tolerance_zero_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["refinement_controls"]["chi_square_change_tolerance"] = 0.0
    with pytest.raises(ValidationError):
        validate_recipe(r)


def test_gradient_tolerance_zero_accepted():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["refinement_controls"]["gradient_tolerance"] = 0.0
    validate_recipe(r)


def test_calculator_fullprof_rejected():
    r = _load_fixture("lab6_crysfml_tch.json")
    r["payload"]["refinement_controls"]["calculator"] = "fullprof"
    with pytest.raises(ValidationError):
        validate_recipe(r)


# --- JSON Schema null constraints (test 11) -----------------------------------------


def test_json_schema_shows_null_only_where_core_allows_it():
    """Null only in BoundedRefinableParameter, FitRange, XRDData defs."""
    from powderline.gateways.easydiffraction.schema import EasydiffractionRietveldRecipe

    schema = EasydiffractionRietveldRecipe.model_json_schema()
    allowed = {"BoundedRefinableParameter", "FitRange", "XRDData"}
    for name, node in schema["$defs"].items():
        if name not in allowed:
            assert "null" not in json.dumps(node), name


# --- parameters_requested (test 12) -------------------------------------------------


def test_parameters_requested_counts_cubic_cell_once():
    """Cubic cell a,b,c refined counts ONCE; compute expected from fixture."""
    r = _load_fixture("lab6_crysfml_tch.json")
    model = validate_recipe(r)

    # Count from lab6_crysfml_tch manually:
    # - instrument.broadening.parameters: broad_gauss_u, v, w (3 params, all refined)
    # - instrument.broadening.parameters: broad_lorentz_x, y (2 params, both refined)
    # - instrument.broadening.parameters: asym_fcj_1, asym_fcj_2 (NOT refined in this fixture)
    # - background.chebyshev: num_coefficients=6, refine_flag=True (6 coefficients)
    # - phases.LaB6.unit_cell: cubic a,b,c all refined but tied -> counts as 1
    # - phases.LaB6.scale: refined (1)
    # Total: 3 + 2 + 6 + 1 + 1 = 13

    expected = 13
    actual = parameters_requested(model)
    assert actual == expected, f"Expected {expected} parameters, got {actual}"


# --- fit_limits_on_data and window_mask (test 13) -----------------------------------


def test_window_mask_selects_weighted_points_in_range():
    """window_mask() selects exactly the weighted points with 1.0 <= tth <= 15.0."""
    r = _load_fixture("lab6_cryspy_pv.json")
    model = validate_recipe(r)

    tth = np.asarray(model.payload.xrd_data.tth, dtype=float)
    weights = np.asarray(model.payload.xrd_data.Itth_weights, dtype=float)

    # Expected: weighted points (w > 0) within fit_range [1.0, 15.0]
    expected_mask = (tth >= 1.0) & (tth <= 15.0) & (weights > 0)

    actual_mask = model.payload.window_mask()

    np.testing.assert_array_equal(actual_mask, expected_mask)


# --- settings per calculator (EB-80) -------------------------------------------------


@pytest.mark.parametrize("setting, calculator, suggestion", [
    ("F d -3 m:1", "crysfml", "F d -3 m:2"),
    ("P n n n:1", "crysfml", "P n n n:2"),
    ("R -3 m:R", "cryspy", "R -3 m:H"),
    ("P 1 1 2", "cryspy", "P 1 2 1"),
])
def test_setting_the_calculator_misreads_rejected(setting, calculator, suggestion):
    from powderline.gateways.easydiffraction.schema import calculator_setting_problem

    why = calculator_setting_problem(setting, calculator)
    assert why is not None and "EB-80" in why and repr(suggestion) in why


@pytest.mark.parametrize("setting, calculator", [("P m -3 m", "crysfml"), ("P m -3 m", "cryspy"),
                                                 ("F d -3 m:1", "cryspy"), ("C 1 2/m 1", "crysfml")])
def test_setting_the_calculator_computes_accepted(setting, calculator):
    from powderline.gateways.easydiffraction.schema import calculator_setting_problem

    assert calculator_setting_problem(setting, calculator) is None


def test_setting_rule_reported_at_the_phase():
    r = _load_fixture("lab6_cryspy_pv.json")
    r["payload"]["refinement_controls"]["calculator"] = "crysfml"
    r["payload"]["phases"]["LaB6"]["space_group"] = "P 1 1 2"
    cell = r["payload"]["phases"]["LaB6"]["unit_cell"]
    cell["gamma"] = P(95.0)
    for label, xyz in (("La", (0.0, 0.0, 0.0)), ("B", (0.2021, 0.31, 0.47))):
        for k, v in zip("xyz", xyz):
            r["payload"]["phases"]["LaB6"]["atoms"][label][k] = P(v)
    _fails(r, ("payload", "phases", "LaB6", "space_group"), "easydiffraction_setting")
