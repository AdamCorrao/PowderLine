"""Tests for the SimulationRecipeBuilder (Materials Project simulation pipeline).

Covers schema correctness, config handling (instrument_defaults/phase_defaults
flow-through, partial-config deep merge, strict validation of unknown keys and
invalid values), simulation-mode invariants (all refine flags
locked), and structural fallbacks (Uiso/occupancy defaults + warnings,
occupancy passthrough). GSASClient behavior lives in
test_gsas_client_visibility.py; the mp_simulate CLI in test_mp_simulate_cli.py.
Unit tests need no GSAS-II; the final end-to-end test (build recipe → run
simulation → check output files) does.
"""

import pytest

from powderline.simulation_builder import SimulationRecipeBuilder
from powderline.schema import RecipeModel


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _collect_refine_flags(obj, path=""):
    """Recursively collect all [value, refine_flag, ...] list refine_flags.

    Returns a list of (dotted_path, refine_flag) tuples for every list that
    looks like a RefinementParameter (len >= 2, second element is bool).
    """
    results = []
    if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], bool):
        results.append((path, obj[1]))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_collect_refine_flags(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(_collect_refine_flags(item, f"{path}[{i}]"))
    return results


# ─── Unit tests ──────────────────────────────────────────────────────────────

def test_builder_recipe_passes_schema_validation(mp_lab6_structure_data):
    """Builder output must pass RecipeModel validation without errors.

    This is the primary regression guard — it would have caught every known
    breakage introduced between schema 0.21 and 0.25.2.
    """
    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    # Should not raise ValidationError
    model = RecipeModel(**recipe)
    assert model.schema_name == "GSASII_Rietveld"
    assert model.schema_version == "0.26.0"


def test_builder_recipe_top_level_fields(mp_lab6_structure_data):
    """Top-level recipe dict must have schema_name, schema_version, payload.

    Fields removed in schema 0.25 (sample_name, recipe_description, comment)
    must be absent.
    """
    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    assert recipe["schema_name"] == "GSASII_Rietveld"
    assert recipe["schema_version"] == "0.26.0"
    assert "payload" in recipe

    # Removed in schema 0.25
    assert "sample_name" not in recipe
    assert "recipe_description" not in recipe
    assert "comment" not in recipe


def test_builder_refinement_controls_in_payload(mp_lab6_structure_data):
    """refinement_cycles must live in payload.refinement_controls, not at top level."""
    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    assert "refinement_cycles" not in recipe, (
        "refinement_cycles must not appear at the top level (schema 0.25 moved it into payload)"
    )
    payload = recipe["payload"]
    assert "refinement_controls" in payload
    assert payload["refinement_controls"]["refinement_cycles"] == 1


def test_builder_peak_broadening_field_names(mp_lab6_structure_data):
    """size_broadening and strain_broadening must use schema 0.25 field names.

    Schema 0.25 renamed:
      size   → isotropic_size   (in size_broadening)
      strain → isotropic_strain (in strain_broadening)
    and added model: 'isotropic' to both blocks.
    """
    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    phase_name = mp_lab6_structure_data["formula"]
    pb = recipe["payload"]["phases"][phase_name]["parameterization"]["peak_broadening"]

    size_b = pb["size_broadening"]
    assert "isotropic_size" in size_b, "Expected 'isotropic_size' (renamed from 'size' in schema 0.25)"
    assert "size" not in size_b, "Old key 'size' must not be present"
    assert size_b["model"] == "isotropic"

    strain_b = pb["strain_broadening"]
    assert "isotropic_strain" in strain_b, "Expected 'isotropic_strain' (renamed from 'strain' in schema 0.25)"
    assert "strain" not in strain_b, "Old key 'strain' must not be present"
    assert strain_b["model"] == "isotropic"


def test_builder_all_refine_flags_false(mp_lab6_structure_data):
    """All RefinementParameter refine_flags must be False in simulation mode.

    Simulation (refinement_cycles=1) is a pure forward calculation.
    kicker.py enforces this as a CRITICAL constraint.
    """
    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    flags = _collect_refine_flags(recipe["payload"])
    assert flags, "No RefinementParameter flags found — helper or recipe structure may be wrong"

    true_flags = [(path, flag) for path, flag in flags if flag is True]
    assert not true_flags, (
        f"Found {len(true_flags)} refine_flag(s) set to True in simulation recipe "
        f"(all must be False):\n" +
        "\n".join(f"  {p}" for p, _ in true_flags)
    )


def test_builder_custom_wavelength(mp_lab6_structure_data):
    """Custom wavelength in defaults must propagate to instrument initialization."""
    custom_wavelength = 0.7
    defaults = {"instrument_defaults": {"wavelength": custom_wavelength}}

    builder = SimulationRecipeBuilder(simulation_defaults=defaults)
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    iparm1 = recipe["payload"]["instrument"]["initialization"][0]
    assert iparm1["Lam"][0] == custom_wavelength
    assert iparm1["Lam"][1] == custom_wavelength


def test_builder_phase_defaults_flow_into_recipe(mp_lab6_structure_data):
    """Configured phase_defaults (scale, models, LG_eta, size/strain) must land
    in the recipe, and instrument_defaults.polarization in Iparm1."""
    defaults = {
        "instrument_defaults": {"polarization": 0.95},
        "phase_defaults": {
            "scale": 2.0,
            "Uiso": 0.02,
            "size_broadening": {"isotropic_size": 0.5, "LG_eta": 0.3},
            "strain_broadening": {"isotropic_strain": 1500.0, "LG_eta": 0.7},
        },
    }
    builder = SimulationRecipeBuilder(simulation_defaults=defaults)
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    phase_name = mp_lab6_structure_data["formula"]
    param = recipe["payload"]["phases"][phase_name]["parameterization"]
    assert param["scale"][0] == 2.0
    pb = param["peak_broadening"]
    assert pb["size_broadening"]["model"] == "isotropic"
    assert pb["size_broadening"]["isotropic_size"][0] == 0.5
    assert pb["size_broadening"]["LG_eta"][0] == 0.3
    assert pb["strain_broadening"]["isotropic_strain"][0] == 1500.0
    assert pb["strain_broadening"]["LG_eta"][0] == 0.7

    iparm1 = recipe["payload"]["instrument"]["initialization"][0]
    assert iparm1["Polariz."][0] == 0.95

    # Uiso fallback (MP provides none) comes from phase_defaults
    atoms = recipe["payload"]["phases"][phase_name]["structure"]["atoms"]
    assert all(atom["Uiso"] == 0.02 for atom in atoms.values())

    RecipeModel(**recipe)


def test_builder_partial_config_merges_with_hardcoded_defaults(mp_lab6_structure_data):
    """A partial simulation_defaults dict must be deep-merged over the
    hardcoded defaults, not replace them."""
    builder = SimulationRecipeBuilder(
        simulation_defaults={"instrument_defaults": {"wavelength": 0.1665}})

    assert builder.defaults["instrument_defaults"]["wavelength"] == 0.1665
    # Untouched siblings and sections survive the merge
    assert builder.defaults["instrument_defaults"]["polarization"] == 0.99
    assert builder.defaults["phase_defaults"]["size_broadening"]["isotropic_size"] == 10.0
    assert "data_range" in builder.defaults

    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)
    RecipeModel(**recipe)


def test_builder_rejects_unknown_top_level_key():
    """Unknown top-level keys — including the retired flat layout — must be
    rejected with an error naming the key and the allowed alternatives."""
    with pytest.raises(ValueError, match=r"simulation_defaults\.sample") as excinfo:
        SimulationRecipeBuilder(simulation_defaults={
            "sample": {"crystallite_size": 5.0},
        })
    msg = str(excinfo.value)
    assert "instrument_defaults" in msg  # allowed keys are listed
    assert ".powderline_config.yaml.example" in msg


def test_builder_rejects_unknown_nested_key():
    """Typos inside a section must be rejected with the full key path."""
    with pytest.raises(ValueError,
                       match=r"instrument_defaults\.wavelenght"):
        SimulationRecipeBuilder(simulation_defaults={
            "instrument_defaults": {"wavelenght": 0.7},  # typo
        })


def test_builder_rejects_null_value_from_empty_yaml_key():
    """An empty YAML key parses as None; it must be rejected with the key
    path and a hint, not silently override a built-in default."""
    with pytest.raises(ValueError,
                       match=r"instrument_defaults\.polarization") as excinfo:
        SimulationRecipeBuilder(simulation_defaults={
            "instrument_defaults": {"polarization": None},
        })
    assert "empty YAML key" in str(excinfo.value)


def test_builder_rejects_non_numeric_value():
    with pytest.raises(ValueError,
                       match=r"phase_defaults\.size_broadening\.isotropic_size"):
        SimulationRecipeBuilder(simulation_defaults={
            "phase_defaults": {"size_broadening": {"isotropic_size": "big"}},
        })


def test_builder_rejects_bad_chebyshev_coefficients():
    with pytest.raises(ValueError,
                       match=r"background\.chebyshev_coefficients"):
        SimulationRecipeBuilder(simulation_defaults={
            "background": {"chebyshev_coefficients": [10.0, None]},
        })


def test_builder_warns_on_missing_structural_values(mp_lab6_structure_data, capsys):
    """Missing Uiso (always, for MP data) and missing occupancy must trigger
    a warning naming the affected atoms; provided values must not.

    Both 'missing' shapes are covered: key absent entirely and key present
    with an explicit None — each must receive the default value (never pass
    None through into the recipe)."""
    data = mp_lab6_structure_data
    data["sites"][1]["occupancy"] = None  # B1: present-but-None counts as missing
    data["sites"].append({'element': 'B', 'x': 0.5, 'y': 0.2021, 'z': 0.5,
                          'label': 'B2'})  # B2: key absent entirely

    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(data)
    out = capsys.readouterr().out

    assert "no Uiso in structure source" in out
    assert "La1" in out and "B1" in out
    assert "no occupancy in structure source" in out
    # Only B1 (explicit None) and B2 (key absent) are missing occupancy —
    # the warning must not list La1 for it
    occ_line = next(l for l in out.splitlines() if "no occupancy" in l)
    assert "B1" in occ_line and "B2" in occ_line and "La1" not in occ_line

    atoms = recipe["payload"]["phases"][data["formula"]]["structure"]["atoms"]
    assert atoms["La1"]["occupancy"] == 1.0  # provided by fixture
    assert atoms["B1"]["occupancy"] == 1.0   # defaulted (explicit None)
    assert atoms["B2"]["occupancy"] == 1.0   # defaulted (key absent)


def test_builder_occupancy_passthrough(mp_lab6_structure_data):
    """Partial occupancies from MP sites must be carried into recipe atoms."""
    data = mp_lab6_structure_data
    data["sites"] = [
        {'element': 'La', 'x': 0.0, 'y': 0.0, 'z': 0.0,
         'occupancy': 0.5, 'label': 'La1'},
        {'element': 'Ce', 'x': 0.0, 'y': 0.0, 'z': 0.0,
         'occupancy': 0.5, 'label': 'Ce1'},
        {'element': 'B', 'x': 0.5, 'y': 0.5, 'z': 0.2021,
         'occupancy': 1.0, 'label': 'B1'},
    ]
    data["is_ordered"] = False

    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(data)

    atoms = recipe["payload"]["phases"][data["formula"]]["structure"]["atoms"]
    assert atoms["La1"]["occupancy"] == 0.5
    assert atoms["Ce1"]["occupancy"] == 0.5
    assert atoms["B1"]["occupancy"] == 1.0
    # Co-occupied site: distinct labels at identical coordinates
    assert atoms["La1"]["x"] == atoms["Ce1"]["x"]

    # Recipe with fractional occupancies must still validate
    RecipeModel(**recipe)


def test_builder_valid_config_does_not_mutate_input(mp_lab6_structure_data):
    """The builder must not modify the caller's simulation_defaults dict
    (it may be a live reference into ConfigLoader.config)."""
    import copy
    defaults = {
        "instrument_defaults": {"wavelength": 0.1665},
        "phase_defaults": {"size_broadening": {"isotropic_size": 2.0}},
    }
    snapshot = copy.deepcopy(defaults)
    builder = SimulationRecipeBuilder(simulation_defaults=defaults)
    builder.build_recipe_from_mp_structure(mp_lab6_structure_data)
    assert defaults == snapshot


# ─── Integration test ─────────────────────────────────────────────────────────

def test_mp_pipeline_end_to_end(mp_lab6_structure_data, tmp_output_dir):
    """Full stack: build recipe → validate → run simulation → check outputs.

    Uses subprocess execution mode to bypass the server requirement.
    Requires GSAS-II to be installed (part of the standard pixi environment).
    """
    import powderline

    builder = SimulationRecipeBuilder()
    recipe = builder.build_recipe_from_mp_structure(mp_lab6_structure_data)

    # Validate before running
    model = RecipeModel(**recipe)

    result = powderline.run(model, output_dir=tmp_output_dir, execution_mode="subprocess")

    assert result is not None
    assert (tmp_output_dir / "fit_profile.txt").exists(), (
        "fit_profile.txt not found — simulation did not complete successfully"
    )
