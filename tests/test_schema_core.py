"""Direct tests for powderline.schema_core validators.

Pure pydantic, engine-free. Tests the core schema 1.0.0 models that every
engine schema shares (re/03).
"""
import json

import pytest
from pydantic import ValidationError

from powderline.schema_core import (
    CORE_SCHEMA_VERSION,
    METADATA_MAX_BYTES,
    SUPPORTED_CORE_SCHEMAS,
    UNIT_DEG_2THETA,
    ChebyshevBackground,
    CoreRecipe,
    FitRange,
    XRDData,
    check_core_schema_version,
    check_fit_range_within_data,
)


# --- XRDData ----------------------------------------------------------------


def _errors(exc_info) -> list:
    """(location, type) of every error, so a test pins which field failed and why."""
    return [(e["loc"], e["type"]) for e in exc_info.value.errors()]


def test_xrd_arrays_same_length_mismatch_raises():
    with pytest.raises(ValidationError, match="same length"):
        XRDData(tth=[1.0, 2.0, 3.0], Itth=[10.0, 20.0], Itth_weights=[1.0, 1.0])


def test_xrd_arrays_same_length_ok():
    m = XRDData(tth=[1.0, 2.0], Itth=[10.0, 20.0], Itth_weights=[1.0, 1.0])
    assert len(m.tth) == 2


def test_xrd_empty_arrays_raise():
    with pytest.raises(ValidationError, match="non-empty"):
        XRDData(tth=[], Itth=[], Itth_weights=[])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_xrd_non_finite_tth_raises(bad):
    with pytest.raises(ValidationError, match="non-finite"):
        XRDData(tth=[1.0, bad, 3.0], Itth=[10.0, 20.0, 30.0], Itth_weights=[1.0, 1.0, 1.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_xrd_non_finite_intensity_raises(bad):
    with pytest.raises(ValidationError, match="non-finite"):
        XRDData(tth=[1.0, 2.0], Itth=[10.0, bad], Itth_weights=[1.0, 1.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_xrd_non_finite_weight_raises(bad):
    with pytest.raises(ValidationError, match="non-finite"):
        XRDData(tth=[1.0, 2.0], Itth=[10.0, 20.0], Itth_weights=[1.0, bad])


@pytest.mark.parametrize(
    "tth",
    [
        [1.0, 1.0, 2.0],   # equal neighbors -> not strictly increasing
        [1.0, 3.0, 2.0],   # out of order
        [3.0, 2.0, 1.0],   # descending
    ],
)
def test_xrd_non_monotonic_tth_raises(tth):
    with pytest.raises(ValidationError, match="strictly increasing"):
        XRDData(tth=tth, Itth=[10.0, 20.0, 30.0], Itth_weights=[1.0, 1.0, 1.0])


def test_xrd_negative_weight_raises():
    with pytest.raises(ValidationError, match=">= 0"):
        XRDData(tth=[1.0, 2.0], Itth=[10.0, 20.0], Itth_weights=[1.0, -0.5])


def test_xrd_all_zero_weights_raise():
    with pytest.raises(ValidationError, match="all zero"):
        XRDData(tth=[1.0, 2.0], Itth=[10.0, 20.0], Itth_weights=[0.0, 0.0])


def test_xrd_negative_intensity_ok():
    # Background-subtracted data legitimately dips below zero.
    m = XRDData(tth=[1.0, 2.0, 3.0], Itth=[-5.0, 20.0, -1.0], Itth_weights=[1.0, 1.0, 1.0])
    assert m.Itth[0] == -5.0


def test_xrd_extra_key_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        XRDData(tth=[1.0, 2.0], Itth=[10.0, 20.0], Itth_weights=[1.0, 1.0], unknown_field="foo")


def test_xrd_unit_metadata():
    schema = XRDData.model_json_schema()
    assert schema["properties"]["tth"]["unit"] == UNIT_DEG_2THETA


# --- FitRange ---------------------------------------------------------------


def test_fit_range_roundtrip_model_dump():
    fr = FitRange.model_validate([5.0, 40.0])
    assert fr.model_dump() == [5.0, 40.0]


def test_fit_range_roundtrip_model_dump_json():
    fr = FitRange.model_validate([5.0, 40.0])
    assert json.loads(fr.model_dump_json()) == [5.0, 40.0]


def test_fit_range_null_ends_allowed():
    fr = FitRange.model_validate([None, None])
    assert fr.model_dump() == [None, None]


def test_fit_range_one_null_end_ok():
    fr1 = FitRange.model_validate([5.0, None])
    assert fr1.model_dump() == [5.0, None]
    fr2 = FitRange.model_validate([None, 40.0])
    assert fr2.model_dump() == [None, 40.0]


def test_fit_range_wrong_length_rejected():
    with pytest.raises(ValidationError, match="exactly 2 elements"):
        FitRange.model_validate([5.0, 10.0, 15.0])


def test_fit_range_non_list_rejected():
    with pytest.raises(ValidationError, match="must be a \\[min, max\\] list"):
        FitRange.model_validate({"min": 5.0, "max": 10.0})


def test_fit_range_max_equals_min_rejected():
    with pytest.raises(ValidationError, match="must be greater than"):
        FitRange.model_validate([10.0, 10.0])


def test_fit_range_max_less_than_min_rejected():
    with pytest.raises(ValidationError, match="must be greater than"):
        FitRange.model_validate([20.0, 10.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fit_range_non_finite_min_rejected(bad):
    with pytest.raises(ValidationError, match="must be finite"):
        FitRange.model_validate([bad, 40.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fit_range_non_finite_max_rejected(bad):
    with pytest.raises(ValidationError, match="must be finite"):
        FitRange.model_validate([5.0, bad])


def test_fit_range_json_schema():
    schema = FitRange.model_json_schema()
    assert schema["type"] == "array"
    assert schema["prefixItems"] == [{"anyOf": [{"type": "number"}, {"type": "null"}]},
                                      {"anyOf": [{"type": "number"}, {"type": "null"}]}]
    assert len(schema["prefixItems"]) == 2
    assert schema["unit"] == UNIT_DEG_2THETA


# --- check_fit_range_within_data --------------------------------------------


def test_check_fit_range_within_data_none_passes():
    data = XRDData(tth=[10.0, 20.0, 30.0], Itth=[100.0, 200.0, 150.0], Itth_weights=[1.0, 1.0, 1.0])
    check_fit_range_within_data(None, data)  # no error


def test_check_fit_range_within_data_inside_passes():
    data = XRDData(tth=[10.0, 20.0, 30.0], Itth=[100.0, 200.0, 150.0], Itth_weights=[1.0, 1.0, 1.0])
    fr = FitRange.model_validate([15.0, 25.0])
    check_fit_range_within_data(fr, data)  # no error


def test_check_fit_range_within_data_exactly_at_limits_passes():
    data = XRDData(tth=[10.0, 20.0, 30.0], Itth=[100.0, 200.0, 150.0], Itth_weights=[1.0, 1.0, 1.0])
    fr = FitRange.model_validate([10.0, 30.0])
    check_fit_range_within_data(fr, data)  # no error


def test_check_fit_range_within_data_below_min_raises():
    data = XRDData(tth=[10.0, 20.0, 30.0], Itth=[100.0, 200.0, 150.0], Itth_weights=[1.0, 1.0, 1.0])
    fr = FitRange.model_validate([5.0, 25.0])
    with pytest.raises(ValueError, match="min"):
        check_fit_range_within_data(fr, data)


def test_check_fit_range_within_data_above_max_raises():
    data = XRDData(tth=[10.0, 20.0, 30.0], Itth=[100.0, 200.0, 150.0], Itth_weights=[1.0, 1.0, 1.0])
    fr = FitRange.model_validate([15.0, 35.0])
    with pytest.raises(ValueError, match="max"):
        check_fit_range_within_data(fr, data)


def test_check_fit_range_within_data_open_ends_pass():
    data = XRDData(tth=[10.0, 20.0, 30.0], Itth=[100.0, 200.0, 150.0], Itth_weights=[1.0, 1.0, 1.0])
    fr1 = FitRange.model_validate([None, 25.0])
    check_fit_range_within_data(fr1, data)  # no error
    fr2 = FitRange.model_validate([15.0, None])
    check_fit_range_within_data(fr2, data)  # no error
    fr3 = FitRange.model_validate([None, None])
    check_fit_range_within_data(fr3, data)  # no error


# --- ChebyshevBackground ----------------------------------------------------


def test_chebyshev_coefficient_count_mismatch_rejected():
    with pytest.raises(ValidationError, match="must match num_coefficients"):
        ChebyshevBackground(num_coefficients=3, coefficients=[1.0, 2.0], refine_flag=False)


def test_chebyshev_coefficient_count_match_ok():
    m = ChebyshevBackground(num_coefficients=3, coefficients=[1.0, 2.0, 3.0], refine_flag=False)
    assert len(m.coefficients) == 3


def test_chebyshev_num_coefficients_must_be_positive():
    with pytest.raises(ValidationError, match="greater than 0"):
        ChebyshevBackground(num_coefficients=0, coefficients=[], refine_flag=False)


def test_chebyshev_refine_flag_null_rejected():
    with pytest.raises(ValidationError):
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag=None)


def test_chebyshev_refine_flag_string_rejected():
    with pytest.raises(ValidationError):
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag="true")


def test_chebyshev_refine_flag_int_rejected():
    with pytest.raises(ValidationError):
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag=1)


def test_chebyshev_extra_key_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag=False, unknown="bar")


# --- CoreRecipe -------------------------------------------------------------


def test_core_recipe_minimal_valid():
    r = CoreRecipe(
        schema_name="gsasii.rietveld",
        core_schema_version="1.0.0",
        engine_schema_version="1.0.0",
        payload={}
    )
    assert r.schema_name == "gsasii.rietveld"


@pytest.mark.parametrize(
    "bad_name",
    [
        "GSASII_Rietveld",   # uppercase
        "gsasii",            # missing workflow
        "Gsasii.rietveld",   # uppercase first letter
        "gsasii.rietveld.x", # too many dots
        "gsasii.Rietveld",   # uppercase in workflow
        "1gsasii.rietveld",  # starts with digit
    ],
)
def test_core_recipe_schema_name_pattern_rejected(bad_name):
    with pytest.raises(ValidationError, match="String should match pattern"):
        CoreRecipe(
            schema_name=bad_name,
            core_schema_version="1.0.0",
            engine_schema_version="1.0.0",
            payload={}
        )


def test_core_recipe_schema_name_pattern_ok():
    r = CoreRecipe(
        schema_name="topas.rietveld",
        core_schema_version="1.0.0",
        engine_schema_version="1.0.0",
        payload={}
    )
    assert r.schema_name == "topas.rietveld"


def test_core_recipe_core_schema_version_1_1_0_rejected():
    with pytest.raises(ValidationError) as exc_info:
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.1.0",
            engine_schema_version="1.0.0",
            payload={}
        )
    err_msg = str(exc_info.value)
    assert SUPPORTED_CORE_SCHEMAS in err_msg
    assert "PowderLine" in err_msg


def test_core_recipe_core_schema_version_2_0_0_rejected():
    with pytest.raises(ValidationError) as exc_info:
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="2.0.0",
            engine_schema_version="1.0.0",
            payload={}
        )
    err_msg = str(exc_info.value)
    assert SUPPORTED_CORE_SCHEMAS in err_msg
    assert "PowderLine" in err_msg


def test_core_recipe_core_schema_version_invalid_string_rejected():
    with pytest.raises(ValidationError, match="not a valid version"):
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="abc",
            engine_schema_version="1.0.0",
            payload={}
        )


def test_core_recipe_engine_schema_version_invalid_string_rejected():
    with pytest.raises(ValidationError, match="not a valid version"):
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.0.0",
            engine_schema_version="abc",
            payload={}
        )


def test_core_recipe_metadata_none_ok():
    r = CoreRecipe(
        schema_name="gsasii.rietveld",
        core_schema_version="1.0.0",
        engine_schema_version="1.0.0",
        metadata=None,
        payload={}
    )
    assert r.metadata is None


def test_core_recipe_metadata_nested_json_ok():
    r = CoreRecipe(
        schema_name="gsasii.rietveld",
        core_schema_version="1.0.0",
        engine_schema_version="1.0.0",
        metadata={"user": "test", "nested": {"key": "value"}},
        payload={}
    )
    assert r.metadata["nested"]["key"] == "value"


def test_core_recipe_metadata_non_serializable_set_rejected():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.0.0",
            engine_schema_version="1.0.0",
            metadata={"bad": {1, 2, 3}},
            payload={}
        )


def test_core_recipe_metadata_non_serializable_object_rejected():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.0.0",
            engine_schema_version="1.0.0",
            metadata={"bad": object()},
            payload={}
        )


def test_core_recipe_metadata_nan_rejected():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.0.0",
            engine_schema_version="1.0.0",
            metadata={"bad": float("nan")},
            payload={}
        )


def test_core_recipe_metadata_over_size_rejected():
    # Build a string just over METADATA_MAX_BYTES.
    # JSON encoding of {"data": "x" * N} is roughly {"data":"xxx..."}.
    # Start with a string that's definitely over the limit.
    big_string = "x" * (METADATA_MAX_BYTES + 100)
    metadata = {"data": big_string}
    # Verify it's over the limit
    size = len(json.dumps(metadata, allow_nan=False, ensure_ascii=False).encode("utf-8"))
    assert size > METADATA_MAX_BYTES

    with pytest.raises(ValidationError) as exc_info:
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.0.0",
            engine_schema_version="1.0.0",
            metadata=metadata,
            payload={}
        )
    err_msg = str(exc_info.value)
    assert "METADATA_MAX_BYTES" in err_msg or str(METADATA_MAX_BYTES) in err_msg


def test_core_recipe_metadata_exactly_at_limit_ok():
    # Build metadata that's exactly at the limit.
    # Start with a reasonable guess and adjust.
    base_overhead = len(json.dumps({"data": ""}, allow_nan=False, ensure_ascii=False).encode("utf-8"))
    target_string_size = METADATA_MAX_BYTES - base_overhead
    big_string = "x" * target_string_size
    metadata = {"data": big_string}

    # Fine-tune to get exactly at the limit
    while True:
        size = len(json.dumps(metadata, allow_nan=False, ensure_ascii=False).encode("utf-8"))
        if size == METADATA_MAX_BYTES:
            break
        elif size > METADATA_MAX_BYTES:
            big_string = big_string[:-1]
            metadata = {"data": big_string}
        else:
            big_string += "x"
            metadata = {"data": big_string}

    r = CoreRecipe(
        schema_name="gsasii.rietveld",
        core_schema_version="1.0.0",
        engine_schema_version="1.0.0",
        metadata=metadata,
        payload={}
    )
    assert r.metadata == metadata


def test_core_recipe_unknown_key_rejected():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CoreRecipe(
            schema_name="gsasii.rietveld",
            core_schema_version="1.0.0",
            engine_schema_version="1.0.0",
            payload={},
            unknown_field="bad"
        )


def test_core_recipe_multiple_errors_reported():
    # Create a recipe with multiple independent errors
    with pytest.raises(ValidationError) as exc_info:
        CoreRecipe(
            schema_name="INVALID_NAME",  # error 1: pattern mismatch
            core_schema_version="abc",    # error 2: invalid version
            engine_schema_version="xyz",  # error 3: invalid version
            payload={},
            unknown_key="bad"             # error 4: extra field
        )
    # pydantic reports every independent problem in one ValidationError.
    locs = {e["loc"] for e in exc_info.value.errors()}
    assert locs == {("schema_name",), ("core_schema_version",),
                    ("engine_schema_version",), ("unknown_key",)}


# --- check_core_schema_version ----------------------------------------------


def test_check_core_schema_version_accepts_current():
    check_core_schema_version(CORE_SCHEMA_VERSION)  # no error


def test_check_core_schema_version_rejects_unsupported():
    with pytest.raises(ValueError, match="not supported"):
        check_core_schema_version("2.0.0")


# --- parameter models (RefinableParameter, BoundedRefinableParameter, etc.) ----


from powderline.schema_core import (
    BoundedRefinableParameter,
    RefinableParameter,
    StructureBoundedRefinableParameter,
    StructureRefinableParameter,
    check_within_bounds,
)
from pydantic import BaseModel


# Wrapper models for testing error locs
class _WrapperRP(BaseModel):
    """Wrapper for RefinableParameter testing."""
    a: RefinableParameter


class _WrapperBRP(BaseModel):
    """Wrapper for BoundedRefinableParameter testing."""
    a: BoundedRefinableParameter


class _WrapperSRP(BaseModel):
    """Wrapper for StructureRefinableParameter testing."""
    a: StructureRefinableParameter


class _WrapperSBRP(BaseModel):
    """Wrapper for StructureBoundedRefinableParameter testing."""
    a: StructureBoundedRefinableParameter


# --- 1. JSON shape round-trip -----------------------------------------------


def test_refinable_parameter_model_dump():
    rp = RefinableParameter.model_validate([1.5, True])
    assert rp.model_dump() == [1.5, True]


def test_refinable_parameter_model_dump_json():
    rp = RefinableParameter.model_validate([1.5, True])
    assert json.loads(rp.model_dump_json()) == [1.5, True]


def test_bounded_refinable_parameter_model_dump():
    brp = BoundedRefinableParameter.model_validate([1.5, True, 0.0, 2.0])
    assert brp.model_dump() == [1.5, True, 0.0, 2.0]


def test_bounded_refinable_parameter_model_dump_json():
    brp = BoundedRefinableParameter.model_validate([1.5, True, 0.0, 2.0])
    assert json.loads(brp.model_dump_json()) == [1.5, True, 0.0, 2.0]


def test_structure_refinable_parameter_model_dump():
    srp = StructureRefinableParameter.model_validate([None, True])
    assert srp.model_dump() == [None, True]


def test_structure_refinable_parameter_model_dump_with_value():
    srp = StructureRefinableParameter.model_validate([1.5, False])
    assert srp.model_dump() == [1.5, False]


def test_structure_bounded_refinable_parameter_model_dump():
    sbrp = StructureBoundedRefinableParameter.model_validate([None, True, 0.0, 2.0])
    assert sbrp.model_dump() == [None, True, 0.0, 2.0]


def test_structure_bounded_refinable_parameter_model_dump_with_value():
    sbrp = StructureBoundedRefinableParameter.model_validate([1.5, True, 0.0, 2.0])
    assert sbrp.model_dump() == [1.5, True, 0.0, 2.0]


# --- 2. RefinableParameter validation ---------------------------------------


def test_refinable_parameter_4_element_list_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a=[1.5, True, 0.0, 2.0])
    err = exc_info.value
    assert any("bounds are not supported" in e["msg"] for e in err.errors())
    assert any(e["loc"] == ("a",) for e in err.errors())


def test_refinable_parameter_1_element_list_rejected():
    with pytest.raises(ValidationError, match="expected \\[value, refine_flag\\]"):
        _WrapperRP(a=[1.5])


def test_refinable_parameter_3_element_list_rejected():
    with pytest.raises(ValidationError, match="expected \\[value, refine_flag\\]"):
        _WrapperRP(a=[1.5, True, 0.0])


def test_refinable_parameter_dict_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a={"value": 1.5, "refine_flag": True})
    err = exc_info.value
    assert any("got dict" in e["msg"] for e in err.errors())


def test_refinable_parameter_scalar_rejected():
    with pytest.raises(ValidationError, match="expected a \\[value, refine_flag\\] list"):
        _WrapperRP(a=1.5)


def test_refinable_parameter_null_value_rejected():
    with pytest.raises(ValidationError):
        _WrapperRP(a=[None, True])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_refinable_parameter_non_finite_value_rejected(bad):
    with pytest.raises(ValidationError, match="must be finite"):
        _WrapperRP(a=[bad, True])


def test_refinable_parameter_refine_flag_null_rejected():
    with pytest.raises(ValidationError):
        _WrapperRP(a=[1.5, None])


def test_refinable_parameter_refine_flag_string_rejected():
    with pytest.raises(ValidationError):
        _WrapperRP(a=[1.5, "true"])


def test_refinable_parameter_refine_flag_int_1_rejected():
    with pytest.raises(ValidationError):
        _WrapperRP(a=[1.5, 1])


def test_refinable_parameter_refine_flag_int_0_rejected():
    with pytest.raises(ValidationError):
        _WrapperRP(a=[1.5, 0])


def test_refinable_parameter_int_value_accepted_and_dumped_as_float():
    w = _WrapperRP(a=[2, True])
    assert w.model_dump()["a"] == [2.0, True]


# --- 3. BoundedRefinableParameter validation --------------------------------


def test_bounded_refinable_parameter_2_element_list_rejected():
    with pytest.raises(ValidationError, match="expected \\[value, refine_flag, min, max\\]"):
        _WrapperBRP(a=[1.5, True])


def test_bounded_refinable_parameter_null_min_accepted():
    brp = BoundedRefinableParameter.model_validate([1.5, True, None, 2.0])
    assert brp.min is None
    assert brp.max == 2.0


def test_bounded_refinable_parameter_null_max_accepted():
    brp = BoundedRefinableParameter.model_validate([1.5, True, 0.0, None])
    assert brp.min == 0.0
    assert brp.max is None


def test_bounded_refinable_parameter_null_both_bounds_accepted():
    brp = BoundedRefinableParameter.model_validate([1.5, True, None, None])
    assert brp.min is None
    assert brp.max is None


def test_bounded_refinable_parameter_min_greater_than_max_rejected():
    with pytest.raises(ValidationError, match="min .* must be <= max"):
        _WrapperBRP(a=[1.5, True, 2.0, 1.0])


def test_bounded_refinable_parameter_value_below_min_rejected():
    with pytest.raises(ValidationError, match="below min"):
        _WrapperBRP(a=[0.5, True, 1.0, 2.0])


def test_bounded_refinable_parameter_value_above_max_rejected():
    with pytest.raises(ValidationError, match="above max"):
        _WrapperBRP(a=[2.5, True, 1.0, 2.0])


def test_bounded_refinable_parameter_value_equals_min_accepted():
    brp = BoundedRefinableParameter.model_validate([1.0, True, 1.0, 2.0])
    assert brp.value == 1.0


def test_bounded_refinable_parameter_value_equals_max_accepted():
    brp = BoundedRefinableParameter.model_validate([2.0, True, 1.0, 2.0])
    assert brp.value == 2.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_bounded_refinable_parameter_non_finite_min_rejected(bad):
    with pytest.raises(ValidationError, match="must be finite"):
        _WrapperBRP(a=[1.5, True, bad, 2.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_bounded_refinable_parameter_non_finite_max_rejected(bad):
    with pytest.raises(ValidationError, match="must be finite"):
        _WrapperBRP(a=[1.5, True, 0.0, bad])


def test_bounded_refinable_parameter_null_value_rejected():
    with pytest.raises(ValidationError):
        _WrapperBRP(a=[None, True, 0.0, 2.0])


# --- 4. StructureRefinableParameter / StructureBoundedRefinableParameter ----


def test_structure_refinable_parameter_null_value_accepted_refine_true():
    srp = StructureRefinableParameter.model_validate([None, True])
    assert srp.value is None
    assert srp.refine_flag is True


def test_structure_refinable_parameter_null_value_accepted_refine_false():
    srp = StructureRefinableParameter.model_validate([None, False])
    assert srp.value is None
    assert srp.refine_flag is False


def test_structure_refinable_parameter_4_element_list_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperSRP(a=[1.5, True, 0.0, 2.0])
    err = exc_info.value
    assert any("bounds are not supported" in e["msg"] for e in err.errors())


def test_structure_refinable_parameter_refine_flag_null_rejected():
    with pytest.raises(ValidationError):
        _WrapperSRP(a=[None, None])


def test_structure_bounded_refinable_parameter_null_value_accepted():
    sbrp = StructureBoundedRefinableParameter.model_validate([None, True, 0.0, 2.0])
    assert sbrp.value is None
    assert sbrp.refine_flag is True


def test_structure_bounded_refinable_parameter_null_value_min_max_order_enforced():
    with pytest.raises(ValidationError, match="min .* must be <= max"):
        _WrapperSBRP(a=[None, True, 2.0, 1.0])


def test_structure_bounded_refinable_parameter_non_null_value_enforces_bounds_below():
    with pytest.raises(ValidationError, match="below min"):
        _WrapperSBRP(a=[0.5, True, 1.0, 2.0])


def test_structure_bounded_refinable_parameter_non_null_value_enforces_bounds_above():
    with pytest.raises(ValidationError, match="above max"):
        _WrapperSBRP(a=[2.5, True, 1.0, 2.0])


def test_structure_bounded_refinable_parameter_refine_flag_null_rejected():
    with pytest.raises(ValidationError):
        _WrapperSBRP(a=[None, None, 0.0, 2.0])


# --- 5. check_within_bounds -------------------------------------------------


def test_check_within_bounds_inside_passes():
    param = BoundedRefinableParameter.model_validate([1.5, True, 1.0, 2.0])
    check_within_bounds(1.5, param, "test_field")  # no error


def test_check_within_bounds_on_min_passes():
    param = BoundedRefinableParameter.model_validate([1.0, True, 1.0, 2.0])
    check_within_bounds(1.0, param, "test_field")  # no error


def test_check_within_bounds_on_max_passes():
    param = BoundedRefinableParameter.model_validate([2.0, True, 1.0, 2.0])
    check_within_bounds(2.0, param, "test_field")  # no error


def test_check_within_bounds_open_min_passes():
    param = BoundedRefinableParameter.model_validate([0.5, True, None, 2.0])
    check_within_bounds(0.5, param, "test_field")  # no error


def test_check_within_bounds_open_max_passes():
    param = BoundedRefinableParameter.model_validate([2.5, True, 1.0, None])
    check_within_bounds(2.5, param, "test_field")  # no error


def test_check_within_bounds_open_both_passes():
    param = BoundedRefinableParameter.model_validate([100.0, True, None, None])
    check_within_bounds(100.0, param, "test_field")  # no error


def test_check_within_bounds_below_min_raises():
    param = BoundedRefinableParameter.model_validate([1.5, True, 1.0, 2.0])
    with pytest.raises(ValueError, match="test_field.*below min"):
        check_within_bounds(0.5, param, "test_field")


def test_check_within_bounds_above_max_raises():
    param = BoundedRefinableParameter.model_validate([1.5, True, 1.0, 2.0])
    with pytest.raises(ValueError, match="test_field.*above max"):
        check_within_bounds(2.5, param, "test_field")


# --- 6. JSON schema ---------------------------------------------------------


def test_refinable_parameter_json_schema():
    schema = _WrapperRP.model_json_schema()
    param_schema = schema["$defs"]["RefinableParameter"]
    assert param_schema["type"] == "array"
    assert len(param_schema["prefixItems"]) == 2
    assert param_schema["minItems"] == 2
    assert param_schema["maxItems"] == 2
    assert param_schema["prefixItems"][0] == {"type": "number"}
    assert param_schema["prefixItems"][1] == {"type": "boolean"}


def test_bounded_refinable_parameter_json_schema():
    schema = _WrapperBRP.model_json_schema()
    param_schema = schema["$defs"]["BoundedRefinableParameter"]
    assert param_schema["type"] == "array"
    assert len(param_schema["prefixItems"]) == 4
    assert param_schema["minItems"] == 4
    assert param_schema["maxItems"] == 4
    assert param_schema["prefixItems"][0] == {"type": "number"}
    assert param_schema["prefixItems"][1] == {"type": "boolean"}
    assert param_schema["prefixItems"][2] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert param_schema["prefixItems"][3] == {"anyOf": [{"type": "number"}, {"type": "null"}]}


def test_structure_refinable_parameter_json_schema():
    schema = _WrapperSRP.model_json_schema()
    param_schema = schema["$defs"]["StructureRefinableParameter"]
    assert param_schema["type"] == "array"
    assert len(param_schema["prefixItems"]) == 2
    assert param_schema["minItems"] == 2
    assert param_schema["maxItems"] == 2
    assert param_schema["prefixItems"][0] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert param_schema["prefixItems"][1] == {"type": "boolean"}
    assert "structure" in param_schema["description"].lower()


def test_structure_bounded_refinable_parameter_json_schema():
    schema = _WrapperSBRP.model_json_schema()
    param_schema = schema["$defs"]["StructureBoundedRefinableParameter"]
    assert param_schema["type"] == "array"
    assert len(param_schema["prefixItems"]) == 4
    assert param_schema["minItems"] == 4
    assert param_schema["maxItems"] == 4
    assert param_schema["prefixItems"][0] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert param_schema["prefixItems"][1] == {"type": "boolean"}
    assert param_schema["prefixItems"][2] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert param_schema["prefixItems"][3] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    assert "structure" in param_schema["description"].lower()


# --- 7. Identity preservation -----------------------------------------------


def test_refinable_parameter_instance_identity_preserved():
    p = RefinableParameter.model_validate([1.5, True])
    w = _WrapperRP(a=p)
    assert w.a is p


def test_bounded_refinable_parameter_instance_identity_preserved():
    p = BoundedRefinableParameter.model_validate([1.5, True, 0.0, 2.0])
    w = _WrapperBRP(a=p)
    assert w.a is p


def test_structure_refinable_parameter_instance_identity_preserved():
    p = StructureRefinableParameter.model_validate([1.5, True])
    w = _WrapperSRP(a=p)
    assert w.a is p


def test_structure_bounded_refinable_parameter_instance_identity_preserved():
    p = StructureBoundedRefinableParameter.model_validate([1.5, True, 0.0, 2.0])
    w = _WrapperSBRP(a=p)
    assert w.a is p


# --- check_element_symbol (A32, A63) ----------------------------------------

from powderline.schema_core import check_element_symbol  # noqa: E402


@pytest.mark.parametrize("symbol", ["Fe", "O", "La", "B", "D", "Og"])
def test_element_symbol_accepted(symbol):
    assert check_element_symbol(symbol) == symbol


@pytest.mark.parametrize("symbol, canonical", [("fe", "Fe"), ("FE", "Fe"), ("la", "La"), ("Fe ", "Fe")])
def test_element_symbol_misspelled_rejected_with_spelling(symbol, canonical):
    with pytest.raises(ValueError, match=f"write the element symbol as '{canonical}'"):
        check_element_symbol(symbol)


@pytest.mark.parametrize("symbol", ["Fe3+", "Fe+3", "O2-", "O-2", "Fe3"])
def test_charged_types_rejected_not_stripped(symbol):
    # A63: gemmi reduces e.g. "Fe3+" to Fe, which would change the scattering;
    # core rejects charged types until they are supported.
    with pytest.raises(ValueError, match="charged scattering types are not supported"):
        check_element_symbol(symbol)


@pytest.mark.parametrize("symbol", ["X", "Xx", "", "  Fe"])
def test_unknown_element_rejected(symbol):
    with pytest.raises(ValueError, match="not a known element symbol"):
        check_element_symbol(symbol)


# --- phase structure ---

import pytest  # noqa: E402
from fractions import Fraction  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from powderline.schema_core import UnitCell, Atom, PhaseStructure  # noqa: E402
from powderline.exceptions import SymmetryError  # noqa: E402


def _valid_pmm_lab6_phase_dict():
    """Helper: valid P m -3 m LaB6-like phase (cubic a=4.15692, La at origin, B at 0.5,0.5,0.2021)."""
    return {
        "phase_name": "LaB6",
        "space_group": "P m -3 m",
        "unit_cell": {
            "a": 4.15692,
            "b": 4.15692,
            "c": 4.15692,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        "atoms": {
            "La1": {
                "element": "La",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "occupancy": 1.0,
                "ADP": "Uiso",
                "Uiso": 0.005,
            },
            "B1": {
                "element": "B",
                "x": 0.5,
                "y": 0.5,
                "z": 0.2021,
                "occupancy": 1.0,
                "ADP": "Uiso",
                "Uiso": 0.006,
            },
        },
    }


def test_phase_structure_valid_validates():
    """Valid phase validates with no warnings."""
    phase = PhaseStructure.model_validate(_valid_pmm_lab6_phase_dict())
    assert phase.warnings() == []


def test_phase_structure_site_multiplicity():
    """Site multiplicity: La (origin) -> 1, B (0.5,0.5,0.2021) -> 6 in P m -3 m."""
    phase = PhaseStructure.model_validate(_valid_pmm_lab6_phase_dict())
    assert phase.site("La1").multiplicity == 1
    assert phase.site("B1").multiplicity == 6


def test_unit_cell_length_zero_rejected():
    """UnitCell: a=0 rejected."""
    with pytest.raises(ValidationError, match="must be positive"):
        UnitCell(a=0.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90)


def test_unit_cell_length_negative_rejected():
    """UnitCell: b=-1 rejected."""
    with pytest.raises(ValidationError, match="must be positive"):
        UnitCell(a=4.0, b=-1.0, c=4.0, alpha=90, beta=90, gamma=90)


def test_unit_cell_length_non_finite_rejected():
    """UnitCell: c=nan rejected."""
    with pytest.raises(ValidationError, match="must be positive"):
        UnitCell(a=4.0, b=4.0, c=float("nan"), alpha=90, beta=90, gamma=90)


def test_unit_cell_angle_zero_rejected():
    """UnitCell: alpha=0 rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell(a=4.0, b=4.0, c=4.0, alpha=0.0, beta=90, gamma=90)


def test_unit_cell_angle_180_rejected():
    """UnitCell: beta=180 rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell(a=4.0, b=4.0, c=4.0, alpha=90, beta=180.0, gamma=90)


def test_unit_cell_angle_above_180_rejected():
    """UnitCell: gamma=181 rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=181.0)


def test_unit_cell_angle_non_finite_rejected():
    """UnitCell: alpha=inf rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell(a=4.0, b=4.0, c=4.0, alpha=float("inf"), beta=90, gamma=90)


def test_unit_cell_extra_key_rejected():
    """UnitCell: 'volume' key rejected (extra='forbid')."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        UnitCell(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90, volume=64.0)


def test_atom_element_uppercase_rejected_with_suggestion():
    """Atom: 'FE' -> message suggests 'Fe'."""
    with pytest.raises(ValidationError, match="write the element symbol as 'Fe'"):
        Atom(element="FE", x=0.0, y=0.0, z=0.0, ADP="Uiso", Uiso=0.01)


def test_atom_element_charged_rejected():
    """Atom: 'Fe3+' rejected as charged."""
    with pytest.raises(ValidationError, match="charged scattering types are not supported"):
        Atom(element="Fe3+", x=0.0, y=0.0, z=0.0, ADP="Uiso", Uiso=0.01)


def test_atom_occupancy_zero_accepted():
    """Atom: occupancy=0 accepted."""
    atom = Atom(element="Fe", x=0.0, y=0.0, z=0.0, occupancy=0.0, ADP="Uiso", Uiso=0.01)
    assert atom.occupancy == 0.0


def test_atom_occupancy_one_accepted():
    """Atom: occupancy=1 accepted."""
    atom = Atom(element="Fe", x=0.0, y=0.0, z=0.0, occupancy=1.0, ADP="Uiso", Uiso=0.01)
    assert atom.occupancy == 1.0


def test_atom_occupancy_negative_rejected():
    """Atom: occupancy=-0.01 rejected."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, occupancy=-0.01, ADP="Uiso", Uiso=0.01)


def test_atom_occupancy_above_one_rejected():
    """Atom: occupancy=1.01 rejected."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, occupancy=1.01, ADP="Uiso", Uiso=0.01)


def test_atom_adp_uiso_without_value_rejected():
    """Atom: ADP='Uiso' without Uiso rejected."""
    with pytest.raises(ValidationError, match="requires a Uiso value"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, ADP="Uiso")


def test_atom_adp_uiso_with_uaniso_rejected():
    """Atom: ADP='Uiso' with Uaniso also given rejected."""
    with pytest.raises(ValidationError, match="must not also give Uaniso"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, ADP="Uiso", Uiso=0.01,
             Uaniso={"U11": 0.01, "U22": 0.01, "U33": 0.01, "U12": 0, "U13": 0, "U23": 0})


def test_atom_adp_uaniso_missing_component_rejected():
    """Atom: ADP='Uaniso' missing U23 rejected."""
    with pytest.raises(ValidationError, match="requires all of U11"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, ADP="Uaniso",
             Uaniso={"U11": 0.01, "U22": 0.01, "U33": 0.01, "U12": 0, "U13": 0})


def test_atom_adp_uaniso_unknown_key_rejected():
    """Atom: ADP='Uaniso' with U14 rejected."""
    with pytest.raises(ValidationError, match="Input should be"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, ADP="Uaniso",
             Uaniso={"U11": 0.01, "U22": 0.01, "U33": 0.01, "U12": 0, "U13": 0, "U23": 0, "U14": 0})


def test_atom_adp_uaniso_with_uiso_rejected():
    """Atom: ADP='Uaniso' with Uiso also given rejected."""
    with pytest.raises(ValidationError, match="must not also give Uiso"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, ADP="Uaniso", Uiso=0.01,
             Uaniso={"U11": 0.01, "U22": 0.01, "U33": 0.01, "U12": 0, "U13": 0, "U23": 0})


def test_atom_adp_wrong_value_rejected():
    """Atom: ADP='Uaniso' but only Uiso given rejected."""
    with pytest.raises(ValidationError, match="requires all of U11"):
        Atom(element="Fe", x=0.0, y=0.0, z=0.0, ADP="Uaniso", Uiso=0.01)


def test_atom_xyz_non_finite_rejected():
    """Atom: x=nan rejected."""
    with pytest.raises(ValidationError, match="must be finite"):
        Atom(element="Fe", x=float("nan"), y=0.0, z=0.0, ADP="Uiso", Uiso=0.01)


def test_phase_structure_space_group_no_origin_rejected():
    """PhaseStructure: 'F d -3 m' (no :1/:2) rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = "F d -3 m"
    with pytest.raises(ValidationError, match="two origin choices"):
        PhaseStructure.model_validate(d)


def test_phase_structure_space_group_no_rhombohedral_setting_rejected():
    """PhaseStructure: 'R -3 m' (no :H/:R) rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = "R -3 m"
    with pytest.raises(ValidationError, match="rhombohedral"):
        PhaseStructure.model_validate(d)


def test_phase_structure_space_group_fd3m_origin2_accepted():
    """PhaseStructure: 'F d -3 m:2' (cubic) accepted."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = "F d -3 m:2"
    # Adjust cell to cubic (already is in the helper, but be explicit)
    d["unit_cell"] = {"a": 8.0, "b": 8.0, "c": 8.0, "alpha": 90, "beta": 90, "gamma": 90}
    # Put an atom at a general position to avoid special-position complications
    d["atoms"] = {
        "A1": {"element": "O", "x": 0.125, "y": 0.125, "z": 0.125, "ADP": "Uiso", "Uiso": 0.01}
    }
    phase = PhaseStructure.model_validate(d)
    assert phase.space_group == "F d -3 m:2"


def test_phase_structure_space_group_r3m_hexagonal_accepted():
    """PhaseStructure: 'R -3 m:H' (hexagonal a=b, gamma=120) accepted."""
    d = {
        "phase_name": "test",
        "space_group": "R -3 m:H",
        "unit_cell": {"a": 3.2, "b": 3.2, "c": 5.0, "alpha": 90, "beta": 90, "gamma": 120},
        "atoms": {
            "A1": {"element": "O", "x": 0.2, "y": 0.4, "z": 0.1, "ADP": "Uiso", "Uiso": 0.01}
        },
    }
    phase = PhaseStructure.model_validate(d)
    assert phase.space_group == "R -3 m:H"


def test_phase_structure_special_position_adjusted():
    """PhaseStructure: P 63/m m c atom at (0.333333,0.666667,0.25) adjusted to 1/3,2/3,1/4."""
    d = {
        "phase_name": "test",
        "space_group": "P 63/m m c",
        "unit_cell": {"a": 3.2, "b": 3.2, "c": 5.2, "alpha": 90, "beta": 90, "gamma": 120},
        "atoms": {
            "A1": {"element": "O", "x": 0.333333, "y": 0.666667, "z": 0.25, "ADP": "Uiso", "Uiso": 0.01}
        },
    }
    phase = PhaseStructure.model_validate(d)
    # Check coordinates are exact fractions
    assert abs(phase.atoms["A1"].x - 1/3) < 1e-15
    assert abs(phase.atoms["A1"].y - 2/3) < 1e-15
    # Check warning
    warnings = phase.warnings()
    assert len(warnings) == 1
    assert warnings[0]["code"] == "special_position_adjusted"
    assert "A1" in warnings[0]["message"]
    assert warnings[0]["field_path"] == "atoms.A1"


def test_phase_structure_special_position_ambiguous_rejected():
    """PhaseStructure: atom at (0.3333,0.6667,0.25) in P 63/m m c rejected (ambiguous)."""
    d = {
        "phase_name": "test",
        "space_group": "P 63/m m c",
        "unit_cell": {"a": 3.2, "b": 3.2, "c": 5.2, "alpha": 90, "beta": 90, "gamma": 120},
        "atoms": {
            "A1": {"element": "O", "x": 0.3333, "y": 0.6667, "z": 0.25, "ADP": "Uiso", "Uiso": 0.01}
        },
    }
    with pytest.raises(ValidationError, match="without being on it") as exc_info:
        PhaseStructure.model_validate(d)
    # Check that the atom label appears in the error
    assert "A1" in str(exc_info.value)


def test_phase_structure_multiplicity_match_accepted():
    """PhaseStructure: stated Multiplicity equal to derived accepted."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["Multiplicity"] = 1
    d["atoms"]["B1"]["Multiplicity"] = 6
    phase = PhaseStructure.model_validate(d)
    assert phase.site("La1").multiplicity == 1
    assert phase.site("B1").multiplicity == 6


def test_phase_structure_multiplicity_mismatch_rejected():
    """PhaseStructure: stated Multiplicity != derived rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["Multiplicity"] = 12  # should be 6
    with pytest.raises(ValidationError) as exc_info:
        PhaseStructure.model_validate(d)
    msg = str(exc_info.value)
    assert "stated Multiplicity" in msg
    assert "derived" in msg


def test_phase_structure_cell_cubic_b_neq_a_rejected():
    """PhaseStructure: cubic with b != a rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["b"] = 4.2
    with pytest.raises(ValidationError) as exc_info:
        PhaseStructure.model_validate(d)
    msg = str(exc_info.value)
    assert "b" in msg.lower()
    assert "inconsistent" in msg.lower()


def test_phase_structure_cell_cubic_gamma_91_rejected():
    """PhaseStructure: cubic with gamma=91 rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["gamma"] = 91.0
    with pytest.raises(ValidationError, match="inconsistent"):
        PhaseStructure.model_validate(d)


def test_phase_structure_cell_hexagonal_gamma_90_rejected():
    """PhaseStructure: hexagonal with gamma=90 rejected."""
    d = {
        "phase_name": "test",
        "space_group": "P 63/m m c",
        "unit_cell": {"a": 3.2, "b": 3.2, "c": 5.2, "alpha": 90, "beta": 90, "gamma": 90},
        "atoms": {
            "A1": {"element": "O", "x": 0.2, "y": 0.4, "z": 0.1, "ADP": "Uiso", "Uiso": 0.01}
        },
    }
    with pytest.raises(ValidationError, match="inconsistent"):
        PhaseStructure.model_validate(d)


def test_phase_structure_cell_monoclinic_accepted():
    """PhaseStructure: valid monoclinic C 1 2/m 1 with beta=104.3 accepted."""
    d = {
        "phase_name": "test",
        "space_group": "C 1 2/m 1",
        "unit_cell": {"a": 5.0, "b": 6.0, "c": 7.0, "alpha": 90, "beta": 104.3, "gamma": 90},
        "atoms": {
            "A1": {"element": "O", "x": 0.2, "y": 0.0, "z": 0.1, "ADP": "Uiso", "Uiso": 0.01}
        },
    }
    phase = PhaseStructure.model_validate(d)
    assert phase.space_group == "C 1 2/m 1"


def test_phase_structure_uaniso_cubic_m3m_isotropic_accepted():
    """PhaseStructure: La at (0,0,0) in P m -3 m with isotropic Uaniso (U11=U22=U33, off-diag 0) accepted."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["ADP"] = "Uaniso"
    d["atoms"]["La1"]["Uaniso"] = {"U11": 0.01, "U22": 0.01, "U33": 0.01, "U12": 0, "U13": 0, "U23": 0}
    del d["atoms"]["La1"]["Uiso"]
    phase = PhaseStructure.model_validate(d)
    assert phase.warnings() == []


def test_phase_structure_uaniso_cubic_m3m_anisotropic_rejected():
    """PhaseStructure: La at (0,0,0) in P m -3 m with U11!=U22 rejected (breaks symmetry)."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["ADP"] = "Uaniso"
    d["atoms"]["La1"]["Uaniso"] = {"U11": 0.01, "U22": 0.02, "U33": 0.01, "U12": 0, "U13": 0, "U23": 0}
    del d["atoms"]["La1"]["Uiso"]
    with pytest.raises(ValidationError, match="break the site symmetry"):
        PhaseStructure.model_validate(d)


def test_phase_structure_uaniso_hexagonal_6h_adjusted():
    """PhaseStructure: P 63/m m c, site 6h, Uaniso with U11=U22/2 adjusted and reported."""
    d = {
        "phase_name": "test",
        "space_group": "P 63/m m c",
        "unit_cell": {"a": 3.2, "b": 3.2, "c": 5.2, "alpha": 90, "beta": 90, "gamma": 120},
        "atoms": {
            "A1": {
                "element": "O",
                "x": 0.2,
                "y": 0.4,
                "z": 0.25,
                "ADP": "Uaniso",
                "Uaniso": {
                    "U11": 0.012,
                    "U22": 0.012347,
                    "U33": 0.02,
                    "U12": 0.006174,
                    "U13": 0.0,
                    "U23": 0.0,
                },
            }
        },
    }
    phase = PhaseStructure.model_validate(d)
    # Check U11 stayed
    assert abs(phase.atoms["A1"].Uaniso["U11"] - 0.012) < 1e-15
    # Check U12 became U22/2
    u22 = phase.atoms["A1"].Uaniso["U22"]
    u12 = phase.atoms["A1"].Uaniso["U12"]
    assert abs(u12 - u22 / 2) < 1e-15
    # Check warning
    warnings = phase.warnings()
    assert any(w["code"] == "adp_symmetry_adjusted" for w in warnings)


def test_phase_structure_multiple_atom_problems_all_reported():
    """PhaseStructure: two atoms with wrong Multiplicity -> both labels appear in one error."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["Multiplicity"] = 2
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        PhaseStructure.model_validate(d)
    msg = str(exc_info.value)
    assert "La1" in msg
    assert "B1" in msg


# --- PhaseStructure: error locations, reporting, no side effects (A70) -------


def _uaniso(u11, u22, u33, u12=0.0, u13=0.0, u23=0.0):
    return {"U11": u11, "U22": u22, "U33": u33, "U12": u12, "U13": u13, "U23": u23}


def test_phase_structure_all_problems_of_one_atom_reported():
    """A wrong Multiplicity does not hide the same atom's Uaniso error."""
    d = _valid_pmm_lab6_phase_dict()
    la = d["atoms"]["La1"]
    la["Multiplicity"] = 2
    la["ADP"] = "Uaniso"
    del la["Uiso"]
    la["Uaniso"] = _uaniso(0.01, 0.02, 0.01)
    with pytest.raises(ValidationError) as exc_info:
        PhaseStructure.model_validate(d)
    assert _errors(exc_info) == [
        (("atoms", "La1", "Multiplicity"), "value_error"),
        (("atoms", "La1", "Uaniso"), "value_error"),
    ]


def test_phase_structure_problems_reported_at_their_locations():
    """Cell, position and multiplicity problems: one error each, at its own field."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["b"] = 4.2
    d["atoms"]["La1"]["x"] = 0.001  # ambiguous band
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        PhaseStructure.model_validate(d)
    assert _errors(exc_info) == [
        (("unit_cell",), "value_error"),
        (("atoms", "La1"), "value_error"),
        (("atoms", "B1", "Multiplicity"), "value_error"),
    ]
    msgs = [e["msg"] for e in exc_info.value.errors()]
    assert "b = 4.2" in msgs[0]
    assert "without being on it" in msgs[1]
    assert "stated Multiplicity 12, derived 6" in msgs[2]


def test_phase_structure_locations_nest_in_enclosing_models():
    """Engine schemas embed PhaseStructure: the locations get the enclosing path, also from JSON."""
    from pydantic import BaseModel

    class Payload(BaseModel):
        structure: PhaseStructure

    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        Payload.model_validate_json(json.dumps({"structure": d}))
    assert _errors(exc_info) == [(("structure", "atoms", "B1", "Multiplicity"), "value_error")]


def test_phase_structure_does_not_modify_caller_atoms():
    """Canonicalization replaces the atom in the structure; the caller's Atom is untouched."""
    atom = Atom.model_validate({"element": "C", "x": 0.3333333, "y": 0.6666667, "z": 0.25,
                                "ADP": "Uiso", "Uiso": 0.01})
    phase = PhaseStructure.model_validate({
        "phase_name": "test",
        "space_group": "P 63/m m c",
        "unit_cell": {"a": 3.2, "b": 3.2, "c": 5.2, "alpha": 90, "beta": 90, "gamma": 120},
        "atoms": {"C1": atom},
    })
    assert (atom.x, atom.y) == (0.3333333, 0.6666667)
    assert (phase.atoms["C1"].x, phase.atoms["C1"].y) == (1 / 3, 2 / 3)
    assert phase.atoms["C1"] is not atom
