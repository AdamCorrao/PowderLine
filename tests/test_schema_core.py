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
