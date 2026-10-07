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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_chebyshev_non_finite_coefficient_rejected(bad):
    with pytest.raises(ValidationError, match=r"coefficients\[1\] must be finite") as exc_info:
        ChebyshevBackground(num_coefficients=2, coefficients=[1.0, bad], refine_flag=False)
    assert _errors(exc_info) == [(("coefficients",), "value_error")]


def test_chebyshev_coefficient_count_match_ok():
    m = ChebyshevBackground(num_coefficients=3, coefficients=[1.0, 2.0, 3.0], refine_flag=False)
    assert len(m.coefficients) == 3


def test_chebyshev_num_coefficients_must_be_positive():
    with pytest.raises(ValidationError, match="greater than 0"):
        ChebyshevBackground(num_coefficients=0, coefficients=[], refine_flag=False)


def test_chebyshev_refine_flag_null_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag=None)
    assert _errors(exc_info) == [(('refine_flag',), 'bool_type')]


def test_chebyshev_refine_flag_string_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag="true")
    assert _errors(exc_info) == [(('refine_flag',), 'bool_type')]


def test_chebyshev_refine_flag_int_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ChebyshevBackground(num_coefficients=1, coefficients=[1.0], refine_flag=1)
    assert _errors(exc_info) == [(('refine_flag',), 'bool_type')]


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
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a=[None, True])
    assert _errors(exc_info) == [(('a', 'value'), 'float_type')]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_refinable_parameter_non_finite_value_rejected(bad):
    with pytest.raises(ValidationError, match="must be finite"):
        _WrapperRP(a=[bad, True])


def test_refinable_parameter_refine_flag_null_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a=[1.5, None])
    assert _errors(exc_info) == [(('a', 'refine_flag'), 'bool_type')]


def test_refinable_parameter_refine_flag_string_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a=[1.5, "true"])
    assert _errors(exc_info) == [(('a', 'refine_flag'), 'bool_type')]


def test_refinable_parameter_refine_flag_int_1_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a=[1.5, 1])
    assert _errors(exc_info) == [(('a', 'refine_flag'), 'bool_type')]


def test_refinable_parameter_refine_flag_int_0_rejected():
    with pytest.raises(ValidationError) as exc_info:
        _WrapperRP(a=[1.5, 0])
    assert _errors(exc_info) == [(('a', 'refine_flag'), 'bool_type')]


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
    with pytest.raises(ValidationError) as exc_info:
        _WrapperBRP(a=[None, True, 0.0, 2.0])
    assert _errors(exc_info) == [(('a', 'value'), 'float_type')]


# --- 4. check_within_bounds -------------------------------------------------


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


# --- 5. JSON schema ---------------------------------------------------------


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


# --- 6. Identity preservation -----------------------------------------------


def test_refinable_parameter_instance_identity_preserved():
    p = RefinableParameter.model_validate([1.5, True])
    w = _WrapperRP(a=p)
    assert w.a is p


def test_bounded_refinable_parameter_instance_identity_preserved():
    p = BoundedRefinableParameter.model_validate([1.5, True, 0.0, 2.0])
    w = _WrapperBRP(a=p)
    assert w.a is p


# --- check_element_symbol (A32, A63) ----------------------------------------

from powderline.schema_core import check_element_symbol  # noqa: E402


@pytest.mark.parametrize("symbol", ["Fe", "O", "La", "B", "D", "Og"])
def test_element_symbol_accepted(symbol):
    assert check_element_symbol(symbol) == symbol


@pytest.mark.parametrize("symbol, canonical", [("fe", "Fe"), ("FE", "Fe"), ("la", "La"), ("Fe ", "Fe"), (" Fe", "Fe"), ("co", "Co")])
def test_element_symbol_misspelled_rejected_with_spelling(symbol, canonical):
    with pytest.raises(ValueError, match=f"write the element symbol as '{canonical}'"):
        check_element_symbol(symbol)


@pytest.mark.parametrize("symbol", ["Fe3+", "Fe+3", "O2-", "O-2", "Fe3"])
def test_charged_types_rejected_not_stripped(symbol):
    # A63: gemmi reduces e.g. "Fe3+" to Fe, which would change the scattering;
    # core rejects charged types until they are supported.
    with pytest.raises(ValueError, match="charged scattering types are not supported"):
        check_element_symbol(symbol)


@pytest.mark.parametrize("symbol", ["X", "Xx", "", "Nax", "Fex"])
def test_unknown_element_rejected(symbol):
    # gemmi reads a prefix ("Nax" -> Na); that must not be offered as a spelling.
    with pytest.raises(ValueError, match="not a known element symbol") as exc_info:
        check_element_symbol(symbol)
    assert "write the element symbol" not in str(exc_info.value)


# --- phase structure ---

import pytest  # noqa: E402
from fractions import Fraction  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from powderline.schema_core import UnitCell, Atom, Phase, RefinableParameter  # noqa: E402
from powderline.exceptions import SymmetryError  # noqa: E402


def _valid_pmm_lab6_phase_dict():
    """Helper: valid P m -3 m LaB6-like phase (cubic a=4.15692, La at origin, B at 0.5,0.5,0.2021)."""
    return {
        "space_group": "P m -3 m",
        "unit_cell": {
            "a": [4.15692, False],
            "b": [4.15692, False],
            "c": [4.15692, False],
            "alpha": [90.0, False],
            "beta": [90.0, False],
            "gamma": [90.0, False],
        },
        "atoms": {
            "La1": {
                "element": "La",
                "x": [0.0, False],
                "y": [0.0, False],
                "z": [0.0, False],
                "occupancy": [1.0, False],
                "ADP": "Uiso",
                "Uiso": [0.005, False],
            },
            "B1": {
                "element": "B",
                "x": [0.5, False],
                "y": [0.5, False],
                "z": [0.2021, False],
                "occupancy": [1.0, False],
                "ADP": "Uiso",
                "Uiso": [0.006, False],
            },
        },
    }


def test_phase_valid_validates():
    """Valid phase validates with no warnings."""
    phase = Phase[RefinableParameter].model_validate(_valid_pmm_lab6_phase_dict())
    assert phase.warnings() == []


def test_phase_site_multiplicity():
    """Site multiplicity: La (origin) -> 1, B (0.5,0.5,0.2021) -> 6 in P m -3 m."""
    phase = Phase[RefinableParameter].model_validate(_valid_pmm_lab6_phase_dict())
    assert phase.site("La1").multiplicity == 1
    assert phase.site("B1").multiplicity == 6


def test_unit_cell_length_zero_rejected():
    """UnitCell: a=0 rejected."""
    with pytest.raises(ValidationError, match="must be positive"):
        UnitCell[RefinableParameter](a=[0.0, False], b=[4.0, False], c=[4.0, False], alpha=[90, False], beta=[90, False], gamma=[90, False])


def test_unit_cell_length_negative_rejected():
    """UnitCell: b=-1 rejected."""
    with pytest.raises(ValidationError, match="must be positive"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[-1.0, False], c=[4.0, False], alpha=[90, False], beta=[90, False], gamma=[90, False])


def test_unit_cell_length_non_finite_rejected():
    """UnitCell: c=nan rejected."""
    with pytest.raises(ValidationError, match="must be finite"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[4.0, False], c=[float("nan"), False], alpha=[90, False], beta=[90, False], gamma=[90, False])


def test_unit_cell_angle_zero_rejected():
    """UnitCell: alpha=0 rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[4.0, False], c=[4.0, False], alpha=[0.0, False], beta=[90, False], gamma=[90, False])


def test_unit_cell_angle_180_rejected():
    """UnitCell: beta=180 rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[4.0, False], c=[4.0, False], alpha=[90, False], beta=[180.0, False], gamma=[90, False])


def test_unit_cell_angle_above_180_rejected():
    """UnitCell: gamma=181 rejected."""
    with pytest.raises(ValidationError, match="must be between 0 and 180"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[4.0, False], c=[4.0, False], alpha=[90, False], beta=[90, False], gamma=[181.0, False])


def test_unit_cell_angle_non_finite_rejected():
    """UnitCell: alpha=inf rejected."""
    with pytest.raises(ValidationError, match="must be finite"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[4.0, False], c=[4.0, False], alpha=[float("inf"), False], beta=[90, False], gamma=[90, False])


def test_unit_cell_extra_key_rejected():
    """UnitCell: 'volume' key rejected (extra='forbid')."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        UnitCell[RefinableParameter](a=[4.0, False], b=[4.0, False], c=[4.0, False], alpha=[90, False], beta=[90, False], gamma=[90, False], volume=64.0)


def test_atom_element_uppercase_rejected_with_suggestion():
    """Atom: 'FE' -> message suggests 'Fe'."""
    with pytest.raises(ValidationError, match="write the element symbol as 'Fe'"):
        Atom[RefinableParameter](element="FE", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uiso", Uiso=[0.01, False])


def test_atom_element_charged_rejected():
    """Atom: 'Fe3+' rejected as charged."""
    with pytest.raises(ValidationError, match="charged scattering types are not supported"):
        Atom[RefinableParameter](element="Fe3+", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uiso", Uiso=[0.01, False])


def test_atom_occupancy_zero_accepted():
    """Atom: occupancy=0 accepted."""
    atom = Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[0.0, False], ADP="Uiso", Uiso=[0.01, False])
    assert atom.occupancy.value == 0.0


def test_atom_occupancy_one_accepted():
    """Atom: occupancy=1 accepted."""
    atom = Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uiso", Uiso=[0.01, False])
    assert atom.occupancy.value == 1.0


def test_atom_occupancy_negative_rejected():
    """Atom: occupancy=-0.01 rejected."""
    with pytest.raises(ValidationError, match="occupancy must be between 0 and 1"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[-0.01, False], ADP="Uiso", Uiso=[0.01, False])


def test_atom_occupancy_above_one_rejected():
    """Atom: occupancy=1.01 rejected."""
    with pytest.raises(ValidationError, match="occupancy must be between 0 and 1"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.01, False], ADP="Uiso", Uiso=[0.01, False])


def test_atom_adp_uiso_without_value_rejected():
    """Atom: ADP='Uiso' without Uiso rejected."""
    with pytest.raises(ValidationError, match="requires a Uiso value"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uiso")


def test_atom_adp_uiso_with_uaniso_rejected():
    """Atom: ADP='Uiso' with Uaniso also given rejected."""
    with pytest.raises(ValidationError, match="must not also give Uaniso"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uiso", Uiso=[0.01, False],
             Uaniso={"U11": [0.01, False], "U22": [0.01, False], "U33": [0.01, False], "U12": [0, False], "U13": [0, False], "U23": [0, False]})


def test_atom_adp_uaniso_missing_component_rejected():
    """Atom: ADP='Uaniso' missing U23 rejected."""
    with pytest.raises(ValidationError, match="requires all of U11"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uaniso",
             Uaniso={"U11": [0.01, False], "U22": [0.01, False], "U33": [0.01, False], "U12": [0, False], "U13": [0, False]})


def test_atom_adp_uaniso_unknown_key_rejected():
    """Atom: ADP='Uaniso' with U14 rejected."""
    with pytest.raises(ValidationError, match="Input should be"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uaniso",
             Uaniso={"U11": [0.01, False], "U22": [0.01, False], "U33": [0.01, False], "U12": [0, False], "U13": [0, False], "U23": [0, False], "U14": [0, False]})


def test_atom_adp_uaniso_with_uiso_rejected():
    """Atom: ADP='Uaniso' with Uiso also given rejected."""
    with pytest.raises(ValidationError, match="must not also give Uiso"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uaniso", Uiso=[0.01, False],
             Uaniso={"U11": [0.01, False], "U22": [0.01, False], "U33": [0.01, False], "U12": [0, False], "U13": [0, False], "U23": [0, False]})


def test_atom_adp_wrong_value_rejected():
    """Atom: ADP='Uaniso' but only Uiso given rejected."""
    with pytest.raises(ValidationError, match="requires all of U11"):
        Atom[RefinableParameter](element="Fe", x=[0.0, False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uaniso", Uiso=[0.01, False])


def test_atom_xyz_non_finite_rejected():
    """Atom: x=nan rejected."""
    with pytest.raises(ValidationError, match="must be finite"):
        Atom[RefinableParameter](element="Fe", x=[float("nan"), False], y=[0.0, False], z=[0.0, False], occupancy=[1.0, False], ADP="Uiso", Uiso=[0.01, False])


def test_phase_space_group_no_origin_rejected():
    """Phase: 'F d -3 m' (no :1/:2) rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = "F d -3 m"
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("space_group",)
    assert "space group 'F d -3 m' has two origin choices" in errors[0]["msg"]
    assert "write 'F d -3 m:1' or 'F d -3 m:2'" in errors[0]["msg"]


def test_phase_space_group_no_rhombohedral_setting_rejected():
    """Phase: 'R -3 m' (no :H/:R) rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = "R -3 m"
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("space_group",)
    assert "space group 'R -3 m' does not name its setting" in errors[0]["msg"]
    assert "write 'R -3 m:H'" in errors[0]["msg"]
    assert "or 'R -3 m:R'" in errors[0]["msg"]


def test_phase_space_group_fd3m_origin2_accepted():
    """Phase: 'F d -3 m:2' (cubic) accepted."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = "F d -3 m:2"
    # Adjust cell to cubic (already is in the helper, but be explicit)
    d["unit_cell"] = {"a": [8.0, False], "b": [8.0, False], "c": [8.0, False], "alpha": [90, False], "beta": [90, False], "gamma": [90, False]}
    # Put an atom at a general position to avoid special-position complications
    d["atoms"] = {
        "A1": {"element": "O", "x": [0.125, False], "y": [0.125, False], "z": [0.125, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
    }
    phase = Phase[RefinableParameter].model_validate(d)
    assert phase.space_group == "F d -3 m:2"


def test_phase_space_group_r3m_hexagonal_accepted():
    """Phase: 'R -3 m:H' (hexagonal a=b, gamma=120) accepted."""
    d = {
        "space_group": "R -3 m:H",
        "unit_cell": {"a": [3.2, False], "b": [3.2, False], "c": [5.0, False], "alpha": [90, False], "beta": [90, False], "gamma": [120, False]},
        "atoms": {
            "A1": {"element": "O", "x": [0.2, False], "y": [0.4, False], "z": [0.1, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
        },
    }
    phase = Phase[RefinableParameter].model_validate(d)
    assert phase.space_group == "R -3 m:H"


def test_phase_special_position_adjusted():
    """Phase: P 63/m m c atom at (0.333333,0.666667,0.25) adjusted to 1/3,2/3,1/4."""
    d = {
        "space_group": "P 63/m m c",
        "unit_cell": {"a": [3.2, False], "b": [3.2, False], "c": [5.2, False], "alpha": [90, False], "beta": [90, False], "gamma": [120, False]},
        "atoms": {
            "A1": {"element": "O", "x": [0.333333, False], "y": [0.666667, False], "z": [0.25, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
        },
    }
    phase = Phase[RefinableParameter].model_validate(d)
    # Check coordinates are exact fractions (now need to read .value)
    assert phase.atoms["A1"].x.value == 1/3
    assert phase.atoms["A1"].y.value == 2/3
    # Check refine flags are preserved
    assert phase.atoms["A1"].x.refine_flag is False
    assert phase.atoms["A1"].y.refine_flag is False
    # Check warning
    warnings = phase.warnings()
    assert len(warnings) == 1
    assert warnings[0]["code"] == "special_position_adjusted"
    assert "A1" in warnings[0]["message"]
    assert warnings[0]["field_path"] == "atoms.A1"


def test_phase_special_position_ambiguous_rejected():
    """Phase: atom at (0.3333,0.6667,0.25) in P 63/m m c rejected (ambiguous)."""
    d = {
        "space_group": "P 63/m m c",
        "unit_cell": {"a": [3.2, False], "b": [3.2, False], "c": [5.2, False], "alpha": [90, False], "beta": [90, False], "gamma": [120, False]},
        "atoms": {
            "A1": {"element": "O", "x": [0.3333, False], "y": [0.6667, False], "z": [0.25, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
        },
    }
    with pytest.raises(ValidationError, match="without being on it") as exc_info:
        Phase[RefinableParameter].model_validate(d)
    # Check that the atom label appears in the error
    assert "A1" in str(exc_info.value)


def test_phase_multiplicity_match_accepted():
    """Phase: stated Multiplicity equal to derived accepted."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["Multiplicity"] = 1
    d["atoms"]["B1"]["Multiplicity"] = 6
    phase = Phase[RefinableParameter].model_validate(d)
    assert phase.site("La1").multiplicity == 1
    assert phase.site("B1").multiplicity == 6


def test_phase_multiplicity_mismatch_rejected():
    """Phase: stated Multiplicity != derived rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["Multiplicity"] = 12  # should be 6
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    msg = str(exc_info.value)
    assert "stated Multiplicity" in msg
    assert "derived" in msg


def test_phase_cell_cubic_b_neq_a_rejected():
    """Phase: cubic with b != a rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["b"] = [4.2, False]
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    msg = str(exc_info.value)
    assert "b = 4.2 (symmetric value" in msg
    assert "inconsistent" in msg
    assert _errors(exc_info) == [(("unit_cell",), "value_error")]


def test_phase_cell_cubic_gamma_91_rejected():
    """Phase: cubic with gamma=91 rejected."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["gamma"] = [91.0, False]
    with pytest.raises(ValidationError, match="inconsistent"):
        Phase[RefinableParameter].model_validate(d)


def test_phase_cell_hexagonal_gamma_90_rejected():
    """Phase: hexagonal with gamma=90 rejected."""
    d = {
        "space_group": "P 63/m m c",
        "unit_cell": {"a": [3.2, False], "b": [3.2, False], "c": [5.2, False], "alpha": [90, False], "beta": [90, False], "gamma": [90, False]},
        "atoms": {
            "A1": {"element": "O", "x": [0.2, False], "y": [0.4, False], "z": [0.1, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
        },
    }
    with pytest.raises(ValidationError, match="inconsistent"):
        Phase[RefinableParameter].model_validate(d)


def test_phase_cell_monoclinic_accepted():
    """Phase: valid monoclinic C 1 2/m 1 with beta=104.3 accepted."""
    d = {
        "space_group": "C 1 2/m 1",
        "unit_cell": {"a": [5.0, False], "b": [6.0, False], "c": [7.0, False], "alpha": [90, False], "beta": [104.3, False], "gamma": [90, False]},
        "atoms": {
            "A1": {"element": "O", "x": [0.2, False], "y": [0.0, False], "z": [0.1, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
        },
    }
    phase = Phase[RefinableParameter].model_validate(d)
    assert phase.space_group == "C 1 2/m 1"


def test_phase_uaniso_cubic_m3m_isotropic_accepted():
    """Phase: La at (0,0,0) in P m -3 m with isotropic Uaniso (U11=U22=U33, off-diag 0) accepted."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["ADP"] = "Uaniso"
    d["atoms"]["La1"]["Uaniso"] = {"U11": [0.01, False], "U22": [0.01, False], "U33": [0.01, False], "U12": [0, False], "U13": [0, False], "U23": [0, False]}
    del d["atoms"]["La1"]["Uiso"]
    phase = Phase[RefinableParameter].model_validate(d)
    assert phase.warnings() == []


def test_phase_uaniso_cubic_m3m_anisotropic_rejected():
    """Phase: La at (0,0,0) in P m -3 m with U11!=U22 rejected (breaks symmetry)."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["ADP"] = "Uaniso"
    d["atoms"]["La1"]["Uaniso"] = {"U11": [0.01, False], "U22": [0.02, False], "U33": [0.01, False], "U12": [0, False], "U13": [0, False], "U23": [0, False]}
    del d["atoms"]["La1"]["Uiso"]
    with pytest.raises(ValidationError, match="break the site symmetry"):
        Phase[RefinableParameter].model_validate(d)


def test_phase_uaniso_hexagonal_6h_adjusted():
    """Phase: P 63/m m c, site 6h, Uaniso with U11=U22/2 adjusted and reported."""
    d = {
        "space_group": "P 63/m m c",
        "unit_cell": {"a": [3.2, False], "b": [3.2, False], "c": [5.2, False], "alpha": [90, False], "beta": [90, False], "gamma": [120, False]},
        "atoms": {
            "A1": {
                "element": "O",
                "x": [0.2, False],
                "y": [0.4, False],
                "z": [0.25, False],
                "occupancy": [1.0, False],
                "ADP": "Uaniso",
                "Uaniso": {
                    "U11": [0.012, False],
                    "U22": [0.012347, False],
                    "U33": [0.02, False],
                    "U12": [0.006174, False],
                    "U13": [0.0, False],
                    "U23": [0.0, False],
                },
            }
        },
    }
    phase = Phase[RefinableParameter].model_validate(d)
    # Check U11 stayed (now need to read .value)
    assert phase.atoms["A1"].Uaniso["U11"].value == 0.012  # free: returned exactly as stated
    # Check U12 became U22/2 (reading .value)
    u22 = phase.atoms["A1"].Uaniso["U22"].value
    u12 = phase.atoms["A1"].Uaniso["U12"].value
    assert abs(u12 - u22 / 2) < 1e-15
    # Check refine flags are preserved
    assert phase.atoms["A1"].Uaniso["U11"].refine_flag is False
    assert phase.atoms["A1"].Uaniso["U12"].refine_flag is False
    # Check warning
    warnings = phase.warnings()
    assert any(w["code"] == "adp_symmetry_adjusted" for w in warnings)


def test_phase_multiple_atom_problems_all_reported():
    """Phase: two atoms with wrong Multiplicity -> both labels appear in one error."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["La1"]["Multiplicity"] = 2
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    msg = str(exc_info.value)
    assert "La1" in msg
    assert "B1" in msg


# --- Phase: error locations, reporting, no side effects (A70) -------


def _uaniso(u11, u22, u33, u12=0.0, u13=0.0, u23=0.0):
    return {"U11": [u11, False], "U22": [u22, False], "U33": [u33, False], "U12": [u12, False], "U13": [u13, False], "U23": [u23, False]}


def test_phase_all_problems_of_one_atom_reported():
    """A wrong Multiplicity does not hide the same atom's Uaniso error."""
    d = _valid_pmm_lab6_phase_dict()
    la = d["atoms"]["La1"]
    la["Multiplicity"] = 2
    la["ADP"] = "Uaniso"
    del la["Uiso"]
    la["Uaniso"] = _uaniso(0.01, 0.02, 0.01)
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    assert _errors(exc_info) == [
        (("atoms", "La1", "Multiplicity"), "value_error"),
        (("atoms", "La1", "Uaniso"), "value_error"),
    ]


def test_phase_problems_reported_at_their_locations():
    """Cell, position and multiplicity problems: one error each, at its own field."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["b"] = [4.2, False]
    d["atoms"]["La1"]["x"] = [0.001, False]  # ambiguous band
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        Phase[RefinableParameter].model_validate(d)
    assert _errors(exc_info) == [
        (("unit_cell",), "value_error"),
        (("atoms", "La1"), "value_error"),
        (("atoms", "B1", "Multiplicity"), "value_error"),
    ]
    msgs = [e["msg"] for e in exc_info.value.errors()]
    assert "b = 4.2" in msgs[0]
    assert "without being on it" in msgs[1]
    assert "stated Multiplicity 12, derived 6" in msgs[2]


def test_phase_locations_nest_in_enclosing_models():
    """Engine schemas embed Phase: the locations get the enclosing path, also from JSON."""
    from pydantic import BaseModel

    class Payload(BaseModel):
        structure: Phase[RefinableParameter]

    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        Payload.model_validate_json(json.dumps({"structure": d}))
    assert _errors(exc_info) == [(("structure", "atoms", "B1", "Multiplicity"), "value_error")]


def test_phase_does_not_modify_caller_atoms():
    """Canonicalization replaces the atom in the structure; the caller's Atom is untouched."""
    atom = Atom[RefinableParameter].model_validate({"element": "C", "x": [0.3333333, False], "y": [0.6666667, False], "z": [0.25, False],
                                "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]})
    phase = Phase[RefinableParameter].model_validate({
        "space_group": "P 63/m m c",
        "unit_cell": {"a": [3.2, False], "b": [3.2, False], "c": [5.2, False], "alpha": [90, False], "beta": [90, False], "gamma": [120, False]},
        "atoms": {"C1": atom},
    })
    assert (atom.x.value, atom.y.value) == (0.3333333, 0.6666667)
    assert (phase.atoms["C1"].x.value, phase.atoms["C1"].y.value) == (1 / 3, 2 / 3)
    assert phase.atoms["C1"] is not atom


# --- Phase[P]: the phase block (re/03b; A93-A96, A100, A105) -----------------

from powderline.schema_core import (  # noqa: E402
    RESERVED_PHASE_FIELDS,
    BoundedRefinableParameter,
    CoreModel,
)

RP, BRP = RefinableParameter, BoundedRefinableParameter


def _bounded(d: dict, lo=None, hi=None) -> dict:
    """The phase dict with every [value, flag] made [value, flag, lo, hi]."""
    def conv(v):
        if isinstance(v, list):
            return [*v, lo, hi]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        return v
    return conv(d)


@pytest.mark.parametrize("space_group, cell, xyz", [
    ("P m -3 m", (4.0, 4.0, 4.0, 90, 90, 90), (0.1, 0.2, 0.3)),
    ("C 1 2/m 1", (9.0, 5.0, 7.0, 90, 105.0, 90), (0.1, 0.2, 0.3)),
    ("P 1 21/c 1", (5.0, 6.0, 7.0, 90, 98.0, 90), (0.1, 0.2, 0.3)),
    ("P 1 1 21/a", (5.0, 6.0, 7.0, 90, 90, 98.0), (0.1, 0.2, 0.3)),
    ("A 1 1 2/a", (5.0, 6.0, 7.0, 90, 90, 98.0), (0.1, 0.2, 0.3)),
    ("C c 1 1", (5.0, 6.0, 7.0, 98.0, 90, 90), (0.1, 0.2, 0.3)),
    ("P 1 c 1", (5.0, 6.0, 7.0, 90, 98.0, 90), (0.1, 0.2, 0.3)),
    ("R -3 m:H", (4.0, 4.0, 20.0, 90, 90, 120), (0.1, 0.2, 0.3)),
    ("R -3 m:R", (5.0, 5.0, 5.0, 70.0, 70.0, 70.0), (0.1, 0.2, 0.3)),
    ("F d -3 m:2", (8.0, 8.0, 8.0, 90, 90, 90), (0.125, 0.125, 0.125)),
    ("F d -3 m:1", (8.0, 8.0, 8.0, 90, 90, 90), (0.0, 0.0, 0.0)),
    ("P 63/m m c", (3.2, 3.2, 5.2, 90, 90, 120), (1 / 3, 2 / 3, 0.25)),
])
def test_phase_canonical_space_group_accepted(space_group, cell, xyz):
    """Canonical names validate, incl. non-standard settings, ':R' and both origins (A105)."""
    d = {"space_group": space_group,
         "unit_cell": dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), ([v, False] for v in cell))),
         "atoms": {"A1": {"element": "Fe", **{k: [v, False] for k, v in zip("xyz", xyz)},
                          "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}}}
    assert Phase[RP].model_validate(d).space_group == space_group


@pytest.mark.parametrize("symbol, message", [
    ("Pm-3m", "space group 'Pm-3m' is not in the canonical form; write 'P m -3 m'"),
    ("p m -3 m", "space group 'p m -3 m' is not in the canonical form; write 'P m -3 m'"),
    ("C2/m", "space group 'C2/m' is not in the canonical form; write 'C 1 2/m 1'"),
    ("C 2/m", "space group 'C 2/m' is not in the canonical form; write 'C 1 2/m 1'"),
    ("P 21/c", "space group 'P 21/c' is not in the canonical form; write 'P 1 21/c 1'"),
    ("R-3m:H", "space group 'R-3m:H' is not in the canonical form; write 'R -3 m:H'"),
    ("R -3 m", "space group 'R -3 m' does not name its setting; write 'R -3 m:H' (hexagonal axes) "
               "or 'R -3 m:R' (rhombohedral axes)"),
    ("Fd-3m", "space group 'Fd-3m' has two origin choices; write 'F d -3 m:1' or 'F d -3 m:2'"),
    ("F d -3 m", "space group 'F d -3 m' has two origin choices; write 'F d -3 m:1' or 'F d -3 m:2'"),
    ("P2_1/c", "unrecognized space-group symbol 'P2_1/c'; write gemmi's canonical name "
               "(e.g. 'P m -3 m', 'C 1 2/m 1', 'R -3 m:H', 'F d -3 m:2')"),
])
def test_phase_non_canonical_space_group_rejected(symbol, message):
    """Any other spelling is an error at space_group naming the canonical form (A105)."""
    d = _valid_pmm_lab6_phase_dict()
    d["space_group"] = symbol
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    assert _errors(exc_info) == [(("space_group",), "value_error")]
    assert exc_info.value.errors()[0]["msg"] == "Value error, " + message


def test_canonical_space_group_every_gemmi_name_round_trips():
    """gemmi's xhm() names are unique over its table and each validates as itself (A105)."""
    import gemmi
    from powderline.symmetry import canonical_space_group

    names = [sg.xhm() for sg in gemmi.spacegroup_table()]
    assert len(names) == len(set(names)) == 564
    assert all(canonical_space_group(n).xhm() == n for n in names)


@pytest.mark.parametrize("field", RESERVED_PHASE_FIELDS)
def test_phase_reserved_field_cannot_be_redeclared(field):
    """An engine subclass may not redefine space_group, unit_cell or atoms (A93)."""
    with pytest.raises(TypeError) as exc_info:
        type("EnginePhase", (Phase[RP],), {"__annotations__": {field: int}, "__module__": __name__})
    assert str(exc_info.value) == (f"EnginePhase redeclares the core phase field(s) {field}; "
                                   "core owns space_group, unit_cell and atoms (A93)")


def test_phase_reserved_fields_guarded_below_an_engine_subclass():
    class EnginePhase(Phase[RP]):
        scale: RP

    with pytest.raises(TypeError, match=r"^Child redeclares the core phase field\(s\) atoms;"):
        class Child(EnginePhase):
            atoms: dict


def test_phase_engine_subclass_adds_flat_fields_and_keeps_core_validation():
    """The gsasii shape: engine fields flat in the phase; core rules still run; locs nest."""
    class EnginePhase(Phase[RP]):
        scale: RP

    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["x"] = [0.4999999, False]
    phase = EnginePhase.model_validate({**d, "scale": [1.0, True]})
    assert phase.scale.model_dump() == [1.0, True]
    assert phase.atoms["B1"].x.value == 0.5  # canonicalized by the core validator
    d["atoms"]["B1"]["Multiplicity"] = 12
    with pytest.raises(ValidationError) as exc_info:
        EnginePhase.model_validate({**d, "scale": [1.0, True]})
    assert _errors(exc_info) == [(("atoms", "B1", "Multiplicity"), "value_error")]
    with pytest.raises(ValidationError) as exc_info:
        EnginePhase.model_validate(d)
    assert _errors(exc_info) == [(("scale",), "missing")]


def test_phase_has_no_phase_name():
    """The phase name is the key of the payload's phases dict (A100)."""
    d = {"phase_name": "LaB6", **_valid_pmm_lab6_phase_dict()}
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    assert _errors(exc_info) == [(("phase_name",), "extra_forbidden")]


def test_atom_occupancy_required():
    """No default: occupancy must be stated (A96)."""
    d = _valid_pmm_lab6_phase_dict()
    del d["atoms"]["La1"]["occupancy"]
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    assert _errors(exc_info) == [(("atoms", "La1", "occupancy"), "missing")]


@pytest.mark.parametrize("field", ["Multiplicity", "Uiso", "Uaniso"])
def test_atom_optional_fields_are_never_null(field):
    """A field that may be left out is never null (A96); leaving it out is the way."""
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"][field] = None
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    assert _errors(exc_info) == [(("atoms", "B1", field), "value_error")]
    assert exc_info.value.errors()[0]["msg"] == f"Value error, {field} must not be null; leave it out instead"


@pytest.mark.parametrize("param", [[None, False], [0.5, None]])
def test_phase_parameter_value_and_flag_never_null(param):
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["z"] = param
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    slot = "value" if param[0] is None else "refine_flag"
    assert _errors(exc_info) == [(("atoms", "B1", "z", slot), "float_type" if slot == "value" else "bool_type")]


def test_phase_dump_round_trips_and_leaves_absent_fields_out():
    d = _valid_pmm_lab6_phase_dict()
    d["atoms"]["B1"]["Multiplicity"] = 6
    phase = Phase[RP].model_validate(d)
    dump = phase.model_dump(mode="json")
    assert dump == d
    assert "Uaniso" not in dump["atoms"]["La1"] and "Multiplicity" not in dump["atoms"]["La1"]
    assert Phase[RP].model_validate_json(phase.model_dump_json()).model_dump() == phase.model_dump()


def test_phase_json_schema_names_and_no_null():
    """Readable $defs (no 'Atom_RefinableParameter_'); optional fields are not nullable (A96)."""
    class EnginePhase(Phase[RP]):
        scale: RP

    schema = EnginePhase.model_json_schema()
    assert sorted(schema["$defs"]) == ["Atom", "RefinableParameter", "UnitCell"]
    atom = schema["$defs"]["Atom"]
    assert atom["required"] == ["element", "x", "y", "z", "occupancy", "ADP"]
    assert atom["properties"]["Uiso"] == {"$ref": "#/$defs/RefinableParameter", "unit": "angstrom^2",
                                          "description": "Isotropic ADP; given when ADP is 'Uiso'"}
    assert atom["properties"]["Multiplicity"]["type"] == "integer"
    assert "null" not in json.dumps(schema)
    assert schema["$defs"]["RefinableParameter"]["prefixItems"] == [{"type": "number"}, {"type": "boolean"}]


def test_phase_json_schema_two_parameter_types_in_one_document_stay_distinct():
    """If two parametrizations ever share a document, the names fall back and the refs stay right."""
    class Both(CoreModel):
        g: Phase[RP]
        t: Phase[BRP]

    defs = Both.model_json_schema()["$defs"]
    atom_refs = {k: v["properties"]["atoms"]["additionalProperties"]["$ref"].rsplit("/", 1)[1]
                 for k, v in defs.items() if "Phase" in k}
    assert len(set(atom_refs.values())) == 2
    for name, ref in atom_refs.items():
        expected = "BoundedRefinableParameter" if "Bounded" in name else "RefinableParameter"
        assert defs[ref]["properties"]["x"]["$ref"] == f"#/$defs/{expected}"


def test_phase_rejects_bounds_where_the_parameter_type_has_none():
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["a"] = [4.15692, False, 4.0, 4.3]
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    assert _errors(exc_info) == [(("unit_cell", "a"), "value_error")]
    assert "bounds are not supported" in exc_info.value.errors()[0]["msg"]


def test_bounded_phase_canonicalization_keeps_flag_and_bounds():
    """Phase[BoundedRefinableParameter]: the canonical value replaces .value; flag and bounds stay."""
    d = _bounded(_valid_pmm_lab6_phase_dict())
    d["atoms"]["B1"]["x"] = [0.4999999, False, None, None]
    d["atoms"]["B1"]["z"] = [0.2021, True, 0.15, 0.25]
    phase = Phase[BRP].model_validate(d)
    assert phase.atoms["B1"].x.model_dump() == [0.5, False, None, None]
    assert phase.atoms["B1"].z.model_dump() == [0.2021, True, 0.15, 0.25]
    assert [w["code"] for w in phase.warnings()] == ["special_position_adjusted"]


def test_phase_value_rules_act_on_value():
    """Cell, occupancy and element rules read .value; each error at its own field (A70)."""
    d = _valid_pmm_lab6_phase_dict()
    d["unit_cell"]["a"] = [-4.0, False]
    d["atoms"]["La1"]["occupancy"] = [1.5, False]
    d["atoms"]["B1"]["element"] = "b"
    with pytest.raises(ValidationError) as exc_info:
        Phase[RP].model_validate(d)
    assert _errors(exc_info) == [(("unit_cell", "a"), "value_error"),
                                 (("atoms", "La1", "occupancy"), "value_error"),
                                 (("atoms", "B1", "element"), "value_error")]
    assert [e["msg"] for e in exc_info.value.errors()] == [
        "Value error, cell length a must be positive, got -4.0",
        "Value error, occupancy must be between 0 and 1 inclusive, got 1.5",
        "Value error, element 'b': write the element symbol as 'B'",
    ]


# --- numbers are JSON numbers: no strings, no booleans (A80) -----------------

import numpy as np  # noqa: E402

from powderline.schema_core import FitRange, XRDData  # noqa: E402


def _cell(**kw):
    base = {"a": [4.0, False], "b": [4.0, False], "c": [4.0, False], "alpha": [90, False], "beta": [90, False], "gamma": [90, False]}
    base.update(kw)
    return base


def _atom(**kw):
    base = {"element": "Na", "x": [0.1, False], "y": [0.2, False], "z": [0.3, False], "occupancy": [1.0, False], "ADP": "Uiso", "Uiso": [0.01, False]}
    base.update(kw)
    return base


def _xrd(**kw):
    return {"tth": [10.0, 20.0], "Itth": [1.0, 2.0], "Itth_weights": [1.0, 1.0], **kw}


# (label, build(v) -> validated model, error location); every numeric core field
_NUMERIC_FIELDS = [
    ("RefinableParameter.value", lambda v: RefinableParameter.model_validate([v, True]), ("value",)),
    ("BoundedRefinableParameter.value", lambda v: BoundedRefinableParameter.model_validate([v, True, None, None]), ("value",)),
    ("BoundedRefinableParameter.min", lambda v: BoundedRefinableParameter.model_validate([1.0, True, v, None]), ("min",)),
    ("BoundedRefinableParameter.max", lambda v: BoundedRefinableParameter.model_validate([0.5, True, None, v]), ("max",)),
    ("FitRange.min", lambda v: FitRange.model_validate([v, None]), ("min",)),
    ("FitRange.max", lambda v: FitRange.model_validate([None, v]), ("max",)),
    ("XRDData.tth", lambda v: XRDData.model_validate(_xrd(tth=[v, 20.0])), ("tth", 0)),
    ("XRDData.Itth", lambda v: XRDData.model_validate(_xrd(Itth=[v, 2.0])), ("Itth", 0)),
    ("XRDData.Itth_weights", lambda v: XRDData.model_validate(_xrd(Itth_weights=[v, 1.0])), ("Itth_weights", 0)),
    ("ChebyshevBackground.coefficients",
     lambda v: ChebyshevBackground(num_coefficients=1, coefficients=[v], refine_flag=False), ("coefficients", 0)),
    *[(f"UnitCell.{k}", (lambda k: lambda v: UnitCell[RefinableParameter].model_validate(_cell(**{k: [v, False]})))(k), (k, "value"))
      for k in ("a", "b", "c", "alpha", "beta", "gamma")],
    *[(f"Atom.{k}", (lambda k: lambda v: Atom[RefinableParameter].model_validate(_atom(**{k: [v, False]})))(k), (k, "value"))
      for k in ("x", "y", "z", "occupancy", "Uiso")],
    ("Atom.Uaniso", lambda v: Atom[RefinableParameter].model_validate({
        "element": "Na", "x": [0.1, False], "y": [0.2, False], "z": [0.3, False], "occupancy": [1.0, False], "ADP": "Uaniso",
        "Uaniso": {"U11": [v, False], "U22": [0.01, False], "U33": [0.01, False], "U12": [0.0, False], "U13": [0.0, False], "U23": [0.0, False]}}), ("Uaniso", "U11", "value")),
]
_FLOAT_IDS = [f[0] for f in _NUMERIC_FIELDS]


@pytest.mark.parametrize("label, build, loc", _NUMERIC_FIELDS, ids=_FLOAT_IDS)
@pytest.mark.parametrize("bad", ["0.5", "1", True, False], ids=["str-float", "str-int", "true", "false"])
def test_numeric_field_rejects_strings_and_booleans(label, build, loc, bad):
    with pytest.raises(ValidationError) as exc_info:
        build(bad)
    assert _errors(exc_info) == [(loc, "float_type")]


@pytest.mark.parametrize("label, build, loc", _NUMERIC_FIELDS, ids=_FLOAT_IDS)
@pytest.mark.parametrize("good", [1, np.float64(0.5), np.int64(1)], ids=["int", "numpy-float", "numpy-int"])
def test_numeric_field_accepts_numbers(label, build, loc, good):
    model = build(good)  # no error; the value is kept as a float
    value = model
    for part in loc:
        value = value[part] if isinstance(value, (list, dict)) else getattr(value, part)
    assert value == float(good) and type(value) is float


def test_numeric_json_strings_rejected():
    """From JSON too: a quoted number is not a number."""
    with pytest.raises(ValidationError) as exc_info:
        Atom[RefinableParameter].model_validate_json(json.dumps(_atom(x=["0.25", False])))
    assert _errors(exc_info) == [(("x", "value"), "float_type")]
    assert Atom[RefinableParameter].model_validate_json(json.dumps(_atom(x=[1, False]))).x.value == 1.0


@pytest.mark.parametrize("good", [4, 4.0, np.int64(4), np.float64(4.0)])
def test_integer_fields_accept_whole_numbers(good):
    assert Atom[RefinableParameter].model_validate(_atom(Multiplicity=good)).Multiplicity == 4
    assert ChebyshevBackground(num_coefficients=good, coefficients=[0.0] * 4, refine_flag=False).num_coefficients == 4


@pytest.mark.parametrize("bad, error_type", [
    (4.5, "int_from_float"), ("4", "int_type"), (True, "int_type"), (False, "int_type"), (b"4", "int_type"),
])
def test_integer_fields_reject_fractions_strings_and_booleans(bad, error_type):
    with pytest.raises(ValidationError) as exc_info:
        Atom[RefinableParameter].model_validate(_atom(Multiplicity=bad))
    assert _errors(exc_info) == [(("Multiplicity",), error_type)]
    with pytest.raises(ValidationError) as exc_info:
        ChebyshevBackground(num_coefficients=bad, coefficients=[0.0] * 4, refine_flag=False)
    assert _errors(exc_info)[0] == (("num_coefficients",), error_type)


def test_integer_json_whole_float_accepted():
    assert Atom[RefinableParameter].model_validate_json(json.dumps(_atom(Multiplicity=4.0))).Multiplicity == 4
