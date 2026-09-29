"""Unit tests for powderline.exceptions (hierarchy, messages, warning shape).

Pure tests: no engine, no network.
"""
import pytest
from pydantic import BaseModel, ValidationError

from powderline.exceptions import (
    EngineExecutionError,
    EngineNotAvailableError,
    EngineVersionError,
    GatewayNotInstalledError,
    PowderLineError,
    RecipeValidationError,
    StructuredWarning,
    SymmetryError,
)


@pytest.mark.parametrize("cls", [
    RecipeValidationError, GatewayNotInstalledError, EngineNotAvailableError,
    EngineVersionError, EngineExecutionError, SymmetryError,
])
def test_all_derive_from_powderline_error(cls):
    assert issubclass(cls, PowderLineError)


def test_engine_not_available_is_an_import_error():
    # A52: callers that caught the underlying ImportError keep working.
    err = EngineNotAvailableError(gateway="easydiffraction", dependency="easydiffraction",
                                  env="easydiff", install_command="pixi install -e easydiff")
    with pytest.raises(ImportError):
        raise err


def test_engine_not_available_message_names_gateway_env_and_command():
    err = EngineNotAvailableError(gateway="easydiffraction", dependency="easydiffraction",
                                  env="easydiff", install_command="pixi install -e easydiff")
    msg = str(err)
    assert "'easydiffraction' gateway" in msg
    assert "'easydiff' pixi environment" in msg
    assert "pixi install -e easydiff" in msg
    assert (err.gateway, err.env) == ("easydiffraction", "easydiff")


def test_engine_version_error_message():
    err = EngineVersionError(gateway="gsasii", engine="GSAS-II", installed="5.8.0",
                             supported="==5.7.9")
    assert str(err) == ("the 'gsasii' gateway supports GSAS-II ==5.7.9, but 5.8.0 is "
                        "installed; install a supported version")
    assert (err.installed, err.supported) == ("5.8.0", "==5.7.9")


def test_gateway_not_installed_keeps_name():
    err = GatewayNotInstalledError("no gateway 'foo'", gateway="foo")
    assert err.gateway == "foo"
    assert str(err) == "no gateway 'foo'"


class _M(BaseModel):
    x: int


def test_recipe_validation_error_from_pydantic():
    with pytest.raises(ValidationError) as info:
        _M.model_validate({"x": "not-an-int"})
    err = RecipeValidationError.from_pydantic(info.value, schema_name="GSASII_Rietveld")
    assert isinstance(err, ValueError)
    assert err.schema_name == "GSASII_Rietveld"
    assert err.errors and err.errors[0]["loc"] == ("x",)
    assert err.__cause__ is info.value
    assert str(err).startswith("GSASII_Rietveld recipe failed validation:")


def test_structured_warning_shape():
    assert set(StructuredWarning.__annotations__) == {"code", "message", "field_path"}
    w: StructuredWarning = {"code": "W001", "message": "m", "field_path": None}
    assert w["field_path"] is None
