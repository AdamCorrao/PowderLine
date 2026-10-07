"""Core recipe schema (core schema 1.0.0): the models every engine schema shares.

Engine-free and pure pydantic. Engine schemas (``gsasii.*``, ``topas.*``,
``easydiffraction.*``; re/04-06) build their payloads from these models; nothing
at runtime routes through this module until then (re/03).

Conventions (master plan A19-A21, A32):

- every model is ``extra='forbid'``;
- units are fixed per field in the schema (``json_schema_extra={"unit": ...}``),
  never carried by a recipe;
- a refinable quantity is a :class:`RefinableParameter` (JSON
  ``[value, refine_flag]``) or, where the engine honors bounds, a
  :class:`BoundedRefinableParameter` (JSON ``[value, refine_flag, min, max]``);
- plain (non-refinable) values get field-specific validation;
- numbers are JSON numbers (:data:`CoreFloat`, :data:`CoreInt`): a string or a
  boolean is never read as a number.

This module must never import an engine (enforced by an import-block test).
"""

from __future__ import annotations

import inspect
import json
import keyword
import math
import re
from typing import Annotated, Any, Generic, Literal, Optional, TypeVar

import gemmi
import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    Strict,
    StrictBool,
    ValidationError,
    WithJsonSchema,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode
from pydantic_core import InitErrorDetails, PydanticCustomError

from powderline.exceptions import StructuredWarning, SymmetryError
from powderline.symmetry import (
    ADJUSTMENT_REPORT_TOL,
    Ties,
    analyze_site,
    canonical_space_group,
    cell_tie_groups,
    check_cell,
    check_uij,
    coupling_groups,
)

# --- versions (A40, A65) ---------------------------------------------------

#: The core schema version new recipes are written against.
CORE_SCHEMA_VERSION = "1.0.0"
#: Core schema versions this PowderLine accepts (PEP 440 specifier). Widened
#: explicitly when a new core version is compatible; never inferred (A40).
SUPPORTED_CORE_SCHEMAS = "==1.0.0"

#: Size cap on the serialized ``metadata`` block, in bytes of UTF-8 JSON (A64).
#: A sanity limit against accidents (e.g. a pasted pattern); amend here and in
#: docs/SCHEMA_HISTORY.md.
METADATA_MAX_BYTES = 1024 * 1024

#: ``schema_name`` format: ``"<engine>.<workflow>"`` (A1, A18).
SCHEMA_NAME_PATTERN = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"


def parse_version(field: str, value: str) -> Version:
    """Parse a PEP 440 version string, raising ``ValueError`` naming ``field``."""
    try:
        return Version(value)
    except InvalidVersion:
        raise ValueError(f"{field} {value!r} is not a valid version (PEP 440, e.g. '1.0.0')") from None


def check_core_schema_version(version: str) -> None:
    """Raise ``ValueError`` unless ``version`` is in :data:`SUPPORTED_CORE_SCHEMAS`."""
    if parse_version("core_schema_version", version) not in SpecifierSet(SUPPORTED_CORE_SCHEMAS):
        from powderline import __version__

        raise ValueError(
            f"core_schema_version {version!r} is not supported by PowderLine {__version__} "
            f"(accepted: {SUPPORTED_CORE_SCHEMAS}; current: {CORE_SCHEMA_VERSION})"
        )


# --- units (A19) ------------------------------------------------------------

#: Unit strings used in ``json_schema_extra={"unit": ...}``.
UNIT_DEG_2THETA = "degree 2theta"
UNIT_DEGREE = "degree"
UNIT_ANGSTROM = "angstrom"
UNIT_ANGSTROM2 = "angstrom^2"
UNIT_ARBITRARY = "arbitrary"
UNIT_INV_INTENSITY2 = "1/intensity^2"


def _unit(unit: str) -> dict:
    return {"unit": unit}


def _array_schema(title: str, items: list[dict], description: str, **extra) -> dict:
    """JSON Schema for a model whose recipe JSON shape is a fixed-length list."""
    return {"title": title, "description": description, "type": "array",
            "prefixItems": items, "minItems": len(items), "maxItems": len(items), **extra}


_NUMBER = {"type": "number"}
_NUMBER_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_BOOLEAN = {"type": "boolean"}


_STRICT = ConfigDict(extra="forbid")


class CoreJsonSchema(GenerateJsonSchema):
    """JSON Schema generator that names a parametrized core model after its origin.

    pydantic names ``Atom[RefinableParameter]`` ``Atom_RefinableParameter_`` in
    ``$defs``; a recipe document holds one engine's schema (one parameter type),
    so plain ``Atom`` is unambiguous and readable. Only the short candidate names
    (no module path, no id) are shortened: pydantic's content-based dedup then
    falls back to its qualified names if two parametrizations ever share one
    document, and the id-bearing name it needs unique is left intact (devkit
    ``tasks/re03b-spike-generics.md``).
    """

    def normalize_name(self, name: str) -> str:
        if "." not in name and ":" not in name:
            name = re.sub(r"\[[^\]]*\]", "", name)
        return super().normalize_name(name)


class CoreModel(BaseModel):
    """Base of the core models: ``extra='forbid'`` and :class:`CoreJsonSchema` by default.

    Engine schemas build on it (directly, or through :class:`CoreRecipe` and
    :class:`Phase`), so their generated JSON Schema is readable too.
    """

    model_config = _STRICT

    @classmethod
    def model_json_schema(cls, by_alias: bool = True, ref_template: str = DEFAULT_REF_TEMPLATE,
                          schema_generator: type[GenerateJsonSchema] = CoreJsonSchema,
                          mode: JsonSchemaMode = "validation", **kwargs) -> dict[str, Any]:
        return super().model_json_schema(by_alias, ref_template, schema_generator, mode, **kwargs)


# --- numbers -----------------------------------------------------------------


def _integer_not_text_or_bool(v: Any) -> Any:
    if isinstance(v, (bool, str, bytes, bytearray)):
        raise PydanticCustomError("int_type", "Input should be a valid integer, got {kind}",
                                  {"kind": type(v).__name__})
    return v


#: A float field: any number (int or float, incl. numpy scalars), never a
#: string or a boolean (the JSON Schema says ``number``; pydantic's default
#: would accept ``"0.25"`` and ``true``). Engine schemas use it too.
CoreFloat = Annotated[float, Strict()]
#: An integer field: a whole number (``4`` or ``4.0``, as JSON Schema's
#: ``integer``), never a string or a boolean; ``4.5`` is rejected.
CoreInt = Annotated[int, BeforeValidator(_integer_not_text_or_bool)]


# --- names ------------------------------------------------------------------

#: A phase name or atom label: an ASCII letter, then ASCII letters, digits or ``_``.
NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"
#: Longest phase name or atom label. Phase names become file names with a suffix of
#: up to 21 characters (``_unit_cell_report.csv``); file systems allow 255 per name
#: and Windows 260 per path by default, so a name stays well inside both.
NAME_MAX_LENGTH = 64
#: Python's (hard) keywords: valid by the pattern but not identifiers (``None``, ``class``).
NAME_KEYWORDS = tuple(keyword.kwlist)
#: JSON Schema of a name (also the ``propertyNames`` of a name-keyed object).
NAME_JSON_SCHEMA = {"type": "string", "pattern": NAME_PATTERN, "maxLength": NAME_MAX_LENGTH,
                    "not": {"enum": list(NAME_KEYWORDS)}}


def _check_name(name: str) -> str:
    """The name rule every engine can honor, which is also a Python identifier (re/03b review C6).

    - GSAS-II renames a phase whose name has surrounding spaces or non-ASCII
      characters, and the parameterization then skips that phase silently.
    - easydiffraction accepts atom labels only of the form ``[A-Za-z_][A-Za-z0-9_]*``.
    - TOPAS emits phase names inside quotes and builds parameter names from them.
    - Phase names become report file names (``<phase>_unit_cell_report.csv``), so
      no character a file system rejects (``/``, ``:``, ...) and a bounded length.
    - Field paths are joined with ``.`` (``atoms.O1.Uiso``).
    """
    if not name.strip():
        raise ValueError("a name must not be empty or blank")
    if re.fullmatch(NAME_PATTERN, name) is None:
        raise ValueError(f"name {name!r} must start with a letter and contain only ASCII letters, digits "
                         "and '_' (e.g. 'O1', 'LaB6_a')")
    if len(name) > NAME_MAX_LENGTH:
        raise ValueError(f"name {name[:20]!r}... has {len(name)} characters; the limit is {NAME_MAX_LENGTH}")
    if keyword.iskeyword(name):
        raise ValueError(f"name {name!r} is a Python keyword; choose another name")
    return name


#: A phase name or an atom label (a dict key), see :func:`_check_name`. Core uses
#: it for the atom labels; engine schemas use it for their ``phases`` keys (with
#: :func:`name_keyed`, and :func:`check_names_unique_ignoring_case` for phases).
CoreName = Annotated[str, AfterValidator(_check_name), WithJsonSchema(NAME_JSON_SCHEMA)]


def check_names_unique_ignoring_case(names, what: str) -> None:
    """Raise ``ValueError`` if two of ``names`` differ only in case.

    Phase names become file names, and Windows and macOS file systems ignore
    case, so ``LaB6`` and ``lab6`` would overwrite each other's reports (and
    easydiffraction lowercases them). Engine payloads call it on their phases.
    """
    seen: dict[str, str] = {}
    for name in names:
        other = seen.setdefault(name.lower(), name)
        if other != name:
            raise ValueError(f"{what} {other!r} and {name!r} differ only in case; rename one")


def name_keyed(schema: dict) -> None:
    """``json_schema_extra`` for a ``dict[CoreName, ...]`` field: state the key rule as ``propertyNames``.

    pydantic writes a key rule as ``patternProperties`` (pattern only), which
    leaves other keys unchecked in JSON Schema; this checks every key against
    :data:`NAME_JSON_SCHEMA`.
    """
    pattern_properties = schema.pop("patternProperties", None)
    if pattern_properties:
        ((_pattern, value),) = pattern_properties.items()
        schema["propertyNames"] = {k: v for k, v in NAME_JSON_SCHEMA.items() if k != "type"}
        schema["additionalProperties"] = value


# --- parameter models (A20, A32, A93) ---------------------------------------


def _finite(name: str, v: Optional[float]) -> Optional[float]:
    if v is not None and not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


class RefinableParameter(BaseModel):
    """A refinable value with no bounds concept: JSON ``[value, refine_flag]``.

    Used where the engine has no native bounds for the quantity. A 4-element
    list here is an error, never silently truncated (A20). ``value`` is
    required (never ``null``, A93/A96).
    """

    model_config = _STRICT

    value: CoreFloat
    refine_flag: StrictBool

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            if len(data) == 4:
                raise ValueError(
                    "bounds are not supported for this parameter in this engine schema: "
                    "expected [value, refine_flag], got a 4-element "
                    "[value, refine_flag, min, max] list"
                )
            if len(data) != 2:
                raise ValueError(
                    f"expected [value, refine_flag], got a {len(data)}-element list"
                )
            return {"value": data[0], "refine_flag": data[1]}
        raise ValueError(
            f"expected a [value, refine_flag] list, got {type(data).__name__}"
        )

    @field_validator("value")
    @classmethod
    def _value_finite(cls, v):
        return _finite("value", v)

    @model_serializer
    def _to_list(self) -> list:
        return [self.value, self.refine_flag]

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler) -> dict:
        return _array_schema(cls.__name__, [_NUMBER, _BOOLEAN], "[value, refine_flag]")


class BoundedRefinableParameter(BaseModel):
    """A refinable value with optional bounds: JSON ``[value, refine_flag, min, max]``.

    ``min``/``max`` may each be ``null`` (unbounded on that side). When given,
    ``min <= max`` and ``min <= value <= max``. Used only where the engine
    honors the bounds natively (A20, A33). ``value`` is required (never
    ``null``, A93/A96).
    """

    model_config = _STRICT

    value: CoreFloat
    refine_flag: StrictBool
    min: Optional[CoreFloat] = None
    max: Optional[CoreFloat] = None

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            if len(data) != 4:
                raise ValueError(
                    "expected [value, refine_flag, min, max] (min/max may be null), "
                    f"got a {len(data)}-element list"
                )
            return {"value": data[0], "refine_flag": data[1], "min": data[2], "max": data[3]}
        raise ValueError(
            f"expected a [value, refine_flag, min, max] list, got {type(data).__name__}"
        )

    @field_validator("value", "min", "max")
    @classmethod
    def _finite(cls, v, info):
        return _finite(info.field_name, v)

    @model_validator(mode="after")
    def _bounds(self) -> "BoundedRefinableParameter":
        lo, hi = self.min, self.max
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"min ({lo}) must be <= max ({hi})")
        check_within_bounds(self.value, self, "value")
        return self

    @model_serializer
    def _to_list(self) -> list:
        return [self.value, self.refine_flag, self.min, self.max]

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler) -> dict:
        return _array_schema(cls.__name__, [_NUMBER, _BOOLEAN, _NUMBER_OR_NULL, _NUMBER_OR_NULL],
                             "[value, refine_flag, min, max]; min/max may be null")


def check_within_bounds(value: float, param: BoundedRefinableParameter, field: str) -> None:
    """Raise ``ValueError`` unless ``param.min <= value <= param.max`` (open ends pass)."""
    if param.min is not None and value < param.min:
        raise ValueError(f"{field} ({value}) is below min ({param.min})")
    if param.max is not None and value > param.max:
        raise ValueError(f"{field} ({value}) is above max ({param.max})")


# --- plain-value checks (A32) ----------------------------------------------


def check_element_symbol(symbol: str) -> str:
    """Return ``symbol`` if it is a bare element symbol in gemmi's table, exactly spelled.

    Raises ``ValueError`` otherwise. gemmi's own lookup is too lenient to use
    alone: it accepts any case (``"FE"``), reduces charged types to the neutral
    atom (``"Fe3+"`` -> Fe, which would change the X-ray scattering), and maps
    unknown symbols to ``X``. Charged scattering types are not supported in core
    schema 1.0.0 yet (A63 obligation); they are rejected, never stripped.
    """
    element = gemmi.Element(symbol)
    if element.atomic_number > 0 and element.name == symbol:
        return symbol
    if any(c in symbol for c in "+-") or (symbol[-1:].isdigit()):
        raise ValueError(
            f"element {symbol!r}: charged scattering types are not supported in core schema "
            f"{CORE_SCHEMA_VERSION} yet; give the bare element symbol"
        )
    # Suggest a spelling only for case/whitespace differences. gemmi also reads
    # a prefix (' Fe' -> F, 'Nax' -> Na), which must never be offered as a fix.
    stripped = gemmi.Element(symbol.strip())
    if stripped.atomic_number > 0 and stripped.name.lower() == symbol.strip().lower():
        raise ValueError(f"element {symbol!r}: write the element symbol as {stripped.name!r}")
    raise ValueError(f"element {symbol!r} is not a known element symbol")


# --- diffraction data -------------------------------------------------------


class XRDData(BaseModel):
    """Measured pattern: 2theta, intensity, weights (ported from schema 0.26 ``XRDDataModel``).

    Every engine consumes ``tth`` in degrees 2theta and ``Itth_weights`` as
    1/sigma^2, as-is.
    """

    model_config = _STRICT

    tth: list[CoreFloat] = Field(description="Two-theta values", json_schema_extra=_unit(UNIT_DEG_2THETA))
    Itth: list[CoreFloat] = Field(description="Intensity values", json_schema_extra=_unit(UNIT_ARBITRARY))
    Itth_weights: list[CoreFloat] = Field(description="Intensity weights (1/sigma^2)",
                                      json_schema_extra=_unit(UNIT_INV_INTENSITY2))
    filename: str | None = Field(default=None, description="Original filename for reference")

    @model_validator(mode='after')
    def validate_all_arrays_same_length(self):
        """Ensure tth, Itth, and Itth_weights have the same length."""
        lengths = [len(self.tth), len(self.Itth), len(self.Itth_weights)]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"XRD data arrays must have the same length. "
                f"Got tth={len(self.tth)}, Itth={len(self.Itth)}, Itth_weights={len(self.Itth_weights)}"
            )
        return self

    @model_validator(mode='after')
    def validate_arrays_wellformed(self):
        """Reject ill-formed XRD data. PowderLine never repairs data — it rejects it.

        Transforming raw data (unit conversion, weight derivation, handling
        background-subtracted negatives) is the job of the reader/parser that builds
        this block, not PowderLine. Here we only enforce that what arrives is usable:

        * arrays must be non-empty;
        * every tth / Itth / Itth_weights value must be finite (no NaN or inf) —
          this catches, e.g., a weight computed as 1/esd**2 with esd == 0 (inf);
        * tth must be strictly increasing (a powder pattern is monotonic in 2-theta);
        * weights must be >= 0 (a 0 weight legitimately excludes a point; a negative
          weight is nonsensical) and at least one weight must be > 0 (all-zero =
          nothing to fit).

        A negative *intensity* is explicitly allowed: background-subtracted data
        legitimately dips below zero, so Itth is checked for finiteness only, not sign.
        """
        if len(self.tth) == 0:
            raise ValueError(
                "XRD data arrays must be non-empty (tth/Itth/Itth_weights have length 0)."
            )

        tth = np.asarray(self.tth, dtype=float)
        Itth = np.asarray(self.Itth, dtype=float)
        weights = np.asarray(self.Itth_weights, dtype=float)

        # All values finite (rejects NaN and +/-inf across all three arrays).
        for name, arr in (("tth", tth), ("Itth", Itth), ("Itth_weights", weights)):
            bad = np.where(~np.isfinite(arr))[0]
            if bad.size:
                i = int(bad[0])
                raise ValueError(
                    f"XRD data '{name}' contains non-finite values (NaN/inf); first at "
                    f"index {i} (value {float(arr[i])!r}). Fix the data upstream — "
                    "PowderLine does not repair xrd_data."
                )

        # tth must be strictly increasing.
        bad = np.where(np.diff(tth) <= 0)[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"XRD 'tth' must be strictly increasing; tth[{i + 1}]={float(tth[i + 1])!r} "
                f"<= tth[{i}]={float(tth[i])!r}. Sort/deduplicate the pattern upstream."
            )

        # Weights: non-negative, with at least one strictly positive.
        bad = np.where(weights < 0)[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"XRD 'Itth_weights' must be >= 0; first negative at index {i} "
                f"(value {float(weights[i])!r}). A 0 weight excludes a point; "
                "negative weights are invalid."
            )
        if not np.any(weights > 0):
            raise ValueError(
                "XRD 'Itth_weights' are all zero — nothing to fit. At least one "
                "weight must be > 0."
            )

        return self


class FitRange(BaseModel):
    """Fit window in 2theta: JSON ``[min, max]``; either end may be ``null`` (open)."""

    model_config = _STRICT

    min: Optional[CoreFloat] = None
    max: Optional[CoreFloat] = None

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, data: Any) -> Any:
        if isinstance(data, (list, tuple)):
            if len(data) != 2:
                raise ValueError(
                    f"fit_range must be a list of exactly 2 elements [min, max], got {len(data)}"
                )
            return {"min": data[0], "max": data[1]}
        raise ValueError(f"fit_range must be a [min, max] list, got {type(data).__name__}")

    @field_validator("min", "max")
    @classmethod
    def _finite(cls, v, info):
        return _finite(info.field_name, v)

    @model_validator(mode="after")
    def _ordered(self) -> "FitRange":
        if self.min is not None and self.max is not None and not self.max > self.min:
            raise ValueError(f"fit_range max ({self.max}) must be greater than min ({self.min})")
        return self

    @model_serializer
    def _to_list(self) -> list:
        return [self.min, self.max]

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler) -> dict:
        return _array_schema("FitRange", [_NUMBER_OR_NULL, _NUMBER_OR_NULL],
                             "[min, max]; either end may be null (open)",
                             unit=UNIT_DEG_2THETA)


def check_fit_range_within_data(fit_range: Optional[FitRange], xrd_data: XRDData) -> None:
    """Raise ``ValueError`` unless both ``fit_range`` ends lie within the data's 2theta span.

    A payload-level check (it needs the pattern): engine payload models call it
    from a model validator (A68). ``None`` (or an open end) means "the data limit".
    """
    if fit_range is None:
        return
    lo, hi = xrd_data.tth[0], xrd_data.tth[-1]
    for name, end in (("min", fit_range.min), ("max", fit_range.max)):
        if end is not None and not lo <= end <= hi:
            raise ValueError(
                f"fit_range {name} ({end}) is outside the data's 2theta range [{lo}, {hi}]"
            )


# --- background -------------------------------------------------------------


class ChebyshevBackground(BaseModel):
    """Chebyshev polynomial background (ported from schema 0.26).

    Coefficient semantics (scaling, which polynomial basis) are engine-specific
    and documented in each engine schema (A22.7), not here.
    """

    model_config = _STRICT

    num_coefficients: CoreInt = Field(gt=0, description="Number of Chebyshev coefficients")
    coefficients: list[CoreFloat] = Field(description="Coefficient values")
    refine_flag: StrictBool = Field(description="Whether to refine background")

    @field_validator('coefficients')
    @classmethod
    def validate_coefficients_length(cls, v, info):
        """Ensure number of coefficients matches num_coefficients."""
        num_coef = info.data.get('num_coefficients')
        if num_coef is not None and len(v) != num_coef:
            raise ValueError(f"Number of coefficients ({len(v)}) must match num_coefficients ({num_coef})")
        return v

    @field_validator('coefficients')
    @classmethod
    def _coefficients_finite(cls, v):
        for i, c in enumerate(v):
            _finite(f"coefficients[{i}]", c)
        return v


# --- phase block (A93-A96, A100, A105; values: A63, A69, A73, A76-A78) -------

#: The parameter type of a phase block: :class:`RefinableParameter`, or
#: :class:`BoundedRefinableParameter` for an engine that honors bounds (A20).
#: One type for every structural field, cell and atoms (review N6).
P = TypeVar("P", RefinableParameter, BoundedRefinableParameter)


def _absent_not_null(v: Any, info) -> Any:
    """A field that may be left out is never ``null`` (A96): reject an explicit null."""
    if v is None:
        raise ValueError(f"{info.field_name} must not be null; leave it out instead")
    return v


def _absent_not_null_schema(unit: Optional[str] = None):
    """JSON Schema of a field that may be left out but is never null (A96).

    The field is ``Optional`` only so that "left out" can be held as ``None``;
    the schema shows the non-null type, no ``"default": null``, and the unit.
    """
    def extra(schema: dict) -> None:
        schema.pop("default", None)
        any_of = schema.pop("anyOf", None)
        if any_of:
            schema.update(next(s for s in any_of if s.get("type") != "null"))
        if unit:
            schema.update(_unit(unit))
    return extra


class UnitCell(CoreModel, Generic[P]):
    """Cell constants, each a refinable parameter. No ``volume``: every engine
    derives it from a..gamma (A75). The values must fit the space group (A77,
    checked by :class:`Phase`)."""

    a: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM))
    b: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM))
    c: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM))
    alpha: P = Field(json_schema_extra=_unit(UNIT_DEGREE))
    beta: P = Field(json_schema_extra=_unit(UNIT_DEGREE))
    gamma: P = Field(json_schema_extra=_unit(UNIT_DEGREE))

    @field_validator("a", "b", "c")
    @classmethod
    def _length(cls, v, info):
        if not v.value > 0:
            raise ValueError(f"cell length {info.field_name} must be positive, got {v.value}")
        if isinstance(v, BoundedRefinableParameter) and v.min is not None and not v.min > 0:
            raise ValueError(f"cell length {info.field_name}: min must be positive (a cell length is), "
                             f"got {v.min}; use null for no lower bound")
        return v

    @field_validator("alpha", "beta", "gamma")
    @classmethod
    def _angle(cls, v, info):
        if not 0 < v.value < 180:
            raise ValueError(f"cell angle {info.field_name} must be between 0 and 180 degrees, got {v.value}")
        if isinstance(v, BoundedRefinableParameter):
            for side, bound in (("min", v.min), ("max", v.max)):
                if bound is not None and not 0 < bound < 180:
                    raise ValueError(f"cell angle {info.field_name}: {side} must be between 0 and 180 degrees "
                                     f"(a cell angle is), got {bound}; use null for no {side} bound")
        return v


_UANISO_KEYS = ("U11", "U22", "U33", "U12", "U13", "U23")


class UanisoTensor(CoreModel, Generic[P]):
    """Anisotropic ADPs: all six of U11, U22, U33, U12, U13, U23, each a parameter (A^2)."""

    U11: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM2))
    U22: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM2))
    U33: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM2))
    U12: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM2))
    U13: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM2))
    U23: P = Field(json_schema_extra=_unit(UNIT_ANGSTROM2))


#: JSON Schema of the ADP pairing (A96): ``ADP`` selects which one of ``Uiso`` /
#: ``Uaniso`` is required; the other must be left out.
_ADP_PAIRING_SCHEMA = {"allOf": [
    {"if": {"properties": {"ADP": {"const": adp}}},
     "then": {"required": [adp], "not": {"required": [other]}}}
    for adp, other in (("Uiso", "Uaniso"), ("Uaniso", "Uiso"))
]}


class Atom(CoreModel, Generic[P]):
    """One atom of a phase: coordinates, occupancy and ADPs are refinable parameters.

    Coordinates are fractional JSON numbers. A special position is stated
    exactly, to floating-point precision (``0.3333333333333333`` for 1/3); a
    position near one but not on it is an error giving the exact values (A73,
    A115). ``occupancy`` is required
    and 0 to 1 inclusive (A76, A96). ``Multiplicity`` is optional and, when
    stated, must match the derived value (A69 T3). ``ADP`` selects which
    thermal parameter is **required**: ``Uiso``, or ``Uaniso`` with all six of
    U11..U23; the other must be left out (an error at that field otherwise, and
    the JSON Schema states the rule). ``Uaniso`` must respect the site symmetry
    exactly; otherwise it is an error, giving the symmetric values when they are
    within 1e-6 A^2 (A78, A115). Nothing is ``null`` (A96).
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_ADP_PAIRING_SCHEMA)

    element: str = Field(description="Bare element symbol, exactly as in the periodic table (e.g. 'Fe')")
    x: P = Field(description="Fractional x coordinate")
    y: P = Field(description="Fractional y coordinate")
    z: P = Field(description="Fractional z coordinate")
    occupancy: P = Field(description="Site occupancy, 0 to 1 inclusive (A76)")
    Multiplicity: Optional[CoreInt] = Field(default=None, json_schema_extra=_absent_not_null_schema(),
                                            description="Site multiplicity; optional, cross-checked when stated")
    ADP: Literal["Uiso", "Uaniso"] = Field(description="Displacement-parameter type")
    Uiso: Optional[P] = Field(default=None, json_schema_extra=_absent_not_null_schema(UNIT_ANGSTROM2),
                              description="Isotropic ADP; required when ADP is 'Uiso', else left out")
    Uaniso: Optional[UanisoTensor[P]] = Field(
        default=None, json_schema_extra=_absent_not_null_schema(),
        description="All six of U11, U22, U33, U12, U13, U23; required when ADP is 'Uaniso', else left out")

    _not_null = field_validator("Multiplicity", "Uiso", "Uaniso", mode="before")(_absent_not_null)

    @field_validator("element")
    @classmethod
    def _element(cls, v: str) -> str:
        return check_element_symbol(v)

    @field_validator("occupancy")
    @classmethod
    def _occupancy(cls, v):
        if not 0.0 <= v.value <= 1.0:
            raise ValueError(f"occupancy must be between 0 and 1 inclusive, got {v.value}")
        return v

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        """Leave out absent optional fields, so a dump validates again (no ``null``, A96)."""
        data = handler(self)
        for name in ("Multiplicity", "Uiso", "Uaniso"):
            if getattr(self, name) is None:
                data.pop(name, None)
        return data

    @model_validator(mode="after")
    def _adp_block(self) -> "Atom":
        """``ADP`` selects the required thermal parameter; errors at the field concerned."""
        required, other = ("Uiso", "Uaniso") if self.ADP == "Uiso" else ("Uaniso", "Uiso")
        problems = []
        if getattr(self, required) is None:
            problems.append(InitErrorDetails(
                type=PydanticCustomError("missing", "{field} is required when ADP is '{adp}'",
                                         {"field": required, "adp": self.ADP}),
                loc=(required,), input=None))
        if getattr(self, other) is not None:
            problems.append(InitErrorDetails(
                type=PydanticCustomError("extra_forbidden", "{field} must be left out when ADP is '{adp}'",
                                         {"field": other, "adp": self.ADP}),
                loc=(other,), input=getattr(self, other).model_dump()))
        if problems:
            raise ValidationError.from_exception_data(type(self).__name__, problems)
        return self


_OPEN_PARAMETER_TYPE = (
    "{name} has no parameter type: subclass Phase[RefinableParameter] (no bounds) or "
    "Phase[BoundedRefinableParameter] (bounds); a deliberately generic subclass declares "
    "Generic[P] and is parametrized before use"
)

#: Phase fields owned by core; an engine subclass may not redeclare them (A93).
RESERVED_PHASE_FIELDS = ("space_group", "unit_cell", "atoms")


class Phase(CoreModel, Generic[P]):
    """A phase block's core part: space group, cell and atoms (A93).

    Engine schemas subclass ``Phase[TheirParameterType]`` and add their own
    phase fields flat (gsasii: ``scale``, ``peak_broadening``); the phase name
    is the key of the payload's ``phases`` dict (A100). Core's field names are
    reserved (:data:`RESERVED_PHASE_FIELDS`): redeclaring one raises
    ``TypeError`` when the subclass is defined.

    ``space_group`` is gemmi's canonical name (A105). Validation derives each
    atom's site (multiplicity, DOF) from it (:func:`powderline.symmetry.analyze_site`)
    and rejects ambiguous positions and stated multiplicities that don't match.
    The unit cell must fit the space group exactly (A77).

    **Validation never changes a recipe** (A115): the recipe is the record of the
    refinement intent, so every value must already be exactly what the engines
    use. A special position must be stated exactly (A73), an anisotropic ADP
    exactly site-symmetric (A78), and tied bounds must follow their tie exactly
    (A108). Otherwise it is an error that gives the exact value to write.
    "Exactly" means to floating-point precision
    (:data:`~powderline.symmetry.ADJUSTMENT_REPORT_TOL`), because a JSON number
    cannot hold 1/3: ``0.3333333333333333`` is the value to write.

    Refine flags and bounds must agree with the symmetry ties
    (:func:`~powderline.symmetry.cell_tie_groups`,
    :func:`~powderline.symmetry.coupling_groups`; A94, A95, A98, A106):
    the members of one tie group carry the same flag (F1); a symmetry-fixed
    parameter has flag false (F2) and no bounds (A106); different groups are
    independent (F3); with bounds, each member's bounds follow the tie's
    relation from the first member's (F4: ``member = k*rep + c`` maps
    ``[min, max]`` to ``[k*min + c, k*max + c]``, swapped for k < 0). Atoms on
    the same site are independent (A104).

    Every problem is reported at its own location (A70): ``unit_cell``,
    ``atoms.<label>`` (the position), ``atoms.<label>.Multiplicity``,
    ``atoms.<label>.Uaniso``; a flag or bound rule at the parameter
    (``unit_cell.b``, ``atoms.<label>.y``, ``atoms.<label>.Uaniso.U12``), for F1
    at every member whose flag differs from the group's first member (review
    N1); all of them together, in one ``ValidationError``.
    These checks run only once every field is valid: pydantic does not run
    model-level validators after a field error, so e.g. an unknown element is
    reported first, and the symmetry checks follow once it is fixed.

    :meth:`warnings` holds informative findings that change nothing: a
    ``Uaniso`` that is not positive definite (A118).
    """

    space_group: str = Field(description="gemmi's canonical extended Hermann-Mauguin name, e.g. "
                                         "'P m -3 m', 'C 1 2/m 1', 'R -3 m:H', 'F d -3 m:2' (A105)")
    unit_cell: UnitCell[P]
    atoms: dict[CoreName, Atom[P]] = Field(
        min_length=1, json_schema_extra=name_keyed,
        description="Atoms keyed by unique label: a letter, then ASCII letters, digits or '_'")

    _sites: dict = PrivateAttr(default_factory=dict)
    _warnings: list = PrivateAttr(default_factory=list)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Guard core's contract in every engine subclass (A93; re/03b review).

        A subclass may add fields and validators, but may not redeclare core's
        fields or reuse the name of a core validator, private attribute or
        method: pydantic collects validators by name, so a same-named engine
        validator would silently replace core's (e.g. the canonical space-group
        check or every symmetry rule). Its ``extra`` must stay ``'forbid'`` (A19).
        """
        super().__pydantic_init_subclass__(**kwargs)
        for klass in cls.__mro__:
            if klass is Phase or not (isinstance(klass, type) and issubclass(klass, Phase)):
                continue
            if klass.__pydantic_generic_metadata__["origin"] is not None:
                continue  # a parametrization (e.g. Phase[RefinableParameter]); its origin is checked itself
            annotations = inspect.get_annotations(klass)
            redeclared = [n for n in RESERVED_PHASE_FIELDS if n in annotations]
            if redeclared:
                raise TypeError(
                    f"{klass.__name__} redeclares the core phase field(s) {', '.join(redeclared)}; "
                    "core owns space_group, unit_cell and atoms (A93)"
                )
            reused = sorted(n for n in _reserved_phase_members() if n in klass.__dict__ or n in annotations)
            if reused:
                raise TypeError(
                    f"{klass.__name__} reuses the core phase name(s) {', '.join(reused)}; a same-named "
                    "validator, method or private attribute would replace core's silently; rename it"
                )
        if cls.model_config.get("extra") != "forbid":
            raise TypeError(f"{cls.__name__} must keep extra='forbid' (A19)")
        if cls.__pydantic_generic_metadata__["parameters"] and "__orig_bases__" not in cls.__dict__:
            # ``class EnginePhase(Phase)`` forgot the parameter type. A deliberately generic
            # subclass declares ``Generic[P]`` and is checked again when it is validated.
            raise TypeError(_OPEN_PARAMETER_TYPE.format(name=cls.__name__))

    @model_validator(mode="before")
    @classmethod
    def _parameter_type_chosen(cls, data: Any) -> Any:
        """Refuse to validate without a parameter type (re/03b review A2).

        Unparametrized, ``P`` would accept both parameter shapes, so one phase could
        mix ``[value, flag]`` and ``[value, flag, min, max]``. An engine schema fixes
        one type; this is a programming error, so it is a ``TypeError``, not a
        validation error.
        """
        if cls.__pydantic_generic_metadata__["parameters"]:
            raise TypeError(_OPEN_PARAMETER_TYPE.format(name=cls.__name__))
        return data

    @field_validator("space_group")
    @classmethod
    def _space_group(cls, v: str) -> str:
        canonical_space_group(v)  # SymmetryError is a ValueError
        return v

    @model_validator(mode="after")
    def _sites_check(self) -> "Phase":
        problems: list[InitErrorDetails] = []

        def problem(loc: tuple, message: str, value: Any) -> None:
            problems.append(InitErrorDetails(type="value_error", loc=loc, input=value,
                                             ctx={"error": ValueError(message)}))

        def tie_rules(loc: tuple, params: dict, ties: Ties, context: str, fixed_value) -> None:
            for name, message in _tie_rules(params, ties, context, fixed_value):
                problem((*loc, name), message, params[name].model_dump())

        sg = self.space_group
        cell = self.unit_cell
        try:
            check_cell(sg, tuple(getattr(cell, k).value for k in _CELL_FIELDS))
        except SymmetryError as exc:
            problem(("unit_cell",), str(exc), cell.model_dump())
        else:  # flag and bound rules need a valid cell (A95, A98)
            ties = cell_tie_groups(sg)
            tie_rules(("unit_cell",), {k: getattr(cell, k) for k in _CELL_FIELDS}, ties,
                      f"{ties.crystal_system} cell ({sg!r})", lambda name: f"{getattr(cell, name).value:g}")
        for label, atom in self.atoms.items():
            try:
                site = analyze_site(sg, (atom.x.value, atom.y.value, atom.z.value))
            except SymmetryError as exc:
                problem(("atoms", label), str(exc), atom.model_dump())
                continue  # the checks below need the site
            self._sites[label] = site
            context = f"atom {label!r} (multiplicity {site.multiplicity} in {sg!r})"
            ties = coupling_groups(sg, site.canonical)
            exact_xyz = lambda name: site.exact["xyz".index(name)]  # noqa: E731
            if site.adjusted:  # A73, A115: on a special position, but not exactly
                off = max(abs(c - v) for c, v in zip(site.canonical, site.stated))
                problem(("atoms", label),
                        f"{context}: {site.stated} is {off:.2g} from a special position, not on it "
                        f"({_ties_text(ties.xyz, exact_xyz)}); write "
                        f"{_values_to_write(ties.xyz, dict(zip('xyz', site.stated)), exact_xyz)}",
                        atom.model_dump())
            if atom.Multiplicity is not None and atom.Multiplicity != site.multiplicity:
                problem(("atoms", label, "Multiplicity"),
                        f"stated Multiplicity {atom.Multiplicity}, derived {site.multiplicity} "
                        f"({sg!r}, site {site.stated})", atom.Multiplicity)
            uaniso_ok = False
            if atom.Uaniso is not None:
                stated_u = tuple(getattr(atom.Uaniso, k).value for k in _UANISO_KEYS)
                try:
                    symmetric_u, u_adjusted = check_uij(sg, site.canonical, stated_u)
                except SymmetryError as exc:
                    problem(("atoms", label, "Uaniso"), str(exc), atom.model_dump()["Uaniso"])
                else:
                    uaniso_ok = True
                    principal = np.linalg.eigvalsh(_uij_matrix(stated_u))
                    if principal[0] <= 0:  # informative, not an error (A118): never changes a value
                        self._warnings.append(StructuredWarning(
                            code="uaniso_not_positive_definite",
                            message=(f"{context}: Uaniso is not positive definite (principal values "
                                     f"{', '.join(f'{v:.3g}' for v in principal)} A^2), so it describes no "
                                     "thermal ellipsoid; it is used as stated"),
                            field_path=f"atoms.{label}.Uaniso",
                        ))
                    if u_adjusted:  # A78, A115: within 1e-6 A^2 of symmetric, but not exactly
                        problem(("atoms", label, "Uaniso"),
                                f"{context}: Uaniso is not exactly site-symmetric "
                                f"({_ties_text(ties.uij, lambda _name: 0)}); write "
                                f"{_values_to_write(ties.uij, dict(zip(_UANISO_KEYS, stated_u)), lambda _n: 0)}",
                                atom.model_dump()["Uaniso"])

            # Flag and bound rules F1, F2, F4, A106 (A95, A98, A106).
            tie_rules(("atoms", label), {k: getattr(atom, k) for k in "xyz"}, ties.xyz, context,
                      lambda name: str(exact_xyz(name)))
            if uaniso_ok:
                tie_rules(("atoms", label, "Uaniso"), {k: getattr(atom.Uaniso, k) for k in _UANISO_KEYS},
                          ties.uij, context, lambda _name: "0")
        if problems:
            raise ValidationError.from_exception_data(type(self).__name__, problems)
        return self

    def site(self, label: str):
        """The :class:`~powderline.symmetry.SiteAnalysis` of atom ``label``."""
        return self._sites[label]

    def warnings(self) -> list:
        """Structured warnings from validating this phase (paths relative to it).

        Informative only: a warning never comes with a changed value (A115), so
        validating the same recipe again gives the same warnings.
        """
        return list(self._warnings)


def _reserved_phase_members() -> frozenset[str]:
    """Names core's :class:`Phase` defines besides its fields: validators, private attributes, methods."""
    decorators = Phase.__pydantic_decorators__
    validators = {n for kind in ("validators", "field_validators", "root_validators", "field_serializers",
                                 "model_serializers", "model_validators", "computed_fields")
                  for n in getattr(decorators, kind)}
    # Methods written in Phase itself (not the ones pydantic or abc generate, e.g. model_post_init).
    methods = {n for n, v in Phase.__dict__.items()
               if inspect.isfunction(v) and not n.startswith("__") and v.__qualname__.startswith("Phase.")}
    return frozenset(validators | methods | set(Phase.__private_attributes__))


_CELL_FIELDS = ("a", "b", "c", "alpha", "beta", "gamma")


def _uij_matrix(u) -> np.ndarray:
    """The symmetric 3x3 matrix of (U11, U22, U33, U12, U13, U23).

    Its eigenvalues have the signs of the Cartesian tensor's (a congruence
    transform keeps them, Sylvester's law of inertia), so it tells whether the
    thermal ellipsoid exists without the cell.
    """
    u11, u22, u33, u12, u13, u23 = u
    return np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]], dtype=float)


def _same(got: float, want: float) -> bool:
    """Equal to floating-point precision (``ADJUSTMENT_REPORT_TOL``, relative above 1)."""
    return abs(got - want) <= ADJUSTMENT_REPORT_TOL * max(1.0, abs(want))


def _ties_text(ties: Ties, exact) -> str:
    """The symmetry relations of a parameter set, e.g. ``"x = 1/3, y = 2/3"`` or ``"y = 2x"``."""
    parts = [f"{name} = {exact(name)}" for name in ties.fixed]
    parts += [g.relations_text() for g in ties.groups if g.relations_text()]
    return ", ".join(parts)


def _values_to_write(ties: Ties, stated: dict, exact) -> str:
    """``"y = 0.2468"``: the values that make ``stated`` exactly symmetric, where they differ.

    Each tie group keeps its first member's stated value (the one parameter,
    A94) and derives the others from it; fixed parameters take their exact
    value. ``repr`` of a float reads back as the same float, so a value can be
    copied into a recipe as-is (``0.3333333333333333`` for 1/3).
    """
    target = {name: float(exact(name)) for name in ties.fixed}
    for group in ties.groups:
        rep = stated[group.members[0]]
        target.update({name: float(k) * rep + float(c)
                       for name, k, c in zip(group.members, group.coefficients, group.offsets)})
    return ", ".join(f"{name} = {target[name]!r}" for name in stated if not _same(stated[name], target[name]))


def _fmt_bounds(lo: Optional[float], hi: Optional[float]) -> str:
    return "[" + ", ".join("null" if v is None else repr(v) for v in (lo, hi)) + "]"


_HOW_MANY = {2: "both", 3: "all three"}


def _tie_rules(params: dict, ties: Ties, context: str, fixed_value) -> list[tuple[str, str]]:
    """The flag and bound rules of one parameter set (cell, a site's x/y/z, or its Uij).

    - **F2** (A95): a symmetry-fixed parameter has refine flag false.
    - **A106**: a symmetry-fixed parameter has no bounds (min and max null).
    - **F1** (A95): the members of a tie group carry the same flag. The error is
      reported at every member whose flag differs from the group's first
      member, naming the whole group (review N1).
    - **F4** (A98, A115): with bounds, each member's bounds follow the tie from
      the first member's: ``member = k*rep + c`` maps ``[min, max]`` to
      ``[k*min + c, k*max + c]``, swapped for k < 0; a null side stays null on
      the matching side. Anything but the exact mapped bounds (to floating-point
      precision) is an error giving them.
    - **F3**: different groups are independent (nothing to check).

    Returns ``[(name, message)]``.
    """
    errors: list[tuple[str, str]] = []
    for name in ties.fixed:
        param = params[name]
        if param.refine_flag:
            errors.append((name, f"{context}: {name} is fixed by symmetry (= {fixed_value(name)}); "
                                 "set its refine flag to false"))
        if isinstance(param, BoundedRefinableParameter) and (param.min is not None or param.max is not None):
            errors.append((name, f"{context}: {name} is fixed by symmetry (= {fixed_value(name)}); "
                                 "give it no bounds (min and max null)"))
    for group in ties.groups:
        if len(group.members) < 2:
            continue
        rep = params[group.members[0]]
        flag_words = _HOW_MANY.get(len(group.members), f"all {len(group.members)}")
        relations = group.relations_text()
        for name in group.members[1:]:
            if params[name].refine_flag != rep.refine_flag:
                errors.append((name, f"{context}: {', '.join(group.members)} are one parameter"
                                     f"{f' ({relations})' if relations else ''}; "
                                     f"set the same refine flag on {flag_words}"))
        if not isinstance(rep, BoundedRefinableParameter):
            continue
        for name in group.members[1:]:
            param = params[name]
            k, c = (float(v) for v in group.relation(name))
            lo, hi = (None if v is None else k * v + c for v in (rep.min, rep.max))
            if k < 0:
                lo, hi = hi, lo
            if not all((got is None and want is None) or (got is not None and want is not None and _same(got, want))
                       for got, want in ((param.min, lo), (param.max, hi))):
                errors.append((name, f"{context}: the bounds of {name} must follow "
                                     f"{group.relation_text(name)} from the bounds of {group.members[0]} "
                                     f"{_fmt_bounds(rep.min, rep.max)}: write {_fmt_bounds(lo, hi)}, "
                                     f"not {_fmt_bounds(param.min, param.max)}"))
    return errors


# --- top-level recipe frame (A1, A18, A21, A40) ------------------------------


def _check_metadata(v: Optional[dict]) -> Optional[dict]:
    if v is None:
        return v
    try:
        text = json.dumps(v, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from None
    size = len(text.encode("utf-8"))
    if size > METADATA_MAX_BYTES:
        raise ValueError(
            f"metadata is {size} bytes as JSON; the limit is {METADATA_MAX_BYTES} "
            "(schema_core.METADATA_MAX_BYTES)"
        )
    return v


class CoreRecipe(CoreModel):
    """The core-owned top level of every recipe; engine schemas subclass it.

    An engine schema narrows ``schema_name`` to its own names, types ``payload``,
    and checks ``engine_schema_version`` against its declarations
    (:mod:`powderline.compat`). ``metadata`` is never interpreted: it is checked
    for JSON-serializability and size, and echoed into results (A21, A64).
    """

    schema_name: str = Field(pattern=SCHEMA_NAME_PATTERN,
                             description='Engine and workflow, e.g. "gsasii.rietveld"')
    core_schema_version: str = Field(description="Core schema version the recipe is written against")
    engine_schema_version: str = Field(description="Engine schema version the recipe is written against")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Free-form, JSON-serializable; echoed into results")
    payload: Any = Field(description="Engine-specific; typed by the engine schema")

    @field_validator("core_schema_version")
    @classmethod
    def _core_version(cls, v: str) -> str:
        check_core_schema_version(v)
        return v

    @field_validator("engine_schema_version")
    @classmethod
    def _engine_version(cls, v: str) -> str:
        parse_version("engine_schema_version", v)
        return v

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, v):
        return _check_metadata(v)
