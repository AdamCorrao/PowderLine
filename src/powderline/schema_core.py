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
import math
import re
from typing import Annotated, Any, Generic, Literal, Optional, TypeVar

import gemmi
import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    Strict,
    StrictBool,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema, JsonSchemaMode
from pydantic_core import InitErrorDetails, PydanticCustomError

from powderline.exceptions import StructuredWarning, SymmetryError
from powderline.symmetry import analyze_site, canonical_space_group, check_cell, check_uij

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
    """Raise ``ValueError`` unless ``param.min <= value <= param.max`` (open ends pass).

    Also used at the phase level, on a value canonicalized after the
    parameter's own check (review N4).
    """
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
        return v

    @field_validator("alpha", "beta", "gamma")
    @classmethod
    def _angle(cls, v, info):
        if not 0 < v.value < 180:
            raise ValueError(f"cell angle {info.field_name} must be between 0 and 180 degrees, got {v.value}")
        return v


_UANISO_KEYS = ("U11", "U22", "U33", "U12", "U13", "U23")


class Atom(CoreModel, Generic[P]):
    """One atom of a phase: coordinates, occupancy and ADPs are refinable parameters.

    Coordinates are fractional JSON numbers. A special position is declared by
    stating it to at least 6 decimals; the validated phase holds the canonical
    (exact) values and reports every adjustment (A73). ``occupancy`` is required
    and 0 to 1 inclusive (A76, A96). ``Multiplicity`` is optional and, when
    stated, must match the derived value (A69 T3). ``ADP`` selects ``Uiso`` or
    ``Uaniso`` (all six of U11..U23); the other is left out. ``Uaniso`` must
    respect the site symmetry: within 1e-6 A^2 it is set to the symmetric values
    and reported, beyond that it is an error (A78). Nothing is ``null`` (A96).
    """

    element: str = Field(description="Bare element symbol, exactly as in the periodic table (e.g. 'Fe')")
    x: P = Field(description="Fractional x coordinate")
    y: P = Field(description="Fractional y coordinate")
    z: P = Field(description="Fractional z coordinate")
    occupancy: P = Field(description="Site occupancy, 0 to 1 inclusive (A76)")
    Multiplicity: Optional[CoreInt] = Field(default=None, json_schema_extra=_absent_not_null_schema(),
                                            description="Site multiplicity; optional, cross-checked when stated")
    ADP: Literal["Uiso", "Uaniso"] = Field(description="Displacement-parameter type")
    Uiso: Optional[P] = Field(default=None, json_schema_extra=_absent_not_null_schema(UNIT_ANGSTROM2),
                              description="Isotropic ADP; given when ADP is 'Uiso'")
    Uaniso: Optional[dict[Literal["U11", "U22", "U33", "U12", "U13", "U23"], P]] = Field(
        default=None, json_schema_extra=_absent_not_null_schema(UNIT_ANGSTROM2),
        description="All six of U11, U22, U33, U12, U13, U23; given when ADP is 'Uaniso'")

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
        if self.ADP == "Uiso":
            if self.Uiso is None:
                raise ValueError("ADP 'Uiso' requires a Uiso value")
            if self.Uaniso is not None:
                raise ValueError("ADP 'Uiso' must not also give Uaniso")
        else:
            if self.Uaniso is None or set(self.Uaniso) != set(_UANISO_KEYS):
                raise ValueError(f"ADP 'Uaniso' requires all of {', '.join(_UANISO_KEYS)}")
            if self.Uiso is not None:
                raise ValueError("ADP 'Uaniso' must not also give Uiso")
        return self


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
    atom's site (multiplicity, DOF) from it (:func:`powderline.symmetry.analyze_site`),
    rejects ambiguous positions and stated multiplicities that don't match, and
    replaces stated coordinates with the canonical ones (keeping each refine
    flag and bounds), so every engine receives the same exact structure. Each
    adjustment is reported by :meth:`warnings` (A73). The unit cell must fit the
    space group exactly (A77), and anisotropic ADPs the site symmetry (A78).

    Every problem is reported at its own location (A70): ``unit_cell``,
    ``atoms.<label>`` (the position), ``atoms.<label>.Multiplicity``,
    ``atoms.<label>.Uaniso``; all of them together, in one ``ValidationError``.
    These checks run only once every field is valid: pydantic does not run
    model-level validators after a field error, so e.g. an unknown element is
    reported first, and the symmetry checks follow once it is fixed.
    Validation never modifies the caller's objects; adjusted atoms are copies.
    """

    space_group: str = Field(description="gemmi's canonical extended Hermann-Mauguin name, e.g. "
                                         "'P m -3 m', 'C 1 2/m 1', 'R -3 m:H', 'F d -3 m:2' (A105)")
    unit_cell: UnitCell[P]
    atoms: dict[str, Atom[P]] = Field(min_length=1, description="Atoms keyed by unique label")

    _sites: dict = PrivateAttr(default_factory=dict)
    _warnings: list = PrivateAttr(default_factory=list)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for klass in cls.__mro__:
            if klass is Phase or not (isinstance(klass, type) and issubclass(klass, Phase)):
                continue
            if klass.__pydantic_generic_metadata__["origin"] is Phase:
                continue  # Phase[SomeParameterType] itself
            redeclared = [n for n in RESERVED_PHASE_FIELDS if n in inspect.get_annotations(klass)]
            if redeclared:
                raise TypeError(
                    f"{klass.__name__} redeclares the core phase field(s) {', '.join(redeclared)}; "
                    "core owns space_group, unit_cell and atoms (A93)"
                )

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

        cell = self.unit_cell
        try:
            check_cell(self.space_group, tuple(getattr(cell, k).value for k in _CELL_FIELDS))
        except SymmetryError as exc:
            problem(("unit_cell",), str(exc), cell.model_dump())
        for label, atom in list(self.atoms.items()):
            try:
                site = analyze_site(self.space_group, (atom.x.value, atom.y.value, atom.z.value))
            except SymmetryError as exc:
                problem(("atoms", label), str(exc), atom.model_dump())
                continue  # the checks below need the site
            update: dict[str, Any] = {}
            if atom.Multiplicity is not None and atom.Multiplicity != site.multiplicity:
                problem(("atoms", label, "Multiplicity"),
                        f"stated Multiplicity {atom.Multiplicity}, derived {site.multiplicity} "
                        f"({self.space_group!r}, site {site.stated})", atom.Multiplicity)
            if atom.Uaniso is not None:
                stated_u = tuple(atom.Uaniso[k].value for k in _UANISO_KEYS)
                try:
                    symmetric_u, u_adjusted = check_uij(self.space_group, site.canonical, stated_u)
                except SymmetryError as exc:
                    problem(("atoms", label, "Uaniso"), str(exc), atom.model_dump()["Uaniso"])
                else:
                    if u_adjusted:
                        update["Uaniso"] = {k: _with_value(atom.Uaniso[k], u)
                                            for k, u in zip(_UANISO_KEYS, symmetric_u)}
                        self._warnings.append(StructuredWarning(
                            code="adp_symmetry_adjusted",
                            message=(f"atom {label!r}: Uaniso {dict(zip(_UANISO_KEYS, stated_u))} "
                                     f"adjusted to the site-symmetric values "
                                     f"{dict(zip(_UANISO_KEYS, symmetric_u))}"),
                            field_path=f"atoms.{label}.Uaniso",
                        ))
            self._sites[label] = site
            if site.adjusted:
                update.update({k: _with_value(getattr(atom, k), v) for k, v in zip("xyz", site.canonical)})
                self._warnings.append(StructuredWarning(
                    code="special_position_adjusted",
                    message=(f"atom {label!r}: coordinates {site.stated} are on a special position "
                             f"(multiplicity {site.multiplicity}); using the exact values "
                             f"{site.canonical}"),
                    field_path=f"atoms.{label}",
                ))
            if update:
                self.atoms[label] = atom.model_copy(update=update)
        if problems:
            raise ValidationError.from_exception_data(type(self).__name__, problems)
        return self

    def site(self, label: str):
        """The :class:`~powderline.symmetry.SiteAnalysis` of atom ``label``."""
        return self._sites[label]

    def warnings(self) -> list:
        """Structured warnings raised while validating this phase (paths relative to it)."""
        return list(self._warnings)


_CELL_FIELDS = ("a", "b", "c", "alpha", "beta", "gamma")


def _with_value(param, value: float):
    """A copy of ``param`` with a new value; refine flag and bounds kept."""
    return param.model_copy(update={"value": value})


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
