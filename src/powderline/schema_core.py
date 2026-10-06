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
- plain (non-refinable) values get field-specific validation.

This module must never import an engine (enforced by an import-block test).
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import gemmi
import numpy as np
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_serializer,
    model_validator,
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


# --- parameter models (A20, A32, A71) ---------------------------------------


def _finite(name: str, v: Optional[float]) -> Optional[float]:
    if v is not None and not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


class RefinableParameter(BaseModel):
    """A refinable value with no bounds concept: JSON ``[value, refine_flag]``.

    Used where the engine has no native bounds for the quantity. A 4-element
    list here is an error, never silently truncated (A20). ``value`` is
    required; see :class:`StructureRefinableParameter` for the quantities whose
    starting value may come from the phase structure.
    """

    model_config = _STRICT

    value: float
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
        nullable = cls.model_fields["value"].annotation is not float
        return _array_schema(cls.__name__, [_NUMBER_OR_NULL if nullable else _NUMBER, _BOOLEAN],
                             "[value, refine_flag]" + (_NULL_VALUE_NOTE if nullable else ""))


class BoundedRefinableParameter(BaseModel):
    """A refinable value with optional bounds: JSON ``[value, refine_flag, min, max]``.

    ``min``/``max`` may each be ``null`` (unbounded on that side). When given,
    ``min <= max`` and ``min <= value <= max``. Used only where the engine
    honors the bounds natively (A20, A33). ``value`` is required; see
    :class:`StructureBoundedRefinableParameter`.
    """

    model_config = _STRICT

    value: float
    refine_flag: StrictBool
    min: Optional[float] = None
    max: Optional[float] = None

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
        if self.value is not None:
            check_within_bounds(self.value, self, "value")
        return self

    @model_serializer
    def _to_list(self) -> list:
        return [self.value, self.refine_flag, self.min, self.max]

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler) -> dict:
        nullable = cls.model_fields["value"].annotation is not float
        return _array_schema(cls.__name__,
                             [_NUMBER_OR_NULL if nullable else _NUMBER, _BOOLEAN,
                              _NUMBER_OR_NULL, _NUMBER_OR_NULL],
                             "[value, refine_flag, min, max]; min/max may be null"
                             + (_NULL_VALUE_NOTE if nullable else ""))


_NULL_VALUE_NOTE = "; value null = start from the phase structure's value"


class StructureRefinableParameter(RefinableParameter):
    """:class:`RefinableParameter` for a quantity that mirrors a phase-structure value.

    For cell lengths/angles and atom x/y/z, occupancy, Uiso/Uaniso only (A71):
    ``value`` may be ``null``, meaning "start from the structure's value"
    (today's behavior: the GSAS-II setters skip a null value).
    """

    value: Optional[float]


class StructureBoundedRefinableParameter(BoundedRefinableParameter):
    """:class:`BoundedRefinableParameter` for a quantity that mirrors a phase-structure value.

    ``value`` may be ``null`` (start from the structure's value, A71); the
    bounds are then checked against that value by the phase-level validator
    (:func:`check_within_bounds`).
    """

    value: Optional[float]


def check_within_bounds(value: float, param: BoundedRefinableParameter, field: str) -> None:
    """Raise ``ValueError`` unless ``param.min <= value <= param.max`` (open ends pass).

    Also used at the phase level for a null-valued
    :class:`StructureBoundedRefinableParameter`, with the structure's value.
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
    if element.atomic_number > 0:
        raise ValueError(f"element {symbol!r}: write the element symbol as {element.name!r}")
    raise ValueError(f"element {symbol!r} is not a known element symbol")


# --- diffraction data -------------------------------------------------------


class XRDData(BaseModel):
    """Measured pattern: 2theta, intensity, weights (ported from schema 0.26 ``XRDDataModel``).

    Every engine consumes ``tth`` in degrees 2theta and ``Itth_weights`` as
    1/sigma^2, as-is.
    """

    model_config = _STRICT

    tth: list[float] = Field(description="Two-theta values", json_schema_extra=_unit(UNIT_DEG_2THETA))
    Itth: list[float] = Field(description="Intensity values", json_schema_extra=_unit(UNIT_ARBITRARY))
    Itth_weights: list[float] = Field(description="Intensity weights (1/sigma^2)",
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

    min: Optional[float] = None
    max: Optional[float] = None

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

    num_coefficients: int = Field(gt=0, description="Number of Chebyshev coefficients")
    coefficients: list[float] = Field(description="Coefficient values")
    refine_flag: StrictBool = Field(description="Whether to refine background")

    @field_validator('coefficients')
    @classmethod
    def validate_coefficients_length(cls, v, info):
        """Ensure number of coefficients matches num_coefficients."""
        num_coef = info.data.get('num_coefficients')
        if num_coef is not None and len(v) != num_coef:
            raise ValueError(f"Number of coefficients ({len(v)}) must match num_coefficients ({num_coef})")
        return v


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


class CoreRecipe(BaseModel):
    """The core-owned top level of every recipe; engine schemas subclass it.

    An engine schema narrows ``schema_name`` to its own names, types ``payload``,
    and checks ``engine_schema_version`` against its declarations
    (:mod:`powderline.compat`). ``metadata`` is never interpreted: it is checked
    for JSON-serializability and size, and echoed into results (A21, A64).
    """

    model_config = _STRICT

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
