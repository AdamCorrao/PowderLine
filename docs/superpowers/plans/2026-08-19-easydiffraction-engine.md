# EasyDiffraction Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `engine="easydiffraction"` to `powderline.run()` — a third refinement backend that translates unmodified `GSASII_Rietveld` recipes into easydiffraction fits and returns the locked standardized result dict.

**Architecture:** One new self-contained subpackage `src/powderline/easydiff/` (conversions → policy/builder → engine adapter), plus a single lazy-import branch in the existing dispatcher `src/powderline/engine.py`. Zero changes to `schema.py`, `kicker.py`, `gsas_*`, or `topas/`.

**Tech Stack:** Python, pydantic recipes (existing), easydiffraction 0.20.1 (Project / StructureFactory / ExperimentFactory, lmfit minimizer, cryspy calculator), pandas result tables, pixi optional feature.

**Spec:** `docs/superpowers/specs/2026-08-19-easydiffraction-engine-design.md` (read it first — it carries the verified API mapping and the unsupported-parameter policy).

## Global Constraints

- Work in `/nsls2/users/dolds/dev/dumb_new_refinement_code/PowderLine`, branch `feat/easydiffraction-engine`. Commit after every task.
- NEVER modify: `src/powderline/kicker.py`, `src/powderline/schema.py`, `src/powderline/gsas_client.py`, `src/powderline/gsas_server.py`, anything under `src/powderline/topas/`, or any existing test file (exception: none in this plan — even `test_api.py` stays untouched; easydiff shape checks live in new files).
- Test interpreter: this sandbox cannot run GSAS-II or `pixi install`. Run ALL tests with the prepared venv:
  `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest <file> -q`
  (venv has easydiffraction 0.20.1, pydantic, pytest, pandas, numpy).
- Regression guard after every task: `tests/test_schema.py` and `tests/test_topas_writer.py` must still pass with the same command (they don't need GSAS-II).
- easydiffraction datablock names must be lowercase (`'LaB6'` is rejected) — always go through `datablock_slug()`.
- easydiffraction prints progress tables; that's fine in tests (do not assert on stdout).
- Every numeric GSAS-II↔easydiffraction conversion lives in `conversions.py` only.
- All new code must not import `easydiffraction` at module level — only inside functions (`check_unsupported` and `validate_only` must work without the library; report `ImportError` with the pixi remedy otherwise).

**Verified reference numbers** (from the spike run of `examples/example_LaB6/input.json`, λ=0.1665 Å, fit_range [1,15], 3767 pts): easydiffraction converges with reduced χ²≈2.5, Rwp≈0.198 (fraction → 19.8%), cell a = 4.15747 ± 0.00005 Å (NIST 4.15682), refln list has 86 rows. GSAS-II reference Rwp (with background peaks + Z + SH/L, which we drop): 6.53%.

---

### Task 1: `conversions.py` — units, slugs, data prep

**Files:**
- Create: `src/powderline/easydiff/__init__.py`
- Create: `src/powderline/easydiff/conversions.py`
- Test: `tests/test_easydiff_conversions.py`

**Interfaces:**
- Consumes: nothing (pure functions; numpy only).
- Produces (used by Tasks 3–4):
  - `GAUSS_CDEG2_TO_DEG2: float`, `CDEG_TO_DEG: float`
  - `gauss_broadening_to_ed(v: float) -> float` and inverse `gauss_broadening_to_gsas(v)`
  - `lorentz_broadening_to_ed(v: float) -> float` and inverse `lorentz_broadening_to_gsas(v)`
  - `zero_to_ed(v: float) -> float` and inverse `zero_to_gsas(v)`
  - `rwp_fraction_to_percent(v: float | None) -> float | None`
  - `datablock_slug(name: str) -> str`
  - `crop_and_sigma(tth, itth, weights, fit_range) -> tuple[np.ndarray, np.ndarray]` returning `(mask, sigma_of_masked_points)`
  - `q_and_d(tth_deg, wavelength) -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Create the subpackage init**

`src/powderline/easydiff/__init__.py`:
```python
"""GSAS-II-free easydiffraction engine for ``powderline.run(engine="easydiffraction")``.

Translates unmodified GSASII_Rietveld recipes into easydiffraction
(https://github.com/easyscience/diffraction-lib) refinements. Requires the
optional ``easydiff`` pixi environment (Python >=3.12 + easydiffraction).
"""
```

- [ ] **Step 2: Write the failing tests**

`tests/test_easydiff_conversions.py`:
```python
"""Unit-convention conversions for the easydiffraction engine (GSAS-II-free)."""
import math

import numpy as np
import pytest

from powderline.easydiff.conversions import (
    crop_and_sigma,
    datablock_slug,
    gauss_broadening_to_ed,
    gauss_broadening_to_gsas,
    lorentz_broadening_to_ed,
    lorentz_broadening_to_gsas,
    q_and_d,
    rwp_fraction_to_percent,
    zero_to_ed,
    zero_to_gsas,
)


def test_gauss_uvw_centidegsq_sigma_to_degsq_fwhm():
    # GSAS-II U (sigma^2, centideg^2) -> easydiffraction Caglioti U (FWHM^2, deg^2)
    # LaB6 example value, cross-checked in the API spike:
    assert gauss_broadening_to_ed(18.71740850558368) == pytest.approx(0.010379135146427253)
    assert gauss_broadening_to_ed(0.0) == 0.0


def test_gauss_roundtrip():
    assert gauss_broadening_to_gsas(gauss_broadening_to_ed(1.147)) == pytest.approx(1.147)


def test_lorentz_and_zero_are_centideg_to_deg():
    assert lorentz_broadening_to_ed(0.28143034323339766) == pytest.approx(0.0028143034323339766)
    assert lorentz_broadening_to_gsas(0.01) == pytest.approx(1.0)
    assert zero_to_ed(2.0) == pytest.approx(0.02)
    assert zero_to_gsas(zero_to_ed(-3.7)) == pytest.approx(-3.7)


def test_rwp_fraction_to_percent():
    assert rwp_fraction_to_percent(0.19769747424440315) == pytest.approx(19.769747424440315)
    assert rwp_fraction_to_percent(None) is None


def test_datablock_slug_lowercases_and_sanitizes():
    # easydiffraction rejects uppercase datablock names ("Use 'lab6' instead")
    assert datablock_slug("LaB6") == "lab6"
    assert datablock_slug("Li4MgWO6 SG12!") == "li4mgwo6_sg12_"
    assert datablock_slug("") == "phase"


def test_crop_and_sigma_masks_range_and_bad_weights():
    tth = np.array([0.5, 1.0, 2.0, 14.0, 15.5])
    itth = np.ones(5)
    w = np.array([4.0, 4.0, 0.0, 25.0, 4.0])
    mask, sigma = crop_and_sigma(tth, itth, w, [1.0, 15.0])
    assert mask.tolist() == [False, True, False, True, False]  # range + w<=0 dropped
    assert sigma == pytest.approx([0.5, 0.2])  # 1/sqrt(w)


def test_crop_and_sigma_none_range_keeps_all_positive_weight_points():
    tth = np.array([1.0, 2.0])
    mask, sigma = crop_and_sigma(tth, np.ones(2), np.array([1.0, 1.0]), None)
    assert mask.all() and sigma == pytest.approx([1.0, 1.0])


def test_q_and_d_against_gsas_reference_row():
    # First row of examples/example_LaB6/output/fit_profile.txt:
    # two_theta=0.64726557, q=0.42630772, d=14.73861496 at lam=0.1665
    q, d = q_and_d(np.array([0.64726557]), 0.1665)
    assert q[0] == pytest.approx(0.42630772, abs=1e-6)
    assert d[0] == pytest.approx(14.73861496, abs=1e-4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest tests/test_easydiff_conversions.py -q`
Expected: FAIL — `ModuleNotFoundError: powderline.easydiff.conversions`

- [ ] **Step 4: Implement `conversions.py`**

```python
"""GSAS-II <-> easydiffraction unit/convention conversions.

Every numeric convention decision for the easydiffraction engine lives here:
- GSAS-II CW Gaussian U/V/W are variances (sigma^2) in centidegrees^2;
  easydiffraction ``broad_gauss_u/v/w`` are Caglioti FWHM^2 terms in deg^2:
  FWHM^2 = 8 ln2 * sigma^2 and 1 centideg^2 = 1e-4 deg^2.
- GSAS-II Lorentzian X/Y and Zero are centidegrees; easydiffraction uses degrees.
- easydiffraction reports Rwp/Rexp as fractions; the run() contract uses percent.
- easydiffraction datablock names must be lowercase [a-z0-9_].
"""

from __future__ import annotations

import math
import re

import numpy as np

GAUSS_CDEG2_TO_DEG2 = 8.0 * math.log(2.0) * 1e-4
CDEG_TO_DEG = 0.01


def gauss_broadening_to_ed(v: float) -> float:
    return v * GAUSS_CDEG2_TO_DEG2


def gauss_broadening_to_gsas(v: float) -> float:
    return v / GAUSS_CDEG2_TO_DEG2


def lorentz_broadening_to_ed(v: float) -> float:
    return v * CDEG_TO_DEG


def lorentz_broadening_to_gsas(v: float) -> float:
    return v / CDEG_TO_DEG


def zero_to_ed(v: float) -> float:
    return v * CDEG_TO_DEG


def zero_to_gsas(v: float) -> float:
    return v / CDEG_TO_DEG


def rwp_fraction_to_percent(v):
    return None if v is None else 100.0 * float(v)


def datablock_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]", "_", str(name).lower())
    return slug or "phase"


def crop_and_sigma(tth, itth, weights, fit_range):
    """Mask points to fit_range with positive weights; sigma = 1/sqrt(w)."""
    tth = np.asarray(tth, dtype=float)
    w = np.asarray(weights, dtype=float)
    lo, hi = -np.inf, np.inf
    if fit_range:
        if fit_range[0] is not None:
            lo = float(fit_range[0])
        if len(fit_range) > 1 and fit_range[1] is not None:
            hi = float(fit_range[1])
    mask = (tth >= lo) & (tth <= hi) & (w > 0)
    sigma = 1.0 / np.sqrt(w[mask])
    return mask, sigma


def q_and_d(tth_deg, wavelength: float):
    theta = np.radians(np.asarray(tth_deg, dtype=float) / 2.0)
    s = np.sin(theta)
    q = 4.0 * math.pi * s / wavelength
    with np.errstate(divide="ignore"):
        d = np.where(s > 0, wavelength / (2.0 * s), np.inf)
    return q, d
```

- [ ] **Step 5: Run tests to verify they pass**

Same command as Step 3. Expected: all PASS. Also run the regression guard:
`... -m pytest tests/test_schema.py tests/test_topas_writer.py -q` — all pass.

- [ ] **Step 6: Commit**

```bash
git add src/powderline/easydiff/__init__.py src/powderline/easydiff/conversions.py tests/test_easydiff_conversions.py
git commit -m "feat(easydiff): unit/convention conversions for the easydiffraction engine"
```

---

### Task 2: `errors.py` + unsupported-feature policy

**Files:**
- Create: `src/powderline/easydiff/errors.py`
- Create: `src/powderline/easydiff/policy.py`
- Test: `tests/test_easydiff_policy.py`

**Interfaces:**
- Consumes: recipe dicts (already schema-shaped; treat defensively with `.get`).
- Produces (used by Tasks 3–4):
  - `errors.EasyDiffractionTranslationError(Exception)`
  - `policy.check_unsupported(recipe: dict) -> list[str]` — returns warnings;
    raises `EasyDiffractionTranslationError` on refined-unsupported features.
  - `policy.param_flag(spec) -> bool` and `policy.param_value(spec, default=None)`
    for `[value, refine_flag, min, max]` 4-tuples.
- MUST NOT import easydiffraction (validate_only path depends on that).

**Policy (from the spec's honesty rule):**

Raise `EasyDiffractionTranslationError` when the recipe asks to REFINE something unmappable:
1. `schema_name != "GSASII_Rietveld"` (message names the schema and says only GSASII_Rietveld recipes are translatable).
2. Iparm1 contains `Lam1` or `Lam2` (two-wavelength lab data).
3. `instrument.parameterization.broadening.Z` refine flag true (no Gaussian Z term).
4. `instrument.parameterization.polarization` refine flag true (non-refinable in easydiffraction).
5. `instrument.parameterization.corrections.axial_divergence` refine flag true.
6. Any `background.single_peaks` positions/intensities refine flag true (no background-peaks concept).
7. Any atom-level flag true in any phase's `parameterization.atoms.*` for keys `x`, `y`, `z`, `occupancy`, `Uiso` (v1 scope).

Warn (returned strings, never raised) when a FIXED unmappable value is dropped:
- `Z` fixed and nonzero → `"instrument Z=... ignored (no Gaussian Z term in easydiffraction)"`
- `SH/L` nonzero → `"axial divergence SH/L=... not modeled (pseudo-Voigt profile); low-angle peak shapes will differ from GSAS-II"`
- `background.single_peaks` present with all flags false → `"background peaks ignored"`
- any non-null value inside a phase's `parameterization.peak_broadening` → `"phase peak_broadening (size/strain) not mapped"`
- always: `"refinement_cycles not used; lmfit runs to convergence"`

- [ ] **Step 1: Write `errors.py`**

```python
"""Errors for the easydiffraction engine."""


class EasyDiffractionTranslationError(Exception):
    """The recipe uses a feature the easydiffraction engine cannot represent."""
```

- [ ] **Step 2: Write the failing tests**

`tests/test_easydiff_policy.py` — build a minimal Rietveld recipe fixture in-file and mutate per test (do NOT load the LaB6 example here; keep unit tests self-contained):
```python
"""Unsupported-feature policy: refined-unmappable raises, fixed-unmappable warns."""
import copy

import pytest

from powderline.easydiff.errors import EasyDiffractionTranslationError
from powderline.easydiff.policy import check_unsupported, param_flag, param_value


def base_recipe():
    return {
        "schema_name": "GSASII_Rietveld",
        "schema_version": "0.25.4",
        "payload": {
            "xrd_data": {"tth": [1, 2], "Itth": [1, 1], "Itth_weights": [1, 1]},
            "fit_range": [1, 15],
            "instrument": {
                "description": "test",
                "initialization": [
                    {"Lam": [0.1665, 0.1665, False], "Zero": [0.0, 0.0, False],
                     "U": [18.7, 18.7, False], "V": [0.6, 0.6, False], "W": [1.1, 1.1, False],
                     "X": [0.28, 0.28, False], "Y": [0.001, 0.001, False], "Z": [0.0, 0.0, False],
                     "SH/L": [0.0, 0.0, False], "Polariz.": [0.99, 0.99, False]},
                    {},
                ],
                "parameterization": {
                    "broadening": {k: [None, False, None, None] for k in "UVWXYZ"},
                    "corrections": {"axial_divergence": None, "zero_shift": None},
                    "polarization": [0.99, False, None, None],
                },
            },
            "phases": {
                "LaB6": {
                    "structure": {"phase_name": "LaB6", "space_group": "P m -3 m",
                                  "unit_cell": {"a": 4.15682, "b": 4.15682, "c": 4.15682,
                                                "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
                                  "atoms": {"La": {"element": "La", "x": 0.0, "y": 0.0, "z": 0.0,
                                                   "occupancy": 1.0, "Uiso": 0.0086, "ADP": "Uiso"}}},
                    "parameterization": {
                        "atoms": {"La": {k: [None, False, None, None]
                                         for k in ("x", "y", "z", "occupancy", "Uiso")}},
                        "scale": [1, True, None, None],
                        "unit_cell": {k: [None, False, None, None]
                                      for k in ("a", "b", "c", "alpha", "beta", "gamma")},
                    },
                }
            },
            "background": {"chebyshev": {"coefficients": [10, 5], "num_coefficients": 2,
                                         "refine_flag": True}},
            "refinement_controls": {"refinement_cycles": 5},
        },
    }


def test_param_flag_and_value():
    assert param_flag([1.0, True, None, None]) is True
    assert param_flag([1.0, False, None, None]) is False
    assert param_flag(None) is False
    assert param_value([2.5, True, None, None]) == 2.5
    assert param_value([None, True, None, None], default=7) == 7
    assert param_value(None, default=7) == 7


def test_clean_recipe_warns_only_about_cycles():
    warnings = check_unsupported(base_recipe())
    assert len(warnings) == 1 and "refinement_cycles" in warnings[0]


def test_spf_schema_rejected():
    r = base_recipe()
    r["schema_name"] = "GSASII_SPF"
    with pytest.raises(EasyDiffractionTranslationError, match="GSASII_Rietveld"):
        check_unsupported(r)


def test_two_wavelength_rejected():
    r = base_recipe()
    r["payload"]["instrument"]["initialization"][0]["Lam1"] = [1.54, 1.54, False]
    with pytest.raises(EasyDiffractionTranslationError, match="Lam1"):
        check_unsupported(r)


@pytest.mark.parametrize("path,match", [
    (("instrument", "parameterization", "broadening", "Z"), "Z"),
    (("instrument", "parameterization", "polarization"), "polarization"),
    (("instrument", "parameterization", "corrections", "axial_divergence"), "axial"),
])
def test_refined_unmappable_instrument_raises(path, match):
    r = base_recipe()
    node = r["payload"]
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = [None, True, None, None]
    with pytest.raises(EasyDiffractionTranslationError, match=match):
        check_unsupported(r)


def test_refined_background_peaks_raise():
    r = base_recipe()
    r["payload"]["background"]["single_peaks"] = {
        "positions": [[1.5, True, None, None]], "intensities": [[1, False, None, None]]}
    with pytest.raises(EasyDiffractionTranslationError, match="background"):
        check_unsupported(r)


def test_fixed_background_peaks_warn():
    r = base_recipe()
    r["payload"]["background"]["single_peaks"] = {
        "positions": [[1.5, False, None, None]], "intensities": [[1, False, None, None]]}
    warnings = check_unsupported(r)
    assert any("background peaks" in w for w in warnings)


def test_atom_flag_raises():
    r = base_recipe()
    r["payload"]["phases"]["LaB6"]["parameterization"]["atoms"]["La"]["Uiso"] = [None, True, None, None]
    with pytest.raises(EasyDiffractionTranslationError, match="atom"):
        check_unsupported(r)


def test_fixed_nonzero_Z_and_shl_warn():
    r = base_recipe()
    i1 = r["payload"]["instrument"]["initialization"][0]
    i1["Z"] = [0.3, 0.3, False]
    i1["SH/L"] = [0.0005, 0.0005, False]
    warnings = check_unsupported(r)
    assert any("Z=" in w for w in warnings)
    assert any("SH/L" in w for w in warnings)


def test_peak_broadening_model_string_alone_does_not_warn():
    # The real LaB6 example carries {"model": "isotropic"} with null magnitudes;
    # a bare model string must not trigger the not-mapped warning.
    r = base_recipe()
    r["payload"]["phases"]["LaB6"]["parameterization"]["peak_broadening"] = {
        "size_broadening": {"model": "isotropic", "isotropic_size": None, "LG_eta": None},
        "strain_broadening": {"model": "isotropic", "isotropic_strain": None, "LG_eta": None},
    }
    assert not any("peak_broadening" in w for w in check_unsupported(r))


def test_peak_broadening_with_value_warns():
    r = base_recipe()
    r["payload"]["phases"]["LaB6"]["parameterization"]["peak_broadening"] = {
        "size_broadening": {"model": "isotropic", "isotropic_size": [50.0, False, None, None]},
    }
    assert any("peak_broadening" in w for w in check_unsupported(r))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest tests/test_easydiff_policy.py -q`
Expected: FAIL — `ModuleNotFoundError: powderline.easydiff.policy`

- [ ] **Step 4: Implement `policy.py`**

```python
"""Unsupported-feature policy for the easydiffraction engine.

Honesty rule (see the design spec): a FIXED unmappable value is dropped with a
recorded warning; a value the user asked to REFINE but that cannot be
represented raises EasyDiffractionTranslationError. This module must stay
importable without easydiffraction installed (validate_only relies on it).
"""

from __future__ import annotations

from .errors import EasyDiffractionTranslationError


def param_flag(spec) -> bool:
    """Refine flag of a [value, refine_flag, min, max] 4-tuple (or None)."""
    try:
        return bool(spec[1])
    except (TypeError, IndexError):
        return False


def param_value(spec, default=None):
    """Value of a 4-tuple, falling back to default when null/absent."""
    try:
        return default if spec[0] is None else spec[0]
    except (TypeError, IndexError):
        return default


def _iparm_current(iparm1: dict, key: str, default=0.0):
    entry = iparm1.get(key)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[1]
    return default


def check_unsupported(recipe: dict) -> list[str]:
    warnings: list[str] = []
    schema_name = recipe.get("schema_name")
    if schema_name != "GSASII_Rietveld":
        raise EasyDiffractionTranslationError(
            f"schema_name {schema_name!r} is not supported by the easydiffraction "
            "engine; only GSASII_Rietveld recipes are translatable (v1)"
        )
    payload = recipe.get("payload", {}) or {}
    inst = payload.get("instrument", {}) or {}
    init = inst.get("initialization") or [{}, {}]
    iparm1 = init[0] if init else {}
    for key in ("Lam1", "Lam2"):
        if key in iparm1:
            raise EasyDiffractionTranslationError(
                f"two-wavelength instrument parameter {key!r} found; the "
                "easydiffraction engine supports single-wavelength (Lam) data only"
            )

    param = inst.get("parameterization") or {}
    broadening = param.get("broadening") or {}
    if param_flag(broadening.get("Z")):
        raise EasyDiffractionTranslationError(
            "instrument broadening term Z is flagged for refinement but has no "
            "easydiffraction equivalent (no Gaussian Z term)"
        )
    if param_flag(param.get("polarization")):
        raise EasyDiffractionTranslationError(
            "polarization is flagged for refinement but is non-refinable in easydiffraction"
        )
    corrections = param.get("corrections") or {}
    if param_flag(corrections.get("axial_divergence")):
        raise EasyDiffractionTranslationError(
            "axial_divergence is flagged for refinement; the easydiffraction engine "
            "uses a pseudo-Voigt profile without an axial-divergence model"
        )

    z_val = _iparm_current(iparm1, "Z", 0.0)
    if z_val:
        warnings.append(
            f"instrument Z={z_val} ignored (no Gaussian Z term in easydiffraction)"
        )
    shl = _iparm_current(iparm1, "SH/L", 0.0)
    if shl:
        warnings.append(
            f"axial divergence SH/L={shl} not modeled (pseudo-Voigt profile); "
            "low-angle peak shapes will differ from GSAS-II"
        )

    background = payload.get("background") or {}
    bkg_peaks = background.get("single_peaks")
    if bkg_peaks:
        specs = list(bkg_peaks.get("positions") or []) + list(bkg_peaks.get("intensities") or [])
        if any(param_flag(s) for s in specs):
            raise EasyDiffractionTranslationError(
                "background single_peaks are flagged for refinement; easydiffraction "
                "has no background-peaks concept"
            )
        warnings.append("background peaks ignored (no background-peaks concept)")

    for phase_name, phase in (payload.get("phases") or {}).items():
        pz = phase.get("parameterization") or {}
        for atom_name, atom in (pz.get("atoms") or {}).items():
            for key in ("x", "y", "z", "occupancy", "Uiso"):
                if param_flag(atom.get(key)):
                    raise EasyDiffractionTranslationError(
                        f"atom-level refinement ({phase_name}/{atom_name}/{key}) is "
                        "not supported by the easydiffraction engine in v1"
                    )
        pb = pz.get("peak_broadening") or {}

        def _has_content(v):
            # Only actual values count: 4-tuples with a value, or bare numbers.
            # Strings like {"model": "isotropic"} with null magnitudes do not.
            if isinstance(v, (list, tuple)):
                return bool(v) and v[0] is not None
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        flat = [v for group in pb.values() if isinstance(group, dict)
                for v in group.values()]
        flat += [v for v in pb.values() if not isinstance(v, dict)]
        if any(_has_content(v) for v in flat):
            warnings.append(
                f"phase {phase_name!r} peak_broadening (size/strain) not mapped"
            )

    warnings.append("refinement_cycles not used; lmfit runs to convergence")
    return warnings
```

- [ ] **Step 5: Run tests to verify they pass**

Same command as Step 3 — all PASS. Regression guard: `tests/test_schema.py tests/test_topas_writer.py -q` still pass.

- [ ] **Step 6: Commit**

```bash
git add src/powderline/easydiff/errors.py src/powderline/easydiff/policy.py tests/test_easydiff_policy.py
git commit -m "feat(easydiff): translation errors + unsupported-feature policy"
```

---

### Task 3: `builder.py` — recipe → easydiffraction Project + manifest

**Files:**
- Create: `src/powderline/easydiff/builder.py`
- Test: `tests/test_easydiff_builder.py`

**Interfaces:**
- Consumes: Task 1 conversions, Task 2 policy (`check_unsupported`, `param_flag`, `param_value`).
- Produces (used by Task 4):

```python
@dataclass
class ManifestEntry:
    parameter: object            # live easydiffraction Parameter
    parameter_name: str          # GSAS-II-style, e.g. ":0:U", ":0:Back;0", "0:0:Scale", "0::a"
    descriptive_name: str        # e.g. "instrument_broadening_U"
    phase_name: str              # recipe phase name or ""
    phase_idx: object            # int or ""
    atom_name: str; atom_idx: object   # "" in v1
    category: str                # "background" | "instrument_broadening" | "scale" | "unit_cell" | "corrections"
    scale_to_recipe: float       # multiply .value/.uncertainty to get GSAS-II-convention numbers

@dataclass
class BuildResult:
    project: object              # easydiffraction Project (fit-ready)
    experiment: object
    phase_slugs: dict            # recipe phase name -> lowercase datablock slug
    manifest: list               # ManifestEntry for every freed parameter
    warnings: list
    tth: np.ndarray; itth: np.ndarray; weights: np.ndarray   # FULL uncropped arrays
    mask: np.ndarray             # fit_range/weights mask over the full arrays
    wavelength: float

def build_project(recipe: dict, workdir) -> BuildResult   # imports easydiffraction INSIDE
```

**Builder logic (all API calls verified in the spike — see spec):**
1. `warnings = check_unsupported(recipe)`.
2. Data: full arrays from `payload.xrd_data`; `mask, sigma = crop_and_sigma(...)` with `payload.fit_range`; write `workdir/easydiff_data.xye` via `np.savetxt(fmt="%.6f")` with columns tth/Itth/sigma (masked).
3. Per phase: `StructureFactory.from_scratch(name=datablock_slug(...))` (uniquify slugs with a `_2` suffix on collision); set `space_group.name_h_m`, `cell.length_a/b/c`, `cell.angle_alpha/beta/gamma`; per atom `atom_sites.create(id=label, type_symbol=element, fract_x/y/z, occupancy)` then `site = model.atom_sites[label]; site.adp_type = "Uiso"; site.adp_iso = Uiso` (recipe convention is Uiso, Å²; easydiffraction default is Biso, so adp_type must be set explicitly).
4. Experiment: `ExperimentFactory.from_data_path(name="powderline", data_path=..., sample_form="powder", beam_mode="constant wavelength", radiation_probe="xray", scattering_type="bragg")` (probe string is `'xray'`, NOT `'x-ray'`).
5. Instrument from Iparm1 current values (index [1]): `setup_wavelength = Lam`; `calib_twotheta_offset = zero_to_ed(Zero)`; `broad_gauss_u/v/w = gauss_broadening_to_ed(U/V/W)`; `broad_lorentz_x/y = lorentz_broadening_to_ed(X/Y)`; `setup_polarization_coefficient = Polariz.` inside try/except (warn on failure). 4-tuple value overrides (parameterization value[0] non-null) take precedence over Iparm, converted the same way.
6. `linked_structures.create(structure_id=slug, scale=param_value(scale_spec, default=1.0))` per phase.
7. Background: `experiment.background.type = "chebyshev"`; one `experiment.background.create(id=str(k), order=k, coef=c)` per recipe coefficient.
8. Free parameters per flags, appending a ManifestEntry each time:
   - broadening U/V/W (`scale_to_recipe = 1/GAUSS_CDEG2_TO_DEG2`), X/Y (`1/CDEG_TO_DEG`), each `":0:{K}"` / `"instrument_broadening_{K}"` / category `"instrument_broadening"`
   - `corrections.zero_shift` flag → `calib_twotheta_offset.free`, `":0:Zero"` / `"zero_shift"` / `"corrections"` / `1/CDEG_TO_DEG`
   - scale per phase i → `f"{i}:0:Scale"` / `f"phase_{i}_scale_factor"` / `"scale"` / `1.0`
   - chebyshev `refine_flag` → every `experiment.background[str(k)].coef.free = True`, `f":0:Back;{k}"` / `f"background_coefficient_{k}"` / `"background"` / `1.0`
   - unit_cell axis flags (`a`→`cell.length_a`, …, `gamma`→`cell.angle_gamma`) → `f"{i}::{axis}"` / `f"phase_{i}_cell_{axis}"` / `"unit_cell"` / `1.0`. Setting `.free` on a symmetry-constrained axis is a no-op warning in the library — only append the ManifestEntry if `parameter.free` is actually True afterwards.
   - bounds: for every freed parameter whose 4-tuple has non-null min/max, set `parameter.fit_min`/`fit_max` (converted with the same to-ed factor).
9. Project: `Project()`, `project.structures.add(model)`, `project.experiments.add(expt)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_easydiff_builder.py` (importorskip guard; reuse the policy fixture pattern):
```python
"""Recipe -> easydiffraction Project translation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

ed = pytest.importorskip("easydiffraction")

from powderline.easydiff.builder import BuildResult, ManifestEntry, build_project
from powderline.easydiff.conversions import GAUSS_CDEG2_TO_DEG2

from test_easydiff_policy import base_recipe  # same tests/ dir; pytest adds it to sys.path


def rich_recipe():
    r = base_recipe()
    tth = np.linspace(1.0, 15.0, 200)
    r["payload"]["xrd_data"] = {
        "tth": tth.tolist(),
        "Itth": (100 + 10 * np.exp(-((tth - 5) ** 2))).tolist(),
        "Itth_weights": (np.ones_like(tth) / 100.0).tolist(),
    }
    pz = r["payload"]["phases"]["LaB6"]["parameterization"]
    pz["unit_cell"]["a"] = [None, True, None, None]
    bz = r["payload"]["instrument"]["parameterization"]["broadening"]
    bz["U"] = [None, True, 0.0, 100.0]
    return r


def test_build_project_returns_fit_ready_result(tmp_path):
    br = build_project(rich_recipe(), tmp_path)
    assert isinstance(br, BuildResult)
    assert br.phase_slugs == {"LaB6": "lab6"}
    assert (tmp_path / "easydiff_data.xye").exists()
    assert br.wavelength == pytest.approx(0.1665)
    assert br.mask.sum() == 200  # all points inside [1, 15] with positive weights
    # structure landed with Uiso convention and cubic symmetry
    model = br.project.structures["lab6"]
    assert model.cell.length_a.value == pytest.approx(4.15682)


def test_manifest_covers_freed_parameters(tmp_path):
    br = build_project(rich_recipe(), tmp_path)
    names = {m.parameter_name for m in br.manifest}
    # scale + cheb 0/1 + U + cell a  (V..Y flags are False in the fixture)
    assert names == {"0:0:Scale", ":0:Back;0", ":0:Back;1", ":0:U", "0::a"}
    u = next(m for m in br.manifest if m.parameter_name == ":0:U")
    assert u.category == "instrument_broadening"
    assert u.scale_to_recipe == pytest.approx(1.0 / GAUSS_CDEG2_TO_DEG2)
    assert u.parameter.free is True
    # bounds converted into easydiffraction units
    assert u.parameter.fit_min == pytest.approx(0.0)
    assert u.parameter.fit_max == pytest.approx(100.0 * GAUSS_CDEG2_TO_DEG2)
    cell = next(m for m in br.manifest if m.parameter_name == "0::a")
    assert cell.phase_name == "LaB6" and cell.category == "unit_cell"


def test_initial_values_converted(tmp_path):
    br = build_project(rich_recipe(), tmp_path)
    expt = br.experiment
    assert expt.peak.broad_gauss_u.value == pytest.approx(18.7 * GAUSS_CDEG2_TO_DEG2)
    assert expt.peak.broad_lorentz_x.value == pytest.approx(0.0028, abs=1e-4)
    assert expt.background["0"].coef.value == pytest.approx(10.0)


def test_symmetry_constrained_axis_not_in_manifest(tmp_path):
    r = rich_recipe()
    r["payload"]["phases"]["LaB6"]["parameterization"]["unit_cell"]["b"] = [None, True, None, None]
    br = build_project(r, tmp_path)
    names = {m.parameter_name for m in br.manifest}
    assert "0::b" not in names  # cubic: b is symmetry-tied to a; library forces free=False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest tests/test_easydiff_builder.py -q`
Expected: FAIL — `ModuleNotFoundError: powderline.easydiff.builder`

- [ ] **Step 3: Implement `builder.py`**

Follow the numbered builder logic above exactly; module docstring explains the translation and points to the spec. Import `easydiffraction` inside `build_project` only. On `ImportError`, re-raise as:
```python
raise ImportError(
    "easydiffraction is not installed. Install PowderLine's optional engine "
    "environment: `pixi install -e easydiff` and run via `pixi run -e easydiff ...`"
) from exc
```
Key skeleton (fill in the loops per the logic block; keep functions small — `_build_structure`, `_build_experiment`, `_free_parameters` helpers are encouraged):
```python
@dataclass
class ManifestEntry:
    parameter: object
    parameter_name: str
    descriptive_name: str
    phase_name: str = ""
    phase_idx: object = ""
    atom_name: str = ""
    atom_idx: object = ""
    category: str = ""
    scale_to_recipe: float = 1.0


@dataclass
class BuildResult:
    project: object
    experiment: object
    phase_slugs: dict
    manifest: list
    warnings: list
    tth: np.ndarray
    itth: np.ndarray
    weights: np.ndarray
    mask: np.ndarray
    wavelength: float
```

- [ ] **Step 4: Run tests to verify they pass**

Same command. Expected: all PASS (easydiffraction prints progress noise; ignore). Regression guard passes.

- [ ] **Step 5: Commit**

```bash
git add src/powderline/easydiff/builder.py tests/test_easydiff_builder.py
git commit -m "feat(easydiff): recipe->Project builder with free-parameter manifest"
```

---

### Task 4: `easydiff/engine.py` — adapter, reports, result dict

**Files:**
- Create: `src/powderline/easydiff/engine.py`
- Test: `tests/test_easydiff_engine.py`

**Interfaces:**
- Consumes: `build_project`/`BuildResult` (Task 3), `check_unsupported` (Task 2), conversions (Task 1).
- Produces (used by Task 5):
  - `run_easydiffraction_recipe(recipe, output_dir, *, verbose=False, validate_only=False) -> dict`
- Result dict keys (locked; mirror `topas/engine.py:_result_dict`): `success, run_id, rwp, r_exp, gof, elapsed_time, method, output_files, fit_profile, unit_cell_data, peak_list_data, refined_parameters, spf_peaks, spf_convergence_diagnostics, error, traceback`. `method` is `"easydiffraction"` or `"easydiffraction_simulation"`. `rwp`/`r_exp` are **percent** (`rwp_fraction_to_percent`); `gof` is reduced χ².
- `refined_parameters` DataFrame columns (exact order): `parameter_name, descriptive_name, phase_name, phase_idx, atom_name, atom_idx, value, esd, category` — values/esds multiplied by `scale_to_recipe` (esd 0.0 when uncertainty is None).
- Report files written to `output_dir`: `refined_parameters.csv`, `fit_profile.txt` (tab-separated, `%.8f`, header `two_theta y_obs y_weights y_calc y_diff y_bkg q_values d_spacings`, **full original grid** — outside the fit mask `y_calc/y_diff/y_bkg = 0`, matching GSAS-II), `{recipe_phase_name}_unit_cell_report.csv` (`parameter,value,esd` rows `cell_a..cell_gamma`; esd from `parameter.uncertainty` when freed else 0), and `{recipe_phase_name}_peak_list_report.csv` (`h,k,l,d_spacing,2theta,F_calc_squared`, filtered by `refln.structure_id == slug`) when `experiment.refln` is not None.
- `validate_only=True`: call `check_unsupported` only (no easydiffraction import), return the slim TOPAS-parity dict: `{"success": True, "rwp": None, "elapsed_time": 0.0, "method": "validate_only", "schema_name": ..., "schema_version": ..., "phases": <count>, "refinement_cycles": ..., "simulation_mode": cycles == 1}` — no `run_id`.
- Behavior: translation errors propagate (TOPAS parity); post-build runtime failures return `success=False` dicts with `error` + `traceback`, never crash. Empty-manifest recipes are simulations: skip `fit()`, call `project.analysis.calculate()`, `rwp/r_exp/gof = None`.
- Fit sequence: `project.analysis.minimizer.type = "lmfit (leastsq)"` then `project.analysis.fit()`. Stats: `_num = lambda x: getattr(x, "value", x)` applied to `project.analysis.fit_result.prof_wr_factor` (Rwp fraction), `.prof_wr_expected` (Rexp fraction), and `project.analysis.fit_results.reduced_chi_square` (gof). Arrays: `experiment.data.fit_data_arrays()` → dict with keys `x, meas, meas_su, calc, diff, bkg` (masked grid; scatter back onto the full grid via `BuildResult.mask`). `y_weights` column = original recipe weights. `q_values/d_spacings` from `q_and_d(tth_full, wavelength)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_easydiff_engine.py`:
```python
"""Result-contract tests for the easydiffraction engine adapter."""
import numpy as np
import pandas as pd
import pytest

ed = pytest.importorskip("easydiffraction")

from powderline.easydiff.engine import run_easydiffraction_recipe

from test_easydiff_builder import rich_recipe
from test_easydiff_policy import base_recipe

RESULT_KEYS = {
    "success", "run_id", "rwp", "r_exp", "gof", "elapsed_time", "method",
    "output_files", "fit_profile", "unit_cell_data", "peak_list_data",
    "refined_parameters", "spf_peaks", "spf_convergence_diagnostics",
    "error", "traceback",
}
REFINED_COLUMNS = ["parameter_name", "descriptive_name", "phase_name", "phase_idx",
                   "atom_name", "atom_idx", "value", "esd", "category"]
PROFILE_COLUMNS = ["two_theta", "y_obs", "y_weights", "y_calc", "y_diff", "y_bkg",
                   "q_values", "d_spacings"]


def test_validate_only_slim_shape(tmp_path):
    result = run_easydiffraction_recipe(base_recipe(), tmp_path, validate_only=True)
    assert result["success"] is True
    assert result["method"] == "validate_only"
    assert result["rwp"] is None
    assert "run_id" not in result
    assert result["phases"] == 1
    assert result["simulation_mode"] is False  # cycles = 5


def test_synthetic_fit_result_contract(tmp_path):
    result = run_easydiffraction_recipe(rich_recipe(), tmp_path)
    assert set(result) == RESULT_KEYS
    assert result["success"] is True, result["error"]
    assert result["method"] == "easydiffraction"
    assert result["rwp"] is not None and 0 < result["rwp"] < 100  # percent
    assert list(result["refined_parameters"].columns) == REFINED_COLUMNS
    assert len(result["refined_parameters"]) == 5  # scale, 2 cheb, U, cell a
    assert list(result["fit_profile"].columns) == PROFILE_COLUMNS
    assert len(result["fit_profile"]) == 200  # full grid
    assert "LaB6" in result["unit_cell_data"]
    assert isinstance(result["spf_peaks"], pd.DataFrame) and result["spf_peaks"].empty
    assert result["error"] is None
    out_names = {p.rsplit("/", 1)[-1] for p in result["output_files"]}
    assert {"refined_parameters.csv", "fit_profile.txt",
            "LaB6_unit_cell_report.csv"} <= out_names


def test_simulation_mode_when_nothing_refined(tmp_path):
    r = rich_recipe()
    pz = r["payload"]["phases"]["LaB6"]["parameterization"]
    pz["scale"] = [1, False, None, None]
    pz["unit_cell"]["a"] = [None, False, None, None]
    r["payload"]["instrument"]["parameterization"]["broadening"]["U"] = [None, False, None, None]
    r["payload"]["background"]["chebyshev"]["refine_flag"] = False
    result = run_easydiffraction_recipe(r, tmp_path)
    assert result["success"] is True
    assert result["method"] == "easydiffraction_simulation"
    assert result["rwp"] is None
    assert result["refined_parameters"].empty
    assert list(result["refined_parameters"].columns) == REFINED_COLUMNS
    assert (result["fit_profile"]["y_calc"] != 0).any()  # pattern actually calculated


def test_unique_run_ids(tmp_path):
    import uuid
    r1 = run_easydiffraction_recipe(rich_recipe(), tmp_path / "a")
    r2 = run_easydiffraction_recipe(rich_recipe(), tmp_path / "b")
    assert uuid.UUID(r1["run_id"]) != uuid.UUID(r2["run_id"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest tests/test_easydiff_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: powderline.easydiff.engine`

- [ ] **Step 3: Implement `engine.py`**

Structure it like `topas/engine.py` (read that file first — it is the pattern):
`_as_dict`, `_empty_df`, `_num`, `_validate_only_result` (copy the TOPAS one, it is engine-agnostic), `_write_reports(build_result, arrays, output_dir) -> None`, `_result_dict(...)`, and the public `run_easydiffraction_recipe`. Scatter masked arrays back to the full grid:
```python
full = np.zeros(len(br.tth))
full[br.mask] = arrays["calc"]
```
Wrap everything after `build_project` in `try/except Exception` returning `success=False` with `error=str(exc)` and `traceback=traceback.format_exc()`. `output_files` = `sorted(str(f) for f in Path(output_dir).glob("*") if f.is_file())`. Update `src/powderline/easydiff/__init__.py` to re-export `run_easydiffraction_recipe` and `EasyDiffractionTranslationError`.

- [ ] **Step 4: Run tests to verify they pass**

Same command (the two fit tests take ~30 s each — be patient). Regression guard passes.

- [ ] **Step 5: Commit**

```bash
git add src/powderline/easydiff/engine.py src/powderline/easydiff/__init__.py tests/test_easydiff_engine.py
git commit -m "feat(easydiff): engine adapter with locked result dict + standardized reports"
```

---

### Task 5: dispatcher branch in `powderline/engine.py`

**Files:**
- Modify: `src/powderline/engine.py` (12-line change: `_ENGINES` tuple, docstrings, one branch)
- Test: `tests/test_easydiff_dispatch.py`

**Interfaces:**
- Consumes: `run_easydiffraction_recipe` (Task 4).
- Produces: `powderline.run(recipe, output_dir, engine="easydiffraction", ...)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_easydiff_dispatch.py`:
```python
"""Dispatcher wiring for engine="easydiffraction" (no easydiffraction needed)."""
import pytest

from powderline.engine import _ENGINES, run


def test_easydiffraction_in_engines_tuple():
    assert "easydiffraction" in _ENGINES


def test_unknown_engine_message_lists_easydiffraction(tmp_path):
    with pytest.raises(ValueError, match="easydiffraction"):
        run({}, tmp_path, engine="bogus")


def test_dispatch_reaches_easydiff_engine(tmp_path, monkeypatch):
    import powderline.easydiff.engine as ede
    calls = {}

    def fake(recipe, output_dir, *, verbose=False, validate_only=False):
        calls["recipe"] = recipe
        calls["validate_only"] = validate_only
        return {"success": True}

    monkeypatch.setattr(ede, "run_easydiffraction_recipe", fake)
    result = run({"schema_name": "GSASII_Rietveld"}, tmp_path,
                 engine="easydiffraction", validate_only=True)
    assert result == {"success": True}
    assert calls["validate_only"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest tests/test_easydiff_dispatch.py -q`
Expected: FAIL — `"easydiffraction" not in _ENGINES` / ValueError message mismatch.

- [ ] **Step 3: Implement the dispatcher branch**

In `src/powderline/engine.py`: change `_ENGINES = ("gsasii", "topas")` to `("gsasii", "topas", "easydiffraction")`; extend the module and `run()` docstrings ("``engine="easydiffraction"`` routes to the GSAS-II-free easydiffraction adapter; requires the optional ``easydiff`` pixi environment; ignores ``execution_mode`` and ``topas_*``"); add before the final `raise`:
```python
    if engine == "easydiffraction":
        # GSAS-II-free; module-level import so tests can monkeypatch the function
        from powderline.easydiff import engine as _easydiff_engine

        return _easydiff_engine.run_easydiffraction_recipe(
            recipe,
            output_dir,
            verbose=verbose,
            validate_only=validate_only,
        )
```
(Import the module, not the bare function — `test_dispatch_reaches_easydiff_engine` monkeypatches `powderline.easydiff.engine.run_easydiffraction_recipe` and needs the dispatcher to resolve it at call time.)

- [ ] **Step 4: Run tests + full non-GSAS-II suite**

`... -m pytest tests/test_easydiff_dispatch.py tests/test_schema.py tests/test_topas_writer.py tests/test_topas_engine.py -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/powderline/engine.py tests/test_easydiff_dispatch.py
git commit -m "feat: register easydiffraction engine in the run() dispatcher"
```

---

### Task 6: LaB6 easydiff example + integration & policy tests

**Files:**
- Create: `examples/example_LaB6_easydiff/input.json`
- Create: `examples/example_LaB6_easydiff/DESCRIPTION.md`
- Test: `tests/test_easydiff_integration.py`

**Interfaces:**
- Consumes: `powderline.run` dispatch (Task 5), stock `examples/example_LaB6/input.json`.
- Produces: the canonical easydiffraction example recipe used in README/docs.

- [ ] **Step 1: Generate the example recipe**

Derive from the stock recipe (run once, commit the JSON):
```python
import json
src = json.load(open("examples/example_LaB6/input.json"))
p = src["payload"]
p["background"].pop("single_peaks", None)          # no background-peaks concept
p["instrument"]["parameterization"]["broadening"]["Z"] = [None, False, None, None]
p["phases"]["LaB6"]["parameterization"]["unit_cell"]["a"] = [None, True, None, None]
json.dump(src, open("examples/example_LaB6_easydiff/input.json", "w"), indent=1)
```
Validate it round-trips the schema:
`MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src <venv-python> -c "import json; from powderline.schema import RecipeModel; RecipeModel(**json.load(open('examples/example_LaB6_easydiff/input.json')))"`

`DESCRIPTION.md` (short): what changed vs the stock example (background peaks removed, Z fixed, cell `a` refined) and the run command `powderline.run(recipe, out, engine="easydiffraction")`.

- [ ] **Step 2: Write the failing integration tests**

`tests/test_easydiff_integration.py`:
```python
"""End-to-end: LaB6 example recipe refined by the easydiffraction engine.

Spike-verified expectations: Rwp ~19.8% (no SH/L asymmetry / background peaks,
so much higher than GSAS-II's 6.53%), cell a -> 4.1575 +/- 0.0001 vs NIST 4.15682.
"""
import json
from pathlib import Path

import pytest

ed = pytest.importorskip("easydiffraction")

from powderline.easydiff.errors import EasyDiffractionTranslationError
from powderline.engine import run

EXAMPLE = Path("examples/example_LaB6_easydiff/input.json")
STOCK = Path("examples/example_LaB6/input.json")


@pytest.fixture(scope="module")
def lab6_result(tmp_path_factory):
    recipe = json.loads(EXAMPLE.read_text())
    out = tmp_path_factory.mktemp("easydiff_lab6")
    return run(recipe, out, engine="easydiffraction"), out


def test_refinement_succeeds_with_sane_rwp(lab6_result):
    result, _ = lab6_result
    assert result["success"] is True, result["error"]
    assert result["method"] == "easydiffraction"
    assert 5.0 < result["rwp"] < 30.0          # percent; spike: ~19.8
    assert result["gof"] is not None and result["gof"] < 10.0


def test_lattice_parameter_close_to_nist(lab6_result):
    result, _ = lab6_result
    df = result["refined_parameters"]
    row = df[df["parameter_name"] == "0::a"].iloc[0]
    assert row["value"] == pytest.approx(4.15682, abs=0.002)   # NIST SRM 660
    assert row["esd"] > 0


def test_report_files_written(lab6_result):
    result, out = lab6_result
    for name in ("refined_parameters.csv", "fit_profile.txt",
                 "LaB6_unit_cell_report.csv"):
        assert (out / name).exists(), name
    assert len(result["fit_profile"]) == 4096  # full original grid, padded


def test_stock_lab6_recipe_rejected_loudly(tmp_path):
    # The stock recipe refines Z and background peaks -> per the honesty rule
    # this engine must refuse, not silently drop them.
    recipe = json.loads(STOCK.read_text())
    with pytest.raises(EasyDiffractionTranslationError):
        run(recipe, tmp_path, engine="easydiffraction")
```

- [ ] **Step 3: Run tests to verify current state**

Run: `MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src /nsls2/users/dolds/dev/dumb_new_refinement_code/spike/venv/bin/python -m pytest tests/test_easydiff_integration.py -q`
Expected: with Tasks 1–5 done and the example JSON generated, this should PASS (~60 s: real 3767-point fit). If Rwp or the lattice tolerance fails, debug the conversions (superpowers:systematic-debugging) — do NOT widen tolerances beyond the values above without recording why in the test docstring.

- [ ] **Step 4: Commit**

```bash
git add examples/example_LaB6_easydiff tests/test_easydiff_integration.py
git commit -m "feat(easydiff): LaB6 example recipe + end-to-end integration and policy tests"
```

---

### Task 7: pixi feature + docs

**Files:**
- Modify: `pixi.toml` (append two sections; do NOT touch existing tables)
- Modify: `README.md` (engines section)
- Modify: `docs/integration-guide.md` (short subsection)

- [ ] **Step 1: Append to `pixi.toml`**

```toml
[feature.easydiff.dependencies]
python = ">=3.12"                 # easydiffraction floor; default env stays >=3.10
"diffpy.pdffit2" = "*"            # from conda-forge: its PyPI sdist needs GSL to build

[feature.easydiff.pypi-dependencies]
easydiffraction = ">=0.20.1,<0.21"

[environments]
easydiff = ["easydiff"]
```

- [ ] **Step 2: Attempt lock verification**

Run: `pixi lock 2>&1 | tail -5` from the repo root.
- If it succeeds: verify with `git diff pixi.lock | head -50` that the **default** environment's packages are unchanged (only an `easydiff` env added). If the solve fails because GSAS-II won't coexist with Python ≥3.12, change the environment line to `easydiff = { features = ["easydiff"], no-default-feature = true }` and re-lock.
- If the network/DNS fails in this sandbox (known issue): leave `pixi.lock` untouched, and add this line to the commit message body: `NOTE: pixi.lock not regenerated (sandbox has no conda network); run 'pixi lock' before merging.`

- [ ] **Step 3: README + integration guide**

README: add an "Engines" subsection (after the TOPAS one if present, else near usage) with:
```python
import powderline
result = powderline.run(recipe, "output", engine="easydiffraction")
```
State: translates unmodified `GSASII_Rietveld` recipes; requires `pixi install -e easydiff` (Python ≥3.12); v1 supports cell/scale/background/U-V-W-X-Y/zero refinement, rejects atom-level and Kα₁/Kα₂ recipes loudly; Rwp values are not directly comparable to GSAS-II (no axial-divergence asymmetry). Link `docs/engine-survey.md` and the spec. Same two paragraphs, condensed, in `docs/integration-guide.md`.

- [ ] **Step 4: Full test sweep + commit**

Run every runnable suite once more:
`MPLCONFIGDIR=$TMPDIR PYTHONPATH=$PWD/src <venv-python> -m pytest tests/test_easydiff_conversions.py tests/test_easydiff_policy.py tests/test_easydiff_builder.py tests/test_easydiff_engine.py tests/test_easydiff_dispatch.py tests/test_easydiff_integration.py tests/test_schema.py tests/test_topas_writer.py tests/test_topas_engine.py -q`
All pass →
```bash
git add pixi.toml README.md docs/integration-guide.md
git commit -m "feat(easydiff): optional pixi feature/environment + docs"
```
