"""Pure unit-conversion math and TOPAS equation-string builders.

Every function here is pure (numbers in, numbers or TOPAS equation strings out)
and unit-tested independently against hand-computed values. No GSAS-II, no I/O.

Unit reminders (findings.md §A, plan §4):

* GSAS-II instrument U, V, W  -> Gaussian **variance** in centideg^2
* GSAS-II instrument X, Y, Z  -> Lorentzian **FWHM** in centideg
* size                        -> micrometres (um)
* microstrain                 -> unitless fraction of Delta d / d, times 1e6
* background-peak sigma       -> centideg (std-dev, Gaussian)
* background-peak gamma       -> centideg (Lorentzian FWHM)
* Uiso                        -> Angstrom^2

Formula constants (verified in findings.md §A/§C; do not re-derive):

* ``SQRT8LN2_OVER_100 = sqrt(8*ln2)/100`` -> sigma(centideg) to Gaussian FWHM(deg)
* ``CENTIDEG_TO_DEG   = 1/100``           -> a centideg FWHM to a degree FWHM
* ``SIZE_COEF   = 1.8/100  = 0.018``      -> GSAS-II size term, centideg to deg
* ``STRAIN_COEF = 0.018/100 = 1.8e-4``    -> GSAS-II strain term, centideg to deg
* ``EIGHT_PI_SQ = 8*pi**2``               -> Uiso(Angstrom^2) to Beq

The TOPAS equations mirror GSAS-II's own broadening math exactly (findings §C.1,
§C.2): they carry the recipe's *own* quantities as named ``prm`` parameters and
perform the unit conversion inside ``gauss_fwhm`` / ``lor_fwhm`` convolutions on
a near-delta pseudo-Voigt base. ``Th`` is theta, ``Lam`` is the wavelength in
Angstrom, ``Pi`` is pi -- all TOPAS reserved names (TR Table 13.1, p. 284).
"""

from __future__ import annotations

import math

# --- verified conversion constants -----------------------------------------

#: sqrt(8*ln2)/100 : GSAS-II sigma (centideg std-dev) -> Gaussian FWHM (deg).
SQRT8LN2_OVER_100 = math.sqrt(8.0 * math.log(2.0)) / 100.0  # 0.02354820045...

#: 1/100 : a FWHM in centidegrees -> a FWHM in degrees.
CENTIDEG_TO_DEG = 0.01

#: 1.8/100 : GSAS-II size term (Sgam, centideg) written in degrees.
SIZE_COEF = 1.8 / 100.0  # 0.018

#: 0.018/100 : GSAS-II strain term (Mgam, centideg) written in degrees.
STRAIN_COEF = SIZE_COEF / 100.0  # 1.8e-4

#: 8*pi**2 : Uiso (Angstrom^2) -> Beq (the TOPAS ``beq`` value).
EIGHT_PI_SQ = 8.0 * math.pi**2  # 78.9568352087...


# --- scalar conversions -----------------------------------------------------


def beq_from_uiso(uiso: float) -> float:
    """Isotropic ADP: ``B = 8*pi**2 * Uiso`` (findings §B.4).

    >>> round(beq_from_uiso(0.00858), 6)
    0.677449
    """
    return EIGHT_PI_SQ * float(uiso)


def bkg_gauss_fwhm_deg(sigma_centideg: float) -> float:
    """Background-peak Gaussian FWHM in degrees from sigma in centidegrees.

    ``FWHM_G(deg) = sqrt(8 ln2) * sigma / 100``.

    >>> round(bkg_gauss_fwhm_deg(10.0), 6)
    0.235482
    """
    return SQRT8LN2_OVER_100 * float(sigma_centideg)


def bkg_lor_fwhm_deg(gamma_centideg: float) -> float:
    """Background-peak Lorentzian FWHM in degrees from gamma (centideg FWHM).

    GSAS-II's gamma is already a FWHM (findings §C.3); only a /100 unit change.
    """
    return CENTIDEG_TO_DEG * float(gamma_centideg)


def strain_lor_coef_deg(microstrain: float, eta: float) -> float:
    """Coefficient on ``tan(theta)`` for the strain Lorentzian FWHM, in degrees.

    ``coef = 1.8e-4 * eta * microstrain / pi``.  Findings §C.1 sanity check:
    microstrain = 1000, eta = 1 gives 1e-3 rad expressed in degrees.

    >>> round(strain_lor_coef_deg(1000.0, 1.0), 10) == round(math.degrees(1e-3), 10)
    True
    """
    return STRAIN_COEF * float(eta) * float(microstrain) / math.pi


def strain_gauss_coef_deg(microstrain: float, eta: float) -> float:
    """Coefficient on ``tan(theta)`` for the strain Gaussian FWHM, in degrees.

    The Gaussian effect carries the ``(1 - eta)`` share of the magnitude.
    """
    return STRAIN_COEF * (1.0 - float(eta)) * float(microstrain) / math.pi


def size_lor_coef_deg(size_um: float, eta: float, wavelength_ang: float) -> float:
    """Coefficient on ``1/cos(theta)`` for the size Lorentzian FWHM, in degrees.

    ``coef = 0.018 * eta * lambda / (pi * size_um)`` (lambda in Angstrom).
    """
    return SIZE_COEF * float(eta) * float(wavelength_ang) / (math.pi * float(size_um))


def size_gauss_coef_deg(size_um: float, eta: float, wavelength_ang: float) -> float:
    """Coefficient on ``1/cos(theta)`` for the size Gaussian FWHM, in degrees."""
    return SIZE_COEF * (1.0 - float(eta)) * float(wavelength_ang) / (math.pi * float(size_um))


# --- deterministic float formatting -----------------------------------------


def fmt(value: float) -> str:
    """Format a float for byte-stable INP output (``%.10g``, no exponent noise).

    Integral floats render without a trailing ``.0`` (matches the recipe's own
    ``a 4.15682`` / ``al 90`` style). ``-0.0`` normalises to ``0``.

    >>> fmt(90.0)
    '90'
    >>> fmt(4.15682)
    '4.15682'
    >>> fmt(-0.0)
    '0'
    """
    v = float(value)
    if v == 0.0:
        v = 0.0  # collapse -0.0
    return "%.10g" % v


# --- name sanitisation ------------------------------------------------------


def sanitize(name: str) -> str:
    """Sanitise an owner/param name into a legal TOPAS ``prm`` identifier.

    Keeps alphanumerics and underscores, replaces every other character with an
    underscore, and prefixes ``p_`` when the result does not start with a letter
    (TOPAS names must start with a letter; TR §3).

    >>> sanitize("LaB6")
    'LaB6'
    >>> sanitize("Li4MgWO6_SG12")
    'Li4MgWO6_SG12'
    >>> sanitize("2theta pos")
    'p_2theta_pos'
    """
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "p_" + cleaned
    return cleaned


# --- x_calculation_step (D8) ------------------------------------------------


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def round_down_1sf(value: float) -> float:
    """Round a positive number *down* to one significant figure.

    >>> round_down_1sf(0.0018585)
    0.001
    >>> round_down_1sf(0.0)
    0.0
    """
    v = float(value)
    if v <= 0.0:
        return 0.0
    exponent = math.floor(math.log10(v))
    factor = 10.0**exponent
    return math.floor(v / factor) * factor


def x_calculation_step(tth: list[float]) -> float:
    """Mandatory TOPAS ``x_calculation_step`` for unequal-step data (D8).

    Rule: half the median adjacent 2-theta step, rounded down to one significant
    figure. For the demo data (~0.00372 deg step) this yields ``0.001``.
    """
    if len(tth) < 2:
        raise ValueError("need at least two 2-theta points for x_calculation_step")
    diffs = [abs(tth[i + 1] - tth[i]) for i in range(len(tth) - 1)]
    return round_down_1sf(_median(diffs) / 2.0)


# --- TOPAS equation-string builders -----------------------------------------
#
# Each returns the right-hand-side expression (no leading keyword, no trailing
# ';'); the writer composes ``gauss_fwhm = <rhs>;`` etc. Names passed in are
# already-sanitised prm identifiers. Formatting of the leading constants is kept
# as literal decimal text so golden files are byte-stable across machines.


#: Floor applied to convolution FWHM equations so a refining parameter (e.g. a
#: GSAS-II Lorentzian Z that goes negative, or a Gaussian share that hits zero
#: when eta sits at its bound) can never produce a non-positive FWHM, which
#: TOPAS rejects with "negative FWHM encountered". 1e-9 deg is numerically
#: negligible (well below any real peak width). See S5 findings.
FWHM_FLOOR = "1e-9"


def guard_fwhm(expr: str) -> str:
    """Wrap a FWHM equation RHS so it stays strictly positive: ``Max(<expr>, 1e-9)``."""
    return f"Max({expr}, {FWHM_FLOOR})"


def bkg_gauss_fwhm_eq(sigma_name: str) -> str:
    """``FWHM_G(deg) = 0.0235482 * sigma`` for a background peak (plan §4)."""
    return f"0.0235482 {sigma_name}"


def bkg_lor_fwhm_eq(gamma_name: str) -> str:
    """``FWHM_L(deg) = 0.01 * gamma`` for a background peak (plan §4)."""
    return f"0.01 {gamma_name}"


def spf_gauss_fwhm_eq(sigma_sq_name: str) -> str:
    """``FWHM_G(deg) = 0.0235482 * sqrt(sigma^2)`` for an SPF peak.

    Single-peak-fitting stores the Gaussian width as a **variance** (sigma^2),
    unlike the background peaks (which store the std-dev sigma), so the FWHM
    conversion takes a square root first.
    """
    return f"0.0235482 Sqrt({sigma_sq_name})"


# TCH pseudo-Voigt combination coefficients (Thompson-Cox-Hastings 1987), shared
# with GSAS-II (findings §A.5) and used for the SPF derived-width report.
_TCH = (2.69269, 2.42843, 4.47163, 0.07842)


def peak_widths(sigma: float, gamma: float) -> tuple[float, float, float, float, float, float]:
    """Derived widths for a pseudo-Voigt peak, matching GSAS-II's SPF report.

    Args mirror ``kicker.calculate_peak_widths``: ``sigma`` is the Gaussian
    std-dev and ``gamma`` the Lorentzian **HWHM**, in the same (degree) units the
    report uses. Returns ``(fwhm_g, fwhm_l, fwhm_pv, ib_g, ib_l, ib_pv)`` — the
    Gaussian/Lorentzian/pseudo-Voigt FWHMs and integral breadths. Pure math, no
    GSAS-II.
    """
    fwhm_g = 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma  # 2.35482 sigma
    fwhm_l = 2.0 * gamma
    a, b, c, d = _TCH
    fwhm_pv = (
        fwhm_g**5
        + a * fwhm_g**4 * fwhm_l
        + b * fwhm_g**3 * fwhm_l**2
        + c * fwhm_g**2 * fwhm_l**3
        + d * fwhm_g * fwhm_l**4
        + fwhm_l**5
    ) ** 0.2
    ib_g = fwhm_g * math.sqrt(math.pi / (4.0 * math.log(2.0)))  # ~1.0645 fwhm_g
    ib_l = math.pi * gamma
    if fwhm_pv > 0:
        q = fwhm_l / fwhm_pv
        eta = 1.36603 * q - 0.47719 * q**2 + 0.11116 * q**3
        ib_pv = eta * ib_l + (1.0 - eta) * ib_g
    else:
        ib_pv = 0.0
    return fwhm_g, fwhm_l, fwhm_pv, ib_g, ib_l, ib_pv


def instrument_gauss_fwhm_eq(u_name: str, v_name: str, w_name: str) -> str:
    """GSAS-II Gaussian instrument profile as a TOPAS ``gauss_fwhm`` RHS (D4).

    ``0.0235482 Sqrt(Max(U tan^2 + V tan + W, 1e-9))`` -- U,V,W are centideg^2
    variance; 0.0235482 = sqrt(8 ln2)/100 converts variance(centideg^2) to a
    FWHM in degrees. ``Max(.,1e-9)`` guards the sqrt against negative variance.
    """
    return (
        f"0.0235482 Sqrt(Max({u_name} Tan(Th)^2 + {v_name} Tan(Th) + {w_name}, 1e-9))"
    )


def instrument_lor_fwhm_eq(x_name: str, y_name: str, z_name: str) -> str:
    """GSAS-II Lorentzian instrument profile as a TOPAS ``lor_fwhm`` RHS (D4).

    ``0.01 (X/cos + Y tan + Z)`` -- X is the 1/cos(theta) (size-like) term, Y is
    the tan(theta) (strain-like) term, Z a constant; all centideg FWHM (findings
    §A.5). The letters keep their GSAS-II meanings (no TCHZ swap -- D4).
    """
    return f"0.01 ({x_name} / Cos(Th) + {y_name} Tan(Th) + {z_name})"


def size_lor_fwhm_eq(eta_name: str, size_name: str) -> str:
    """Size Lorentzian FWHM RHS: ``0.018 eta Lam / (Pi size Cos(Th))`` (§C.1)."""
    return f"0.018 {eta_name} Lam / (Pi {size_name} Cos(Th))"


def size_gauss_fwhm_eq(eta_name: str, size_name: str) -> str:
    """Size Gaussian FWHM RHS: ``0.018 (1 - eta) Lam / (Pi size Cos(Th))``."""
    return f"0.018 (1 - {eta_name}) Lam / (Pi {size_name} Cos(Th))"


def strain_lor_fwhm_eq(eta_name: str, strain_name: str) -> str:
    """Strain Lorentzian FWHM RHS: ``1.8e-4 eta strain Tan(Th) / Pi`` (§C.1)."""
    return f"1.8e-4 {eta_name} {strain_name} Tan(Th) / Pi"


def strain_gauss_fwhm_eq(eta_name: str, strain_name: str) -> str:
    """Strain Gaussian FWHM RHS: ``1.8e-4 (1 - eta) strain Tan(Th) / Pi``."""
    return f"1.8e-4 (1 - {eta_name}) {strain_name} Tan(Th) / Pi"
