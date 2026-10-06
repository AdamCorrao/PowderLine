"""gemmi-backed crystallographic rules (engine-free core module).

Pure, engine-free capabilities:

* :func:`cell_constraints` -- crystal-system cell equalities and fixed angles,
  so the writer can share one ``prm`` name across equal cell lengths and emit
  symmetry-fixed angles as bare constants.
* :func:`site_dof` -- per-axis site-symmetry degrees of freedom (FREE / FIXED /
  COUPLED) plus the orbit size, so the writer can decide whether a *refined*
  atomic coordinate is legal and warn on suspicious multiplicities.

* :func:`analyze_site` -- the core schema's site validation (KI-03; A35, A69,
  A73): multiplicity + DOF from the space group, the ambiguous-band error, and
  the canonical (exact) special-position coordinates every gateway sends to its
  engine. Policy and evidence: devkit ``tasks/re03-symmetry-tolerance.md``.

Space-group operations come from gemmi; the stabiliser / projector math is our
own (~small, spec in plan §5). ``gemmi`` is an approved runtime dependency (D3).

Errors are :class:`~powderline.exceptions.SymmetryError`; each gateway converts
it to its own error type and applies its engine's own restrictions (e.g. TOPAS
rejects the rhombohedral ``:R`` setting) (A61, A62).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import gemmi
import numpy as np

from powderline.exceptions import SymmetryError


# Fractional-coordinate tolerances (plan §5).
_POS_TOL = 1e-5   # position / orbit equality mod 1 (legacy site_dof/adp_dof)
_MAT_TOL = 1e-6   # rotation-matrix / projector comparisons

#: Core schema special-position rules (A69, A73), as a fractional *image
#: distance* d = max_i |wrap(R x + t - x)_i|: d < SPECIAL_POSITION_TOL means x is
#: fixed by that operation (>= 6 decimals declare a special position);
#: SPECIAL_POSITION_TOL <= d < AMBIGUOUS_BAND is an error (ambiguous intent).
SPECIAL_POSITION_TOL = 5e-6
AMBIGUOUS_BAND = 2e-3
#: Canonical coordinates that differ from the stated ones by more than this are
#: reported as adjusted (floating-point noise is not).
ADJUSTMENT_REPORT_TOL = 1e-12
_MAX_DENOMINATOR = 48  # fixed special-position coordinates are multiples of 1/24

_AXIS_LETTERS = ("x", "y", "z")
_LENGTHS = ("a", "b", "c")
_ANGLES = ("alpha", "beta", "gamma")


# --- data returned to the writer -------------------------------------------


@dataclass(frozen=True)
class CellRules:
    """Crystal-system cell constraints for one space group.

    Attributes:
        crystal_system: gemmi's crystal-system string (``"cubic"`` etc.).
        length_groups: groups of cell-length names that must share one refined
            ``prm``. Every length appears in exactly one group; a singleton
            group is an independent length. E.g. cubic ``(("a", "b", "c"),)``,
            monoclinic ``(("a",), ("b",), ("c",))``.
        fixed_angles: angle names fixed by symmetry -- emitted as bare numeric
            constants from the recipe structure (a ``refine=true`` flag on any
            of these is a hard error).
        free_angles: angle names that are genuine refinable DOF (monoclinic
            unique-axis angle, or all three for triclinic).
        unique_axis: ``"a"``/``"b"``/``"c"`` for monoclinic, else ``None``.
    """

    crystal_system: str
    length_groups: tuple[tuple[str, ...], ...]
    fixed_angles: tuple[str, ...]
    free_angles: tuple[str, ...]
    unique_axis: str | None = None


@dataclass(frozen=True)
class SiteDof:
    """Per-axis site degrees of freedom and orbit size for one atomic site.

    Attributes:
        axes: classification for ``(x, y, z)`` -- each ``"FREE"``, ``"FIXED"``
            or ``"COUPLED"``.
        coupled_axes: axis letters (``"x"``/``"y"``/``"z"``) that are coupled
            (non-axis-aligned allowed-displacement subspace); empty when none.
        orbit_size: number of distinct symmetry-equivalent positions (the true
            multiplicity for this position and space group).
        stabilizer_order: number of symmetry operations that fix the position.
    """

    axes: tuple[str, str, str]
    coupled_axes: tuple[str, ...]
    orbit_size: int
    stabilizer_order: int

    def is_free(self, axis: str) -> bool:
        return self.axes[_AXIS_LETTERS.index(axis)] == "FREE"

    def classification(self, axis: str) -> str:
        return self.axes[_AXIS_LETTERS.index(axis)]


# --- space-group resolution -------------------------------------------------


def resolve_space_group(space_group: str, *, explicit_setting: bool = False) -> gemmi.SpaceGroup:
    """Resolve a recipe space-group string to a gemmi ``SpaceGroup``.

    Tries the symbol verbatim, then whitespace-stripped (gemmi accepts both
    ``"P m -3 m"`` and ``"Pm-3m"``). Unresolvable symbols raise
    :class:`SymmetryError`. Two-origin groups passed without an explicit
    ``:1``/``:2`` selector also error rather than guess an origin (plan §5(6)).
    Both rhombohedral settings (``:H``, ``:R``) are accepted; engines that
    cannot handle ``:R`` reject it themselves (A62). With
    ``explicit_setting=True`` (the core schema rule, A62) a rhombohedral group
    must also name its setting (gemmi would otherwise assume ``:H``).
    """
    raw = str(space_group)
    sg = gemmi.find_spacegroup_by_name(raw)
    if sg is None:
        sg = gemmi.find_spacegroup_by_name(raw.replace(" ", ""))
    if sg is None:
        raise SymmetryError(
            f"unrecognized space-group symbol {space_group!r} (gemmi could not resolve it)"
        )
    if explicit_setting and sg.ext in ("H", "R") and ":" not in raw:
        raise SymmetryError(
            f"space group {space_group!r} is rhombohedral; specify the setting "
            "explicitly: append ':H' (hexagonal axes) or ':R' (rhombohedral axes)"
        )
    if sg.ext in ("1", "2") and ":" not in raw:
        raise SymmetryError(
            f"space group {space_group!r} has two origin choices; specify one "
            "explicitly (e.g. append ':2' for the GSAS-II origin-2 convention)"
        )
    return sg


def _expanded_ops(sg: gemmi.SpaceGroup) -> list[tuple[np.ndarray, np.ndarray]]:
    """All symmetry operations (centering expanded) as (R float 3x3, t float 3)."""
    den = float(gemmi.Op.DEN)
    ops = []
    for op in sg.operations():
        R = np.array(op.rot, dtype=float) / den
        t = np.array(op.tran, dtype=float) / den
        ops.append((R, t))
    return ops


# --- (a) cell rules ---------------------------------------------------------


def _monoclinic_unique_axis(ops: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Unique axis of a monoclinic group from its proper 2-fold direction.

    The single proper 2-fold (det +1, trace -1) fixes the unique axis; its
    invariant direction maps to the dominant cell axis a/b/c.
    """
    for R, _t in ops:
        if abs(np.linalg.det(R) - 1.0) < _MAT_TOL and abs(np.trace(R) + 1.0) < _MAT_TOL:
            evals, evecs = np.linalg.eig(R)
            for i in range(3):
                if abs(evals[i].real - 1.0) < _MAT_TOL and abs(evals[i].imag) < _MAT_TOL:
                    direction = np.abs(evecs[:, i].real)
                    return _LENGTHS[int(np.argmax(direction))]
    raise SymmetryError(
        "could not determine the monoclinic unique axis (no proper 2-fold found)"
    )


def cell_constraints(space_group: str) -> CellRules:
    """Cell-length equality groups and fixed/free angles for ``space_group``.

    See :class:`CellRules`. Unknown symbols raise :class:`SymmetryError` (via
    :func:`resolve_space_group`), as does the rhombohedral ``:R`` setting (its
    tied angles are not expressible as :class:`CellRules`).
    """
    sg = resolve_space_group(space_group)
    system = sg.crystal_system_str()

    singles = tuple((name,) for name in _LENGTHS)
    all_angles_fixed = _ANGLES

    if system == "cubic":
        return CellRules(system, (("a", "b", "c"),), all_angles_fixed, ())
    if system == "tetragonal":
        return CellRules(system, (("a", "b"), ("c",)), all_angles_fixed, ())
    if system in ("hexagonal", "trigonal"):
        if sg.ext == "R":
            # Rhombohedral axes need alpha=beta=gamma tied together, which
            # CellRules cannot express; refuse rather than return wrong rules.
            raise SymmetryError(
                f"cell rules for the rhombohedral ':R' setting are not implemented "
                f"({space_group!r}); use the hexagonal (:H) setting"
            )
        # Hexagonal axes (incl. the :H setting of rhombohedral groups): a=b, 90/90/120.
        return CellRules(system, (("a", "b"), ("c",)), all_angles_fixed, ())
    if system == "orthorhombic":
        return CellRules(system, singles, all_angles_fixed, ())
    if system == "monoclinic":
        unique = _monoclinic_unique_axis(_expanded_ops(sg))
        free_angle = {"a": "alpha", "b": "beta", "c": "gamma"}[unique]
        fixed = tuple(a for a in _ANGLES if a != free_angle)
        return CellRules(system, singles, fixed, (free_angle,), unique_axis=unique)
    if system == "triclinic":
        return CellRules(system, singles, (), _ANGLES)

    raise SymmetryError(
        f"unsupported crystal system {system!r} for space group {space_group!r}"
    )


# --- (b) site DOF -----------------------------------------------------------


def _wrap_symmetric(delta: np.ndarray) -> np.ndarray:
    """Map a fractional difference into (-0.5, 0.5] per component (mod-1 aware)."""
    return (delta + 0.5) % 1.0 - 0.5


def _orbit_size(ops, xyz: np.ndarray) -> int:
    points: list[np.ndarray] = []
    for R, t in ops:
        y = (R @ xyz + t) % 1.0
        if not any(np.all(np.abs(_wrap_symmetric(y - p)) < _POS_TOL) for p in points):
            points.append(y)
    return len(points)


def _classify(projector: np.ndarray) -> list[str]:
    """Per-component DOF from a projector ``P`` onto the allowed displacements.

    FIXED: row j of P is zero (no allowed displacement changes component j).
    FREE: ``P e_j == e_j`` (component j changes on its own). Otherwise COUPLED.
    Rows and columns of P differ when the rotations are not orthogonal in
    fractional coordinates (hexagonal/trigonal axes), so the FIXED test must use
    the row: e.g. 6h ``(x, 2x, 1/4)`` in P6_3/mmc has x COUPLED, not FIXED. The
    classification is checked against GSAS-II's site-symmetry tables for all 230
    space groups (``tests/test_symmetry_characterization.py``).
    """
    n = projector.shape[0]
    out = []
    for j in range(n):
        unit = np.zeros(n)
        unit[j] = 1.0
        if np.linalg.norm(projector[j, :]) < _MAT_TOL:
            out.append("FIXED")
        elif np.linalg.norm(projector[:, j] - unit) < _MAT_TOL:
            out.append("FREE")
        else:
            out.append("COUPLED")
    return out


def _adp_projector(stab_rots) -> np.ndarray:
    """Projector onto site-allowed U tensors (``U -> R U R^T``) in u11..u23 coordinates."""
    projector = np.zeros((6, 6))
    for R in stab_rots:
        for j in range(6):
            basis = np.zeros(6)
            basis[j] = 1.0
            projector[:, j] += _vec_from_sym(R @ _sym_from_vec(basis) @ R.T)
    return projector / len(stab_rots)


def site_dof(space_group: str, xyz) -> SiteDof:
    """Classify the site DOF of fractional position ``xyz`` in ``space_group``.

    Algorithm (plan §5(b)):

    1. Expand all symmetry operations (with centering).
    2. Stabiliser = ops with ``R x + t == x (mod 1)`` within ``_POS_TOL``.
    3. Allowed-displacement subspace = fixed space of the stabiliser rotations,
       given by the Reynolds projector ``P = mean(R_i)``.
    4. Per axis: see :func:`_classify` (FIXED = row of P is zero, FREE =
       ``P e_j == e_j``, else COUPLED).
    """
    x = np.asarray(xyz, dtype=float)
    if x.shape != (3,):
        raise ValueError("xyz must be a 3-vector")
    ops = _expanded_ops(resolve_space_group(space_group))

    stab_rots = [R for (R, t) in ops if np.all(np.abs(_wrap_symmetric(R @ x + t - x)) < _POS_TOL)]
    if not stab_rots:  # pragma: no cover - identity is always in the group
        raise SymmetryError("empty stabilizer (should be impossible)")

    axes = _classify(sum(stab_rots) / len(stab_rots))
    coupled = [_AXIS_LETTERS[j] for j in range(3) if axes[j] == "COUPLED"]

    return SiteDof(
        axes=(axes[0], axes[1], axes[2]),
        coupled_axes=tuple(coupled),
        orbit_size=_orbit_size(ops, x),
        stabilizer_order=len(stab_rots),
    )


# --- (c) anisotropic ADP site symmetry (advisory) ---------------------------

_ADP_LABELS = ("u11", "u22", "u33", "u12", "u13", "u23")


@dataclass(frozen=True)
class AdpDof:
    """Per-component symmetry DOF of the anisotropic ADP tensor at a site.

    ``components`` classifies ``u11,u22,u33,u12,u13,u23`` as ``"FREE"``,
    ``"FIXED"`` (forced to 0 by site symmetry), or ``"COUPLED"`` (locked to
    another component). Advisory: the permissive writer warns when a recipe
    refines a non-FREE component; it does not enforce the constraint.
    """

    components: tuple[str, str, str, str, str, str]
    stabilizer_order: int

    def classification(self, comp: str) -> str:
        return self.components[_ADP_LABELS.index(comp)]

    def free(self) -> tuple[str, ...]:
        return tuple(c for c, cls in zip(_ADP_LABELS, self.components) if cls == "FREE")


def _sym_from_vec(v: np.ndarray) -> np.ndarray:
    return np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])


def _vec_from_sym(u: np.ndarray) -> np.ndarray:
    return np.array([u[0, 0], u[1, 1], u[2, 2], u[0, 1], u[0, 2], u[1, 2]])


def adp_dof(space_group: str, xyz) -> AdpDof:
    """Classify the anisotropic-ADP DOF of a site (which ``u_ij`` are free).

    The symmetric U tensor transforms as ``U -> R U R^T`` under a site-symmetry
    rotation ``R``; the site-allowed U is the fixed space of the stabiliser,
    given by the Reynolds projector on the 6-dim symmetric-tensor space,
    classified by :func:`_classify` (e.g. cubic ``m-3m`` -> ``u11=u22=u33``
    coupled, off-diagonals FIXED).
    """
    x = np.asarray(xyz, dtype=float)
    if x.shape != (3,):
        raise ValueError("xyz must be a 3-vector")
    ops = _expanded_ops(resolve_space_group(space_group))
    stab = [R for (R, t) in ops if np.all(np.abs(_wrap_symmetric(R @ x + t - x)) < _POS_TOL)]
    comps = _classify(_adp_projector(stab))
    return AdpDof(components=tuple(comps), stabilizer_order=len(stab))


# --- (d) core site validation (KI-03; A35, A69, A73) ------------------------


@dataclass(frozen=True)
class SiteAnalysis:
    """One atomic site, analyzed under the core schema rules.

    Attributes:
        stated: the coordinates as given.
        canonical: the coordinates every gateway uses -- equal to ``stated`` for
            a general position; on a special position, symmetry-fixed axes are
            exact (see ``exact``) and coupled axes satisfy their relation (to
            floating-point precision), by the least change from ``stated``.
        exact: per axis, the exact rational value of a symmetry-fixed
            coordinate (``None`` for a free or coupled axis).
        axes: ``"FREE"`` / ``"FIXED"`` / ``"COUPLED"`` per axis.
        adp: the same classification for ``U11, U22, U33, U12, U13, U23``.
        multiplicity: site multiplicity in the conventional cell.
        stabilizer_order: number of operations fixing the site.
    """

    stated: tuple[float, float, float]
    canonical: tuple[float, float, float]
    exact: tuple[Fraction | None, Fraction | None, Fraction | None]
    axes: tuple[str, str, str]
    adp: tuple[str, str, str, str, str, str]
    multiplicity: int
    stabilizer_order: int

    @property
    def adjusted(self) -> bool:
        """True when ``canonical`` differs from ``stated`` beyond float noise."""
        return any(abs(c - s) > ADJUSTMENT_REPORT_TOL for c, s in zip(self.canonical, self.stated))


def _fmt_xyz(xyz) -> str:
    return "(" + ", ".join(f"{v:.6g}" for v in xyz) + ")"


def _fmt_exact(values, exact) -> str:
    return "(" + ", ".join(str(e) if e is not None else f"{v:.6g}" for v, e in zip(values, exact)) + ")"


def _barycenter(x: np.ndarray, ops) -> np.ndarray:
    """Mean of the stabilizer images of ``x``, each unwrapped next to ``x``.

    The barycenter of an orbit under a finite group is fixed by every element,
    so this is the nearest point fixed by all of ``ops``.
    """
    return x + np.mean([_wrap_symmetric(R @ x + t - x) for R, t in ops], axis=0)


def _is_group(ops) -> bool:
    """True when ``ops`` is closed under composition (translations mod 1)."""
    def key(R, t):
        return (tuple(np.rint(R).astype(int).ravel()), tuple(np.round(t % 1.0, 9) % 1.0))

    keys = {key(R, t) for R, t in ops}
    return all(key(R1 @ R2, R1 @ t2 + t1) in keys for R1, t1 in ops for R2, t2 in ops)


def _axes(stab_rots) -> list[str]:
    return _classify(sum(stab_rots) / len(stab_rots))


def _orthogonal_projector(reynolds: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the image of a (possibly oblique) Reynolds projector.

    The Reynolds average ``mean(R)`` projects onto the symmetric subspace, but not
    orthogonally when the rotations are not orthogonal (hexagonal/trigonal axes).
    The orthogonal projector gives the *least change* onto the same subspace, so
    a free component stays as stated and only inconsistent ones move (A73, A78).
    """
    u, sv, _vt = np.linalg.svd(reynolds)
    basis = u[:, sv > 0.5]  # a projector's nonzero singular values are >= 1 (1 if orthogonal)
    return basis @ basis.T


def analyze_site(space_group: str, xyz) -> SiteAnalysis:
    """Validate one site under the core schema rules and return its :class:`SiteAnalysis`.

    For every operation (centering expanded), the image distance ``d`` is
    computed. ``d < SPECIAL_POSITION_TOL`` puts the operation in the stabilizer.
    Any ``SPECIAL_POSITION_TOL <= d < AMBIGUOUS_BAND`` raises
    :class:`SymmetryError`: the site is near a special position without being on
    it (or is stated with too few decimals). The message names the special
    position and its multiplicity.

    ``space_group`` must name its setting explicitly (two-origin and
    rhombohedral groups, A62).
    """
    x = np.asarray(xyz, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise SymmetryError("xyz must be a finite 3-vector")
    ops = _expanded_ops(resolve_space_group(space_group, explicit_setting=True))
    dist = [float(np.max(np.abs(_wrap_symmetric(R @ x + t - x)))) for R, t in ops]

    ambiguous = [d for d in dist if SPECIAL_POSITION_TOL <= d < AMBIGUOUS_BAND]
    if ambiguous:
        near = [op for op, d in zip(ops, dist) if d < AMBIGUOUS_BAND]
        if _is_group(near):
            target = _barycenter(x, near)
            axes = _axes([R for R, _t in near])
            exact = [Fraction(float(v)).limit_denominator(_MAX_DENOMINATOR) if a == "FIXED" else None
                     for v, a in zip(target, axes)]
            where = (f"the special position {_fmt_exact(target, exact)} "
                     f"(multiplicity {len(ops) // len(near)})")
        else:  # near more than one special position: no single one to name
            where = "a special position"
        raise SymmetryError(
            f"position {_fmt_xyz(x)} is {max(ambiguous):.2g} from {where} in "
            f"{space_group!r} without being on it: state the special-position "
            f"coordinates to at least 6 decimals, or move the atom at least "
            f"{AMBIGUOUS_BAND:g} (fractional) off it"
        )

    stab = [op for op, d in zip(ops, dist) if d < SPECIAL_POSITION_TOL]
    if not stab:  # pragma: no cover - identity is always in the group
        raise SymmetryError("empty stabilizer (should be impossible)")
    axes = _axes([R for R, _t in stab])

    # Least change onto the stabilizer's fixed set: anchor at a fixed point (the
    # barycenter), then project the offset orthogonally onto the allowed directions.
    anchor = _barycenter(x, stab)
    canonical = anchor + _orthogonal_projector(sum(R for R, _t in stab) / len(stab)) @ (x - anchor)
    canonical = np.where(np.abs(canonical - x) <= ADJUSTMENT_REPORT_TOL, x, canonical)  # drop float noise
    exact: list[Fraction | None] = []
    for j in range(3):
        if axes[j] != "FIXED":
            exact.append(None)
            continue
        frac = Fraction(float(canonical[j])).limit_denominator(_MAX_DENOMINATOR)
        if abs(float(frac) - canonical[j]) > 1e-9:  # pragma: no cover - guards the 1/24 premise
            raise SymmetryError(
                f"fixed coordinate {canonical[j]!r} of {_fmt_xyz(x)} in {space_group!r} is not a "
                f"fraction with denominator <= {_MAX_DENOMINATOR}"
            )
        exact.append(frac)
        canonical[j] = float(frac)

    # Consistency: the canonical point is fixed exactly by the whole stabilizer,
    # and multiplicity x stabilizer order = group order.
    residual = max(float(np.max(np.abs(_wrap_symmetric(R @ canonical + t - canonical))))
                   for R, t in stab)
    if residual > 1e-9 or len(ops) % len(stab):  # pragma: no cover - internal invariant
        raise SymmetryError(
            f"inconsistent site analysis for {_fmt_xyz(x)} in {space_group!r} "
            f"(residual {residual:.2g}, |G|={len(ops)}, |stabilizer|={len(stab)})"
        )

    return SiteAnalysis(
        stated=(float(x[0]), float(x[1]), float(x[2])),
        canonical=(float(canonical[0]), float(canonical[1]), float(canonical[2])),
        exact=(exact[0], exact[1], exact[2]),
        axes=(axes[0], axes[1], axes[2]),
        adp=tuple(_classify(_adp_projector([R for R, _t in stab]))),
        multiplicity=len(ops) // len(stab),
        stabilizer_order=len(stab),
    )


# --- (e) cell and Uij consistency with symmetry (A76-A78) -------------------

#: Cell parameters must equal their symmetric values up to floating-point noise
#: (relative on lengths, degrees on angles): no adjustment, only an error (A77).
CELL_TOL = 1e-9
#: Uij within this of their site-symmetric values (Angstrom^2) are set to those
#: values and reported; beyond it is an error (A78). Allows for values printed
#: to ~6 decimals breaking relations such as U12 = U22/2.
UIJ_TOL = 1e-6

_CELL_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


def _metric(cell) -> np.ndarray:
    a, b, c, al, be, ga = cell
    ca, cb, cg = (np.cos(np.radians(v)) for v in (al, be, ga))
    return np.array([[a * a, a * b * cg, a * c * cb],
                     [a * b * cg, b * b, b * c * ca],
                     [a * c * cb, b * c * ca, c * c]])


def _cell_from_metric(g: np.ndarray) -> tuple[float, ...]:
    a, b, c = np.sqrt(np.diag(g))
    al = np.degrees(np.arccos(np.clip(g[1, 2] / (b * c), -1.0, 1.0)))
    be = np.degrees(np.arccos(np.clip(g[0, 2] / (a * c), -1.0, 1.0)))
    ga = np.degrees(np.arccos(np.clip(g[0, 1] / (a * b), -1.0, 1.0)))
    return tuple(float(v) for v in (a, b, c, al, be, ga))


def check_cell(space_group: str, cell) -> tuple[float, ...]:
    """Raise :class:`SymmetryError` unless ``cell`` (a, b, c, alpha, beta, gamma) fits the group.

    The cell's metric tensor must be invariant under every rotation of the
    space group (``R^T G R = G``). The check compares the stated cell with the
    cell of the group-averaged metric, within :data:`CELL_TOL`. Returns the
    symmetric cell. Covers every crystal system, including monoclinic unique
    axes and the rhombohedral ``:R`` setting.
    """
    rots = [R for R, _t in _expanded_ops(resolve_space_group(space_group, explicit_setting=True))]
    stated = tuple(float(v) for v in cell)
    g = _metric(stated)
    symmetric = _cell_from_metric(sum(R.T @ g @ R for R in rots) / len(rots))
    bad = []
    for name, v, s in zip(_CELL_NAMES, stated, symmetric):
        off = abs(v - s) / s if name in ("a", "b", "c") else abs(v - s)
        if off > CELL_TOL:
            bad.append(f"{name} = {v:g} (symmetric value {s:.10g})")
    if bad:
        system = resolve_space_group(space_group, explicit_setting=True).crystal_system_str()
        raise SymmetryError(
            f"unit cell is inconsistent with {space_group!r} ({system}): " + "; ".join(bad)
            + ". Symmetry-equivalent lengths must be equal and symmetry-fixed angles exact "
            "(e.g. 90 or 120 degrees)"
        )
    return symmetric


_UIJ_NAMES = ("U11", "U22", "U33", "U12", "U13", "U23")


def check_uij(space_group: str, xyz, uij) -> tuple[tuple[float, ...], bool]:
    """Fit the anisotropic ADPs ``uij`` (U11, U22, U33, U12, U13, U23) to the site symmetry.

    ``U`` must be invariant under the site's stabilizer (``R U R^T = U``, the
    convention GSAS-II uses). The site-symmetric tensor is the least-change
    (orthogonal) projection of ``uij`` onto the invariant tensors. Components further than :data:`UIJ_TOL` from it raise
    :class:`SymmetryError`; otherwise the symmetric values are returned, with a
    flag saying whether any component changed (by more than 1e-12).
    """
    site = analyze_site(space_group, xyz)
    ops = _expanded_ops(resolve_space_group(space_group, explicit_setting=True))
    x = np.asarray(site.canonical)
    stab = [R for R, t in ops if np.max(np.abs(_wrap_symmetric(R @ x + t - x))) < SPECIAL_POSITION_TOL]
    stated = np.asarray(uij, dtype=float)
    symmetric = _orthogonal_projector(_adp_projector(stab)) @ stated
    symmetric = np.where(np.abs(symmetric - stated) <= ADJUSTMENT_REPORT_TOL, stated, symmetric)
    bad = [f"{n} = {v:g} (site-symmetric value {s:.6g})"
           for n, v, s in zip(_UIJ_NAMES, stated, symmetric) if abs(v - s) > UIJ_TOL]
    if bad:
        raise SymmetryError(
            f"anisotropic ADPs break the site symmetry at {_fmt_xyz(x)} in {space_group!r}: "
            + "; ".join(bad)
        )
    if not np.any(np.abs(symmetric - stated) > ADJUSTMENT_REPORT_TOL):
        return tuple(float(v) for v in stated), False
    return tuple(float(v) for v in symmetric), True
