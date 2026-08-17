"""gemmi-backed crystallographic rules for the TOPAS writer (plan §5).

Two pure capabilities, both GSAS-II-free:

* :func:`cell_constraints` -- crystal-system cell equalities and fixed angles,
  so the writer can share one ``prm`` name across equal cell lengths and emit
  symmetry-fixed angles as bare constants.
* :func:`site_dof` -- per-axis site-symmetry degrees of freedom (FREE / FIXED /
  COUPLED) plus the orbit size, so the writer can decide whether a *refined*
  atomic coordinate is legal and warn on suspicious multiplicities.

Space-group operations come from gemmi; the stabiliser / projector math is our
own (~small, spec in plan §5). ``gemmi`` is an approved runtime dependency (D3).
"""

from __future__ import annotations

from dataclasses import dataclass

import gemmi
import numpy as np

from .errors import TopasTranslationError

# Fractional-coordinate tolerances (plan §5).
_POS_TOL = 1e-5   # position / orbit equality mod 1
_MAT_TOL = 1e-6   # rotation-matrix / projector comparisons

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


def resolve_space_group(space_group: str) -> gemmi.SpaceGroup:
    """Resolve a recipe space-group string to a gemmi ``SpaceGroup``.

    Tries the symbol verbatim, then whitespace-stripped (gemmi accepts both
    ``"P m -3 m"`` and ``"Pm-3m"``). Rhombohedral ``:R`` settings and
    unresolvable symbols raise :class:`TopasTranslationError`. Two-origin groups
    passed without an explicit ``:1``/``:2`` selector also error rather than
    guess an origin (plan §5(6); demo groups are single-origin).
    """
    raw = str(space_group)
    sg = gemmi.find_spacegroup_by_name(raw)
    if sg is None:
        sg = gemmi.find_spacegroup_by_name(raw.replace(" ", ""))
    if sg is None:
        raise TopasTranslationError(
            f"unrecognized space-group symbol {space_group!r} (gemmi could not resolve it)"
        )
    if sg.ext == "R":
        raise TopasTranslationError(
            f"rhombohedral ':R' setting not supported for {space_group!r}; "
            "use the hexagonal (:H) setting"
        )
    if sg.ext in ("1", "2") and ":" not in raw:
        raise TopasTranslationError(
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
    raise TopasTranslationError(
        "could not determine the monoclinic unique axis (no proper 2-fold found)"
    )


def cell_constraints(space_group: str) -> CellRules:
    """Cell-length equality groups and fixed/free angles for ``space_group``.

    See :class:`CellRules`. Rhombohedral ``:R`` and unknown symbols raise
    :class:`TopasTranslationError` (via :func:`resolve_space_group`).
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
        # trigonal reaches here only in the hexagonal (:H) setting -- :R errored
        # out in resolve_space_group. Cell is a=b, 90/90/120.
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

    raise TopasTranslationError(
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


def site_dof(space_group: str, xyz) -> SiteDof:
    """Classify the site DOF of fractional position ``xyz`` in ``space_group``.

    Algorithm (plan §5(b)):

    1. Expand all symmetry operations (with centering).
    2. Stabiliser = ops with ``R x + t == x (mod 1)`` within ``_POS_TOL``.
    3. Allowed-displacement subspace = fixed space of the stabiliser rotations,
       given by the Reynolds projector ``P = mean(R_i)`` (an orthogonal
       projector for a group of orthogonal matrices).
    4. Per axis ``e_j``: ``P e_j == e_j`` -> FREE; ``P e_j == 0`` -> FIXED;
       otherwise -> COUPLED (subspace not axis-aligned along ``j``).
    """
    x = np.asarray(xyz, dtype=float)
    if x.shape != (3,):
        raise ValueError("xyz must be a 3-vector")
    ops = _expanded_ops(resolve_space_group(space_group))

    stab_rots = [R for (R, t) in ops if np.all(np.abs(_wrap_symmetric(R @ x + t - x)) < _POS_TOL)]
    if not stab_rots:  # pragma: no cover - identity is always in the group
        raise TopasTranslationError("empty stabilizer (should be impossible)")

    projector = sum(stab_rots) / len(stab_rots)

    axes: list[str] = []
    coupled: list[str] = []
    for j in range(3):
        col = projector[:, j]
        unit = np.zeros(3)
        unit[j] = 1.0
        if np.linalg.norm(col - unit) < _MAT_TOL:
            axes.append("FREE")
        elif np.linalg.norm(col) < _MAT_TOL:
            axes.append("FIXED")
        else:
            axes.append("COUPLED")
            coupled.append(_AXIS_LETTERS[j])

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
    given by the Reynolds projector on the 6-dim symmetric-tensor space. A
    component ``e_j`` with ``P e_j == e_j`` is FREE, ``== 0`` is FIXED, else
    COUPLED (e.g. cubic ``m-3m`` -> ``u11=u22=u33`` coupled, off-diagonals FIXED).
    """
    x = np.asarray(xyz, dtype=float)
    if x.shape != (3,):
        raise ValueError("xyz must be a 3-vector")
    ops = _expanded_ops(resolve_space_group(space_group))
    stab = [R for (R, t) in ops if np.all(np.abs(_wrap_symmetric(R @ x + t - x)) < _POS_TOL)]

    projector = np.zeros((6, 6))
    for R in stab:
        for j in range(6):
            basis = np.zeros(6)
            basis[j] = 1.0
            projector[:, j] += _vec_from_sym(R @ _sym_from_vec(basis) @ R.T)
    projector /= len(stab)

    comps = []
    for j in range(6):
        col = projector[:, j]
        unit = np.zeros(6)
        unit[j] = 1.0
        if np.linalg.norm(col - unit) < _MAT_TOL:
            comps.append("FREE")
        elif np.linalg.norm(col) < _MAT_TOL:
            comps.append("FIXED")
        else:
            comps.append("COUPLED")
    return AdpDof(components=tuple(comps), stabilizer_order=len(stab))
