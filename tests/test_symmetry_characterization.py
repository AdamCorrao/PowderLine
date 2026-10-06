"""Characterization: core site analysis vs GSAS-II, all 230 space groups (re/03 T5, A69).

For every space group (two-origin groups in origin choice 2, rhombohedral groups
on hexagonal axes: the settings GSAS-II uses), many special and general
positions are analyzed by ``powderline.symmetry.analyze_site`` and by GSAS-II's
own ``SytSym`` / ``GetCSxinel`` / ``GetCSuinel``. They must agree on:

- multiplicity;
- per-axis coordinate DOF (FIXED / FREE / COUPLED);
- per-component Uij DOF.

The legacy ``site_dof`` / ``adp_dof`` (used by the 0.26.0 TOPAS writer) are held
to the same reference. GSAS-II is read only; nothing in it is changed. Skipped
where GSAS-II is not installed. A disagreement must be escalated, not
tolerated (A35).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

G2spc = pytest.importorskip("GSASII.GSASIIspc")
gemmi = pytest.importorskip("gemmi")

from powderline.exceptions import SymmetryError  # noqa: E402
from powderline.symmetry import _expanded_ops, adp_dof, analyze_site, site_dof  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_POINTS_PER_PATTERN = 2  # per space group: 12 patterns x this many draws


def _gemmi_group(number: int):
    sg = gemmi.find_spacegroup_by_number(number)
    if sg.ext == "1":  # two-origin group: GSAS-II uses origin choice 2
        sg = gemmi.find_spacegroup_by_name(sg.hm + ":2")
    return sg  # rhombohedral groups default to ':H' in gemmi, as in GSAS-II


def _same_mod1(p, q, tol=1e-6) -> bool:
    return bool(np.all(np.abs(((p - q) + 0.5) % 1.0 - 0.5) < tol))


def _orbit(ops, x):
    pts = []
    for R, t in ops:
        y = (R @ x + t) % 1.0
        if not any(_same_mod1(y, p, 1e-8) for p in pts):
            pts.append(y)
    return pts


def _gsas_symbol(sg):
    """A GSAS-II symbol for the same group *and setting*: the general-position orbits must match."""
    x = np.array([0.1234, 0.2345, 0.3456])
    ref = _orbit(_expanded_ops(sg), x)
    parts = sg.hm.split()
    candidates = [sg.hm, G2spc.StandardizeSpcName(sg.short_name())]
    if len(parts) == 4 and parts[1] == "1" and parts[3] == "1":  # 'P 1 21/c 1' -> 'P 21/c'
        candidates.append(f"{parts[0]} {parts[2]}")
    for cand in candidates:
        err, sgdata = G2spc.SpcGroup(cand)
        if err:
            continue
        theirs = [np.array(e[0]) % 1.0 for e in G2spc.GenAtom(x, sgdata, All=False)]
        if len(theirs) == len(ref) and all(any(_same_mod1(p, q) for q in theirs) for p in ref):
            return cand, sgdata
    return None, None


def _gsas_classes(codes) -> tuple:
    """GSAS-II constraint codes -> FIXED (0) / FREE (own group) / COUPLED (shared group)."""
    groups = Counter(c for c in codes if c != 0)
    return tuple("FIXED" if c == 0 else ("FREE" if groups[c] == 1 else "COUPLED") for c in codes)


def _positions(rng):
    """Grid points (multiples of 1/24) in patterns that hit special positions, incl. coupled ones."""
    g = lambda: rng.randrange(24) / 24  # noqa: E731
    pts = []
    for _ in range(_POINTS_PER_PATTERN):
        a, b, c = g(), g(), g()
        pts += [(a, b, c), (a, a, c), (a, -a, c), (a, 2 * a, c), (2 * a, a, c), (a, a, a),
                (a, a + 0.5, c), (a, a + 0.25, c), (a, c, c), (a, c, -c), (a, -a + 0.5, c),
                (a, b, b + 0.25)]
    return sorted({tuple(round(v % 1.0, 12) for v in p) for p in pts})


@pytest.mark.parametrize("number", range(1, 231))
def test_site_analysis_matches_gsasii(number):
    sg = _gemmi_group(number)
    name = sg.xhm()
    symbol, sgdata = _gsas_symbol(sg)
    assert symbol is not None, f"no GSAS-II symbol reproduces {name!r}"

    rng = random.Random(number)
    disagreements = []
    for x in _positions(rng):
        site = analyze_site(name, x)  # exact grid points never fall in the ambiguous band
        sitesym, mult = G2spc.SytSym(list(x), sgdata)[0:2]
        ours = {"multiplicity": site.multiplicity, "legacy multiplicity": site_dof(name, x).orbit_size}
        theirs = {"multiplicity": mult, "legacy multiplicity": mult}
        try:
            xyz = _gsas_classes(G2spc.GetCSxinel(sitesym)[0])
            uij = _gsas_classes(G2spc.GetCSuinel(sitesym)[0])
        except KeyError:
            # GSAS-II could not name this site symmetry ('sp'), so it has no
            # constraint table entry (seen once over all 230 groups: F432 24e).
            # Multiplicity is still compared.
            pass
        else:
            ours.update({"xyz": site.axes, "uij": site.adp,
                         "legacy xyz": site_dof(name, x).axes,
                         "legacy uij": adp_dof(name, x).components})
            theirs.update({"xyz": xyz, "uij": uij, "legacy xyz": xyz, "legacy uij": uij})
        for key in ours:
            if tuple(np.atleast_1d(ours[key])) != tuple(np.atleast_1d(theirs[key])):
                disagreements.append((x, sitesym, key, ours[key], theirs[key]))
    assert not disagreements, f"{name} (GSAS-II {symbol!r}): {disagreements[:5]}"


@pytest.mark.parametrize("space_group, exact, rounded", [
    ("P 63/m m c", (1 / 3, 2 / 3, 0.25), (0.333333, 0.666667, 0.25)),
    ("P 63/m m c", (1 / 6, 1 / 3, 0.25), (0.166667, 0.333333, 0.25)),
    ("F d -3 m:2", (0.125, 0.125, 0.125), (0.125, 0.125, 0.125)),
    ("R -3 m:H", (0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
    ("F m -3 m", (0.0, 0.0, 0.0), (-1e-9, 1e-9, 0.0)),
])
def test_canonical_coordinates_are_read_identically_by_gsasii(space_group, exact, rounded):
    """6-decimal input is canonicalized; GSAS-II gets the same site from the canonical values.

    The last case is GSAS-II's cell-edge defect (it misreads values just below an
    integer): the canonical exact 0 avoids it.
    """
    site = analyze_site(space_group, rounded)
    sg = gemmi.find_spacegroup_by_name(space_group)
    _symbol, sgdata = _gsas_symbol(sg)
    assert G2spc.SytSym(list(site.canonical), sgdata)[1] == site.multiplicity
    assert G2spc.SytSym(list(exact), sgdata)[1] == site.multiplicity
    exact_site = analyze_site(space_group, exact)
    assert np.allclose(site.canonical, exact_site.canonical, atol=1e-12)


@pytest.mark.parametrize("space_group, xyz", [
    ("P 63/m m c", (0.3333, 0.6667, 0.25)),
    ("P 63/m m c", (0.33333, 0.66667, 0.25)),
    ("P 63/m m c", (0.333, 0.667, 0.25)),
    ("F m -3 m", (1e-5, 0.0, 0.0)),
    ("F d -3 m:2", (0.12501, 0.125, 0.125)),
])
def test_ambiguous_band_is_rejected(space_group, xyz):
    with pytest.raises(SymmetryError, match="without being on it"):
        analyze_site(space_group, xyz)


def test_committed_examples_agree_with_gsasii():
    """Every atom of every committed example: same multiplicity from core and GSAS-II."""
    checked = 0
    for path in sorted(EXAMPLES.glob("*/input.json")):
        if path.parent.name == "example_template":
            continue
        phases = (json.loads(path.read_text(encoding="utf-8")).get("payload") or {}).get("phases") or {}
        for phase in phases.values():
            structure = phase["structure"]
            sg = gemmi.find_spacegroup_by_name(structure["space_group"])
            _symbol, sgdata = _gsas_symbol(sg)
            for atom in structure["atoms"].values():
                xyz = (atom["x"], atom["y"], atom["z"])
                site = analyze_site(sg.xhm(), xyz)
                assert site.multiplicity == G2spc.SytSym(list(xyz), sgdata)[1]
                checked += 1
    assert checked >= 20
