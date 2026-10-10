"""Verify which space-group settings each easydiffraction calculator computes correctly (re/06, EB-42, EB-80).

For every entry of ``space_group_table.SPACE_GROUPS``: one Fe atom at a general
position (0.137, 0.271, 0.389) in a cell of the setting's crystal system.

Each calculator is verified on its own (A170):

- CrysPy: per-reflection F^2 against a reference computed here (the orbit
  expanded with gemmi's operations of the canonical setting, i.e. the ITC
  operators; IT92 form factors, isotropic Debye-Waller), R after a least-squares
  scale. Sensitive to the hand: CrysPy computes P 31 with P 32's operators (same
  powder pattern, the other hand per reflection), so P 31 fails.
- CrysFML (no reflection list, operators not exposed): its calculated pattern
  against its own pattern of the same orbit written out in P 1 (no symmetry
  handling involved), same pseudo-Voigt profile, R after a least-squares scale.

A setting passes with R < ``R_PASS`` (easydiffraction 0.21.1: CrysPy passes R <= 0.009,
failures >= 0.075; CrysFML passes R < 1e-4, failures >= 0.11; a failure means a misread
setting). An engine error fails that calculator only (EB-84: the 53 crashes are CrysPy's). The result is written to
``src/powderline/gateways/easydiffraction/space_group_support.py``, which the schema
uses so that validation needs no easydiffraction. About an hour; ``--jobs N`` runs
N processes. ``--check XHM ...`` re-verifies only the named settings against the
committed module (the test does this for a fixed sample).

Usage: pixi run -e easydiff python scripts/verify_easydiffraction_settings.py [--jobs N] [--check XHM ...]
"""

from __future__ import annotations

import math
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).parent.parent
MODULE = ROOT / "src" / "powderline" / "gateways" / "easydiffraction" / "space_group_support.py"
XYZ = (0.137, 0.271, 0.389)
U = 0.005
WAVELENGTH = 0.5
R_PASS = 0.03
_CELLS = {"cubic": (5.1, 5.1, 5.1, 90, 90, 90), "tetragonal": (5.1, 5.1, 7.3, 90, 90, 90),
          "hexagonal": (5.1, 5.1, 7.3, 90, 90, 120), "trigonal": (5.1, 5.1, 7.3, 90, 90, 120),
          "orthorhombic": (5.1, 6.2, 7.3, 90, 90, 90), "triclinic": (5.1, 6.2, 7.3, 81, 103, 95)}


def _unique_axis(sg: gemmi.SpaceGroup) -> str:
    for op in sg.operations():
        rot = np.array(op.rot) / op.DEN
        if np.isclose(np.trace(rot), -1) and np.isclose(np.linalg.det(rot), 1):  # a 2-fold rotation
            return "abc"[int(np.argmax(np.diag(rot)))]
    return "b"


def cell_for(sg: gemmi.SpaceGroup) -> tuple:
    system = sg.crystal_system_str()
    if system == "trigonal" and sg.xhm().endswith(":R"):
        return (5.6, 5.6, 5.6, 77, 77, 77)
    if system == "monoclinic":
        angles = {"a": (103, 90, 90), "b": (90, 103, 90), "c": (90, 90, 103)}[_unique_axis(sg)]
        return (5.1, 6.2, 7.3, *angles)
    return _CELLS[system]


def orbit(sg: gemmi.SpaceGroup) -> list[np.ndarray]:
    """The general position ``XYZ``'s orbit under gemmi's (ITC) operations, each point once, in [0, 1)."""
    pts = []
    for op in sg.operations():
        q = np.mod(np.array(op.apply_to_xyz(list(XYZ)), dtype=float), 1.0)
        if not any(np.allclose(np.mod(q - r + 0.5, 1.0) - 0.5, 0, atol=1e-9) for r in pts):
            pts.append(q)
    return pts


def reference_f2(sg: gemmi.SpaceGroup, cell: tuple, hkls) -> np.ndarray:
    pts = orbit(sg)
    uc = gemmi.UnitCell(*cell)
    out = []
    for hkl in hkls:
        s2 = 1.0 / (4 * uc.calculate_d(list(hkl)) ** 2)
        f = gemmi.Element("Fe").it92.calculate_sf(s2) * math.exp(-8 * math.pi ** 2 * U * s2)
        phases = [2 * math.pi * float(np.dot(hkl, r)) for r in pts]
        out.append(abs(sum(f * complex(math.cos(a), math.sin(a)) for a in phases)) ** 2)
    return np.array(out)


def _experiment(name: str, code: str, cell: tuple, calculator: str, data_path: Path, atoms=(XYZ,)):
    from easydiffraction import ExperimentFactory, Project, StructureFactory

    s = StructureFactory.from_scratch(name="t")
    s.space_group.name_h_m = name
    if code:
        s.space_group.coord_system_code = code
    for k, v in zip(("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"), cell):
        setattr(s.cell, k, v)
    for i, xyz in enumerate(atoms):
        label = f"Fe{i + 1}"
        s.atom_sites.create(id=label, type_symbol="Fe", fract_x=float(xyz[0]), fract_y=float(xyz[1]),
                            fract_z=float(xyz[2]), occupancy=1.0)
        s.atom_sites[label].adp_type = "Uiso"
        s.atom_sites[label].adp_iso = U
    e = ExperimentFactory.from_data_path(name="e", data_path=str(data_path), sample_form="powder",
                                         beam_mode="constant wavelength", radiation_probe="xray",
                                         scattering_type="bragg")
    e.calculator.type = calculator
    e.peak.type = "cwl-pseudo-voigt"
    e.peak.broad_gauss_u, e.peak.broad_gauss_v, e.peak.broad_gauss_w = 0.0, 0.0, 0.002
    e.peak.broad_lorentz_x, e.peak.broad_lorentz_y = 0.0, 0.01
    e.instrument.setup_wavelength = WAVELENGTH
    e.linked_structures.create(structure_id="t", scale=1.0)
    p = Project()
    p.structures.add(s)
    p.experiments.add(e)
    p.analysis.calculate()
    return e


def _ls_r(a: np.ndarray, b: np.ndarray) -> float:
    s = float(np.dot(a, b) / np.dot(b, b))
    return float(np.sum(np.abs(a - s * b)) / np.sum(np.abs(s * b)))


def _pattern(experiment) -> np.ndarray:
    arrays = experiment.data.fit_data_arrays()
    return np.asarray(arrays["calc"]) - np.asarray(arrays["bkg"])


def verify(xhm: str, entry: tuple, data_path: Path) -> tuple[bool, bool, str]:
    """``(cryspy_ok, crysfml_ok, note)`` for one setting; each calculator on its own (A170)."""
    name, code, _default = entry
    sg = gemmi.find_spacegroup_by_name(xhm)
    cell = cell_for(sg)
    try:
        r = _experiment(name, code, cell, "cryspy", data_path).refln
        hkl = [(int(h), int(k), int(l)) for h, k, l in zip(r.index_h, r.index_k, r.index_l)]
        fc = np.array(r.f_squared_calc, dtype=float)
        fr = reference_f2(sg, cell, hkl)
        keep = fr > 1e-6 * fr.max()
        r_cryspy = _ls_r(fc[keep], fr[keep])
        cryspy_ok, cryspy_note = r_cryspy < R_PASS, f"cryspy R={r_cryspy:.3f}"
    except Exception as exc:  # noqa: BLE001  an engine crash means the setting is unusable
        cryspy_ok, cryspy_note = False, f"cryspy {type(exc).__name__}: {str(exc)[:60]}"
    try:
        y_setting = _pattern(_experiment(name, code, cell, "crysfml", data_path))
        y_p1 = _pattern(_experiment("P 1", "", cell, "crysfml", data_path, atoms=orbit(sg)))
        r_crysfml = _ls_r(y_setting, y_p1)
        crysfml_ok, crysfml_note = r_crysfml < R_PASS, f"crysfml R={r_crysfml:.3f}"
    except Exception as exc:  # noqa: BLE001
        crysfml_ok, crysfml_note = False, f"crysfml {type(exc).__name__}: {str(exc)[:60]}"
    return cryspy_ok, crysfml_ok, f"{cryspy_note}, {crysfml_note}"


def _grid(directory: Path) -> Path:
    path = directory / "grid.xye"
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x in np.arange(5.0, 30.0, 0.005):
            f.write(f"{x:.4f} 1.0 1.0\n")
    return path


def _verify_chunk(items: list) -> list:
    from easydiffraction.utils.logging import Logger

    Logger.configure(reaction=Logger.Reaction.RAISE)
    with tempfile.TemporaryDirectory() as tmp:
        data = _grid(Path(tmp))
        return [(xhm, *verify(xhm, entry, data)) for xhm, entry in items]


def render(results: list) -> str:
    cryspy = [xhm for xhm, ok, _f, _n in results if ok]
    crysfml = [xhm for xhm, _c, ok, _n in results if ok]
    lines = [
        '"""Space-group settings each easydiffraction calculator computes correctly (generated; do not edit).',
        "",
        "Generated by ``scripts/verify_easydiffraction_settings.py`` from easydiffraction",
        f"{version('easydiffraction')} by computation (EB-42, EB-80): a setting is listed when the calculator's",
        "per-reflection structure factors (CrysPy, against an ITC reference) or pattern (CrysFML, against its own",
        f"pattern of the orbit written out in P 1) for a general-position atom match within R < {R_PASS} (A170).",
        "CrysFML is handed only the short symbol, so it misreads most non-default settings; easydiffraction's cell",
        "constraints assume b-unique monoclinic cells; some settings crash in CrysPy.",
        '"""',
        "",
        f"CRYSPY_SETTINGS: frozenset[str] = frozenset({{  # {len(cryspy)} of {len(results)}",
        *[f"    {x!r}," for x in cryspy],
        "})",
        "",
        f"CRYSFML_SETTINGS: frozenset[str] = frozenset({{  # {len(crysfml)} of {len(results)}",
        *[f"    {x!r}," for x in crysfml],
        "})",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    from powderline.gateways.easydiffraction.space_group_table import SPACE_GROUPS

    args = sys.argv[1:]
    if "--check" in args:
        names = args[args.index("--check") + 1:]
        from powderline.gateways.easydiffraction import space_group_support as support

        bad = []
        for xhm, ok_cryspy, ok_crysfml, note in _verify_chunk([(x, SPACE_GROUPS[x]) for x in names]):
            if (ok_cryspy, ok_crysfml) != (xhm in support.CRYSPY_SETTINGS, xhm in support.CRYSFML_SETTINGS):
                bad.append(f"{xhm}: computed cryspy={ok_cryspy} crysfml={ok_crysfml} ({note}); module disagrees")
        print("\n".join(bad) or "ok")
        return 1 if bad else 0
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
    items = list(SPACE_GROUPS.items())
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunks = [items[i::jobs] for i in range(jobs)]
        with ProcessPoolExecutor(jobs) as pool:
            parts = list(pool.map(_verify_chunk, chunks))
        done = {xhm: row for part in parts for xhm, *row in part}
        results = [(xhm, *done[xhm]) for xhm, _ in items]
    else:
        results = _verify_chunk(items)
    MODULE.write_text(render(results), encoding="utf-8", newline="\n")
    print(f"wrote {MODULE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
