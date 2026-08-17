"""Tests for powderline.topas.symmetry (plan §5 targets + error cases)."""

from __future__ import annotations

import pytest

from powderline.topas import symmetry as sym
from powderline.topas.errors import TopasTranslationError


# --- (a) cell rules ---------------------------------------------------------


def test_cubic_cell_rules():
    r = sym.cell_constraints("P m -3 m")
    assert r.crystal_system == "cubic"
    assert r.length_groups == (("a", "b", "c"),)
    assert r.fixed_angles == ("alpha", "beta", "gamma")
    assert r.free_angles == ()
    assert r.unique_axis is None


def test_cubic_accepts_spaceless_symbol():
    assert sym.cell_constraints("Pm-3m").length_groups == (("a", "b", "c"),)
    assert sym.cell_constraints("Fm-3m").length_groups == (("a", "b", "c"),)


def test_monoclinic_unique_b_cell_rules():
    r = sym.cell_constraints("C 1 2/m 1")
    assert r.crystal_system == "monoclinic"
    assert r.length_groups == (("a",), ("b",), ("c",))
    assert r.unique_axis == "b"
    assert r.free_angles == ("beta",)
    assert set(r.fixed_angles) == {"alpha", "gamma"}


def test_monoclinic_c2overm_shorthand():
    assert sym.cell_constraints("C2/m").unique_axis == "b"


def test_monoclinic_unique_a_and_c():
    assert sym.cell_constraints("P 2/m 1 1").unique_axis == "a"
    assert sym.cell_constraints("P 2/m 1 1").free_angles == ("alpha",)
    assert sym.cell_constraints("P 1 1 2/m").unique_axis == "c"
    assert sym.cell_constraints("P 1 1 2/m").free_angles == ("gamma",)


def test_tetragonal_hexagonal_ortho_triclinic():
    assert sym.cell_constraints("P 4/m m m").length_groups == (("a", "b"), ("c",))
    hexr = sym.cell_constraints("P 6/m m m")
    assert hexr.length_groups == (("a", "b"), ("c",))
    assert hexr.fixed_angles == ("alpha", "beta", "gamma")
    orth = sym.cell_constraints("P m m m")
    assert orth.length_groups == (("a",), ("b",), ("c",))
    tri = sym.cell_constraints("P -1")
    assert tri.free_angles == ("alpha", "beta", "gamma")
    assert tri.fixed_angles == ()


def test_trigonal_hex_setting_is_hexagonal_cell():
    r = sym.cell_constraints("R -3 m")  # defaults to :H
    assert r.length_groups == (("a", "b"), ("c",))
    assert r.fixed_angles == ("alpha", "beta", "gamma")


# --- error cases ------------------------------------------------------------


def test_unknown_symbol_errors():
    with pytest.raises(TopasTranslationError, match="unrecognized"):
        sym.cell_constraints("NotASpaceGroup")


def test_rhombohedral_R_setting_errors():
    with pytest.raises(TopasTranslationError, match="rhombohedral"):
        sym.cell_constraints("R -3 m :R")
    with pytest.raises(TopasTranslationError, match="rhombohedral"):
        sym.site_dof("R -3 m :R", (0.0, 0.0, 0.0))


def test_two_origin_without_selector_errors():
    with pytest.raises(TopasTranslationError, match="two origin"):
        sym.cell_constraints("F d -3 m")
    # explicit origin is accepted
    assert sym.cell_constraints("F d -3 m:2").crystal_system == "cubic"


# --- (b) site DOF -----------------------------------------------------------


def test_pm3m_origin_all_fixed_orbit1():
    d = sym.site_dof("P m -3 m", (0.0, 0.0, 0.0))
    assert d.axes == ("FIXED", "FIXED", "FIXED")
    assert d.orbit_size == 1
    assert d.coupled_axes == ()


def test_pm3m_6f_site_z_free_orbit6():
    d = sym.site_dof("P m -3 m", (0.5, 0.5, 0.2021))
    assert d.axes == ("FIXED", "FIXED", "FREE")
    assert d.orbit_size == 6
    assert d.is_free("z") and not d.is_free("x")


def test_fm3m_origin_all_fixed_orbit4():
    d = sym.site_dof("F m -3 m", (0.0, 0.0, 0.0))
    assert d.axes == ("FIXED", "FIXED", "FIXED")
    assert d.orbit_size == 4  # matches DRX_33 recipe Multiplicity


def test_c2m_general_position_all_free_orbit8():
    d = sym.site_dof("C 1 2/m 1", (0.238, 0.343, 0.234))
    assert d.axes == ("FREE", "FREE", "FREE")
    assert d.orbit_size == 8


def test_c2m_special_y_free_xz_fixed():
    # DRX_33 Li1-type (0, 0.33, 0.5) and shared Li2/Mg site (0, 0.167, 0)
    for pos in [(0.0, 0.33, 0.5), (0.0, 0.167, 0.0)]:
        d = sym.site_dof("C 1 2/m 1", pos)
        assert d.axes == ("FIXED", "FREE", "FIXED")
        assert d.orbit_size == 4  # real multiplicity (recipe stores 1 -> writer warns)


def test_coupled_site_detected():
    # P4/mmm (x, x, z): a diagonal mirror couples x and y -> COUPLED
    d = sym.site_dof("P 4/m m m", (0.3, 0.3, 0.2))
    assert d.axes == ("COUPLED", "COUPLED", "FREE")
    assert set(d.coupled_axes) == {"x", "y"}


def test_site_dof_rejects_bad_shape():
    with pytest.raises(ValueError):
        sym.site_dof("P m -3 m", (0.0, 0.0))


# --- anisotropic-ADP DOF (adp_dof) ------------------------------------------


def test_adp_dof_fm3m_origin_isotropic():
    # Cubic m-3m site: u11=u22=u33 coupled (isotropic), off-diagonals fixed to 0.
    d = sym.adp_dof("F m -3 m", (0.0, 0.0, 0.0))
    assert d.classification("u11") == "COUPLED"
    assert d.classification("u22") == "COUPLED"
    assert d.classification("u33") == "COUPLED"
    assert d.classification("u12") == "FIXED"
    assert d.classification("u13") == "FIXED"
    assert d.classification("u23") == "FIXED"
    assert d.free() == ()  # no independently free component


def test_adp_dof_c2m_general_all_free():
    # C2/m general position: all six u_ij independently refinable.
    d = sym.adp_dof("C 1 2/m 1", (0.238, 0.343, 0.234))
    assert d.components == ("FREE",) * 6
    assert set(d.free()) == {"u11", "u22", "u33", "u12", "u13", "u23"}


def test_adp_dof_c2m_unique_b_off_diagonals_fixed():
    # C2/m with 2-fold along b at a special site: u12 and u23 forbidden (=0),
    # u11,u22,u33,u13 free.
    d = sym.adp_dof("C 1 2/m 1", (0.0, 0.33, 0.5))
    assert d.classification("u12") == "FIXED"
    assert d.classification("u23") == "FIXED"
    assert set(d.free()) == {"u11", "u22", "u33", "u13"}


def test_adp_dof_rejects_bad_shape():
    with pytest.raises(ValueError):
        sym.adp_dof("P m -3 m", (0.0, 0.0))
