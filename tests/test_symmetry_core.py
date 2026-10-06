"""Tests for powderline.symmetry core validation functions (check_cell, check_uij, analyze_site).

Core-only: no GSAS-II needed; must run in a core-only environment.
"""
import pytest

from powderline.exceptions import SymmetryError
from powderline.symmetry import check_cell, check_uij, analyze_site, SiteAnalysis
from fractions import Fraction


# --- check_cell -------------------------------------------------------------


def test_check_cell_cubic_valid_returned():
    """check_cell: cubic P m -3 m (a=b=c, 90/90/90) returns symmetric cell."""
    sg = "P m -3 m"
    cell = (4.15692, 4.15692, 4.15692, 90.0, 90.0, 90.0)
    symmetric = check_cell(sg, cell)
    assert len(symmetric) == 6
    assert abs(symmetric[0] - cell[0]) < 1e-12


def test_check_cell_cubic_a_perturbed_rejected():
    """check_cell: cubic with b != a by 1e-6 relative rejected."""
    sg = "P m -3 m"
    a = 4.15692
    cell = (a, a * (1 + 1e-6), a, 90.0, 90.0, 90.0)
    with pytest.raises(SymmetryError, match="inconsistent"):
        check_cell(sg, cell)


def test_check_cell_cubic_gamma_perturbed_rejected():
    """check_cell: cubic with gamma != 90 by 1e-6 degree rejected."""
    sg = "P m -3 m"
    cell = (4.0, 4.0, 4.0, 90.0, 90.0, 90.0 + 1e-6)
    with pytest.raises(SymmetryError, match="inconsistent"):
        check_cell(sg, cell)


def test_check_cell_cubic_tiny_perturbation_accepted():
    """check_cell: cubic with perturbation below CELL_TOL (1e-12 relative) accepted."""
    sg = "P m -3 m"
    a = 4.0
    cell = (a, a * (1 + 1e-13), a, 90.0, 90.0, 90.0)
    symmetric = check_cell(sg, cell)
    assert symmetric is not None


def test_check_cell_tetragonal_valid_returned():
    """check_cell: tetragonal I 41/a m d:2 (a=b, c, 90/90/90) returns symmetric cell."""
    sg = "I 41/a m d:2"
    cell = (5.0, 5.0, 7.0, 90.0, 90.0, 90.0)
    symmetric = check_cell(sg, cell)
    assert abs(symmetric[0] - symmetric[1]) < 1e-12
    assert abs(symmetric[2] - 7.0) < 1e-12


def test_check_cell_tetragonal_a_neq_b_rejected():
    """check_cell: tetragonal with a != b by 1e-6 relative rejected."""
    sg = "I 41/a m d:2"
    cell = (5.0, 5.0 * (1 + 1e-6), 7.0, 90.0, 90.0, 90.0)
    with pytest.raises(SymmetryError, match="inconsistent"):
        check_cell(sg, cell)


def test_check_cell_hexagonal_valid_returned():
    """check_cell: hexagonal P 63/m m c (a=b, c, 90/90/120) returns symmetric cell."""
    sg = "P 63/m m c"
    cell = (3.2, 3.2, 5.2, 90.0, 90.0, 120.0)
    symmetric = check_cell(sg, cell)
    assert abs(symmetric[0] - symmetric[1]) < 1e-12
    assert abs(symmetric[5] - 120.0) < 1e-12


def test_check_cell_hexagonal_gamma_neq_120_rejected():
    """check_cell: hexagonal with gamma != 120 by 1e-6 degree rejected."""
    sg = "P 63/m m c"
    cell = (3.2, 3.2, 5.2, 90.0, 90.0, 120.0 + 1e-6)
    with pytest.raises(SymmetryError, match="inconsistent"):
        check_cell(sg, cell)


def test_check_cell_monoclinic_valid_returned():
    """check_cell: monoclinic C 1 2/m 1 (a!=b!=c, 90/beta/90) returns symmetric cell."""
    sg = "C 1 2/m 1"
    cell = (5.0, 6.0, 7.0, 90.0, 104.3, 90.0)
    symmetric = check_cell(sg, cell)
    assert abs(symmetric[4] - 104.3) < 1e-9


def test_check_cell_monoclinic_alpha_neq_90_rejected():
    """check_cell: monoclinic with alpha != 90 by 1e-6 degree rejected."""
    sg = "C 1 2/m 1"
    cell = (5.0, 6.0, 7.0, 90.0 + 1e-6, 104.3, 90.0)
    with pytest.raises(SymmetryError, match="inconsistent"):
        check_cell(sg, cell)


def test_check_cell_triclinic_valid_returned():
    """check_cell: triclinic P 1 (all free) returns cell unchanged."""
    sg = "P 1"
    cell = (5.0, 6.0, 7.0, 80.0, 85.0, 95.0)
    symmetric = check_cell(sg, cell)
    for i in range(6):
        assert abs(symmetric[i] - cell[i]) < 1e-12


def test_check_cell_rhombohedral_r_setting_valid_returned():
    """check_cell: rhombohedral R -3 m:R (a=b=c, alpha=beta=gamma!=90) returns symmetric cell."""
    sg = "R -3 m:R"
    alpha = 60.0
    cell = (5.0, 5.0, 5.0, alpha, alpha, alpha)
    symmetric = check_cell(sg, cell)
    assert abs(symmetric[0] - symmetric[1]) < 1e-12
    assert abs(symmetric[1] - symmetric[2]) < 1e-12
    assert abs(symmetric[3] - symmetric[4]) < 1e-12
    assert abs(symmetric[4] - symmetric[5]) < 1e-12


def test_check_cell_rhombohedral_r_setting_angle_perturbed_rejected():
    """check_cell: rhombohedral R -3 m:R with alpha != beta by 1e-6 degree rejected."""
    sg = "R -3 m:R"
    cell = (5.0, 5.0, 5.0, 60.0, 60.0 + 1e-6, 60.0)
    with pytest.raises(SymmetryError, match="inconsistent"):
        check_cell(sg, cell)


# --- check_uij --------------------------------------------------------------


def test_check_uij_identity_unchanged():
    """check_uij: conforming tensor returned unchanged, adjusted=False."""
    sg = "P m -3 m"
    xyz = (0.0, 0.0, 0.0)
    # Cubic m-3m: U11=U22=U33, off-diag 0
    uij = (0.01, 0.01, 0.01, 0.0, 0.0, 0.0)
    symmetric, adjusted = check_uij(sg, xyz, uij)
    assert adjusted is False
    for i in range(6):
        assert abs(symmetric[i] - uij[i]) < 1e-15


def test_check_uij_adjusted_within_tol():
    """check_uij: tensor within UIJ_TOL adjusted, adjusted=True."""
    sg = "P 63/m m c"
    xyz = (0.2, 0.4, 0.25)
    # Site 6h: U11 free, U22 free, U33 free, U12 = U22/2, U13=U23=0
    # State U12 slightly off from U22/2 by < UIJ_TOL
    uij = (0.012, 0.024, 0.02, 0.012 + 5e-7, 0.0, 0.0)
    symmetric, adjusted = check_uij(sg, xyz, uij)
    assert adjusted is True
    # Check U12 became exactly U22/2
    assert abs(symmetric[3] - symmetric[1] / 2) < 1e-15


def test_check_uij_beyond_tol_raises():
    """check_uij: tensor beyond UIJ_TOL raises SymmetryError."""
    sg = "P m -3 m"
    xyz = (0.0, 0.0, 0.0)
    # Cubic m-3m: U11=U22=U33; state U11 != U22 by > UIJ_TOL
    uij = (0.01, 0.02, 0.01, 0.0, 0.0, 0.0)
    with pytest.raises(SymmetryError, match="break the site symmetry"):
        check_uij(sg, xyz, uij)


def test_check_uij_general_position_unchanged():
    """check_uij: general position in P 1 accepts any tensor unchanged."""
    sg = "P 1"
    xyz = (0.123, 0.456, 0.789)
    uij = (0.01, 0.02, 0.03, 0.005, 0.007, 0.009)
    symmetric, adjusted = check_uij(sg, xyz, uij)
    assert adjusted is False
    for i in range(6):
        assert abs(symmetric[i] - uij[i]) < 1e-15


# --- analyze_site -----------------------------------------------------------


def test_analyze_site_general_position_multiplicity():
    """analyze_site: P m -3 m general position -> multiplicity 48 (number of operations)."""
    sg = "P m -3 m"
    xyz = (0.123, 0.456, 0.789)
    site = analyze_site(sg, xyz)
    assert site.multiplicity == 48


def test_analyze_site_adjusted_false_for_exact():
    """analyze_site: exact input -> adjusted=False."""
    sg = "P m -3 m"
    xyz = (0.0, 0.0, 0.0)
    site = analyze_site(sg, xyz)
    assert site.adjusted is False


def test_analyze_site_exact_fractions():
    """analyze_site: fixed coordinates in `exact` as Fraction(1, 3) etc."""
    sg = "P 63/m m c"
    xyz = (0.333333, 0.666667, 0.25)
    site = analyze_site(sg, xyz)
    # Check exact contains Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)
    assert site.exact[0] == Fraction(1, 3)
    assert site.exact[1] == Fraction(2, 3)
    assert site.exact[2] == Fraction(1, 4)


def test_analyze_site_explicit_setting_required():
    """analyze_site: 'F d -3 m' without :1/:2 raises SymmetryError."""
    xyz = (0.125, 0.125, 0.125)
    with pytest.raises(SymmetryError, match="two origin choices"):
        analyze_site("F d -3 m", xyz)


def test_analyze_site_non_finite_raises():
    """analyze_site: non-finite xyz raises SymmetryError."""
    sg = "P 1"
    xyz = (float("nan"), 0.0, 0.0)
    with pytest.raises(SymmetryError, match="must be a finite 3-vector"):
        analyze_site(sg, xyz)


def test_analyze_site_cubic_origin_multiplicity_1():
    """analyze_site: P m -3 m origin (0,0,0) -> multiplicity 1."""
    sg = "P m -3 m"
    xyz = (0.0, 0.0, 0.0)
    site = analyze_site(sg, xyz)
    assert site.multiplicity == 1


def test_analyze_site_hexagonal_6h_multiplicity_6():
    """analyze_site: P 63/m m c (0.2,0.4,0.25) -> multiplicity 6 (site 6h)."""
    sg = "P 63/m m c"
    xyz = (0.2, 0.4, 0.25)
    site = analyze_site(sg, xyz)
    assert site.multiplicity == 6


# --- ambiguous-band message ----------------------------------------------------


def test_is_group_detects_closure():
    from powderline.symmetry import _expanded_ops, _is_group, resolve_space_group

    ops = _expanded_ops(resolve_space_group("P 63/m m c"))
    assert _is_group(ops)
    assert _is_group(ops[:1])  # identity alone
    two_fold = [op for op in ops if round(float(op[0].trace())) == -1][0]
    assert not _is_group([two_fold])  # missing the identity


def test_ambiguous_band_message_names_the_special_position():
    with pytest.raises(SymmetryError, match=r"from the special position \(1/3, 2/3, 1/4\) \(multiplicity 2\)"):
        analyze_site("P 63/m m c", (0.3333, 0.6667, 0.25))
