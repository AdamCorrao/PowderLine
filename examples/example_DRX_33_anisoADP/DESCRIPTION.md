# Multi-phase refinement with anisotropic displacement parameters (ADPs)

## Scientific Purpose

This example demonstrates **anisotropic atomic displacement parameter refinement** in PowderLine. While `example_DRX_33_atomrefine` uses isotropic displacement (single Uiso value per atom), this example refines **full anisotropic displacement tensors** with six independent components (U11, U22, U33, U12, U13, U23).

Anisotropic ADPs are critical for:
- **High-resolution structural analysis** - capturing directional thermal motion
- **Materials with anisotropic bonding** - different thermal motion along different crystal axes
- **Low-temperature refinements** - where anisotropic effects become significant
- **Publication-quality structures** - anisotropic ADPs are standard for single-crystal refinements

The refinement uses the same disordered rocksalt battery material (DRX_33 + Li4MgWO6_SG12) but focuses on the **DRX_33 phase** where the Li atom is refined with anisotropic ADPs while other atoms remain isotropic.

## Input Parameters

- **Data source**: Same as example_DRX_33 (embedded XRD data in JSON payload)
- **Phases being refined**: DRX_33 (cubic) and Li4MgWO6_SG12 (monoclinic)

### Atom Parameters Being Refined:

**DRX_33 Phase (cubic):**
- **Anisotropic displacement (Li atom only)**:
  - U11, U22, U33 (diagonal elements) - thermal motion along a, b, c axes
  - U12, U13, U23 (off-diagonal elements) - thermal motion correlations between axes
  - All 6 components have `refine_flag=true`
  - Coordinates: All locked (x=false, y=false, z=false)
  - Occupancy: Locked (occupancy=false)

- **Other atoms (Mg, Mn1, O1, W)**: All parameters locked
  - Isotropic displacement (Uiso): false
  - Coordinates: false
  - Occupancies: false

**Li4MgWO6_SG12 Phase (monoclinic):**
- All atomic parameters locked (no atom refinement)

### Parameters Constrained:
- **All atomic coordinates**: Locked in both phases
- **All occupancies**: Locked in both phases
- **Displacement parameters**: All locked except DRX_33 Li atom Uaniso
- **Instrument parameters**: Wavelength, polarization, broadening locked

### Parameters Refined:
- **Lattice parameters**: DRX_33 phase only (a-axis for cubic system)
- **Phase scale factors**: Both phases
- **Peak broadening**: Size and strain for both phases
- **Background**: Chebyshev with 6 terms
- **Anisotropic displacement**: DRX_33 Li atom (all 6 Uij components)

### Starting Values:
- **Anisotropic ADPs**: Initialized with reasonable values reflecting expected anisotropy
  - U11 = 0.0051, U22 = 0.004, U33 = 0.006 (diagonal)
  - U12 = U13 = U23 = 0.004 (off-diagonal)
- Other parameters: Same as example_DRX_33

## Expected Behavior

### Convergence:
- Refinement should converge after 5 cycles
- Anisotropic ADPs are highly correlated - expect large parameter correlations (>95%)
- U11, U22, U33 may shift differently depending on thermal ellipsoid orientation
- Off-diagonal terms (U12, U13, U23) may be small or near zero depending on symmetry

### Fit Quality:
- **Expected Rwp**: 12.58%
- Anisotropic ADPs increase parameter count but improve model flexibility
- Fit improvement over isotropic ADPs may be modest for powder data (anisotropy is often better resolved in single-crystal data)

### Typical Runtime:
- **Subprocess mode**: 2-4 seconds
- **Server mode**: 0.4-0.6 seconds

### Normal Warnings/Messages:
Typical refinement output with anisotropic ADPs:
```
Warning: Highly correlated parameters:
** Li:U11 and Li:U22 (@98.5%)
** Li:U11 and Li:U33 (@96.2%)
** Li:U12 and Li:U13 (@97.8%)
```

**This is EXPECTED behavior** for anisotropic displacement parameters:
- Powder diffraction provides limited information about anisotropy compared to single-crystal data
- High correlations between Uij components are normal and acceptable
- Off-diagonal terms (U12, U13, U23) are particularly correlated
- As long as the thermal ellipsoid is positive-definite and physically reasonable, correlations are not a problem

Additional warnings may include:
```
Warning: 2 soft (SVD) Hessian singularities
SVD problem(s) likely from: 0:0:Size;mx, 1:0:Size;i
```
These are related to peak broadening parameters and are independent of the anisotropic ADP refinement.

## Schema Features Demonstrated

This example showcases **anisotropic ADP support** and clarifies the distinction between **ADP type** and **refinement flags**:

### Understanding ADP:

The `ADP` field is a **type indicator**, not a refinement flag:
- `"ADP": "Uiso"` → tells kicker to look for the `Uiso` key
- `"ADP": "Uaniso"` → tells kicker to look for the `Uaniso` dict with U11-U23 keys
- **Refinement is controlled** by `refine_flag` in `[value, refine_flag, min, max]`

### Structure Section:
DRX_33 Li atom declares anisotropic displacement:
```json
"atoms": {
  "Li": {
    "ADP": "Uaniso",  // Type: use Uaniso dict
    "Uiso": 0.0051,
    "Uaniso": {
      "U11": 0.0051,
      "U22": 0.0051,
      "U33": 0.0051,
      "U12": 0.0,
      "U13": 0.0,
      "U23": 0.0
    },
    "element": "Li",
    "x": 0.0, "y": 0.0, "z": 0.0
  }
}
```

### Parameterization Section:
Refinement flags control which Uij components are refined:
```json
"parameterization": {
  "atoms": {
    "Li": {
      "ADP": "Uaniso",  // Type: look for Uaniso dict
      "Uaniso": {
        "U11": [0.0051, true, null, null],   // REFINED (true)
        "U22": [0.004, true, null, null],    // REFINED (true)
        "U33": [0.006, true, null, null],    // REFINED (true)
        "U12": [0.004, true, null, null],    // REFINED (true)
        "U13": [0.004, true, null, null],    // REFINED (true)
        "U23": [0.004, true, null, null]     // REFINED (true)
      },
      "Uiso": [null, false, null, null],     // NOT refined (false)
      "x": [null, false, null, null],        // NOT refined (false)
      "occupancy": [null, false, null, null] // NOT refined (false)
    },
    "Mg": {
      "ADP": "Uiso",  // Type: look for Uiso key
      "Uiso": [null, false, null, null],     // NOT refined (false)
      "x": [null, false, null, null],        // NOT refined (false)
      "occupancy": [null, false, null, null] // NOT refined (false)
    }
  }
}
```

**Key Points:**
- When `ADP="Uaniso"`, all 6 Uij values **must be provided** in both structure and parameterization
- Structure provides initial values; parameterization controls which are refined via `refine_flag`
- Uiso can coexist with Uaniso - it's calculated as (U11 + U22 + U33) / 3
- Mixed ADP types allowed: Li uses anisotropic, other atoms use isotropic
- `ADP` string is a **type indicator**, `refine_flag` controls actual refinement

## Physical Interpretation

### Thermal Ellipsoid:
The refined Uij values define a thermal ellipsoid representing the atom's vibrational motion:
- **Diagonal terms (U11, U22, U33)**: Motion along crystallographic axes a, b, c
- **Off-diagonal terms (U12, U13, U23)**: Correlations between axes

For a cubic material like DRX_33 (Fm-3m), we expect:
- U11 ≈ U22 ≈ U33 (isotropic-like behavior due to cubic symmetry)
- U12, U13, U23 near zero (no preferred correlation direction)

Significant deviations suggest:
- Anisotropic bonding environment
- Static disorder (atoms displaced from ideal positions)
- Incorrect structural model

### Positive-Definite Constraint:
The Uij tensor must be **positive-definite** (all eigenvalues > 0) to represent physical thermal motion. GSAS-II enforces this constraint during refinement. If refinement produces a non-positive-definite tensor, it indicates:
- Over-parameterization (too many refined ADPs for available data)
- Poor starting values
- Model errors

## Output Files

This two-phase Rietveld refinement produces:

- **dummy.gpx** - GSAS-II project file (reopenable in GUI)
- **dummy.lst** - Human-readable refinement log with Rwp and parameter tables
- **refined_parameters.csv** - All refined parameters with ESDs (9 columns: parameter_name, descriptive_name, phase_name, phase_idx, atom_name, atom_idx, value, esd, category)
- **DRX_33_unit_cell_report.csv** - Phase 1 unit cell parameters with ESDs (3 columns: parameter, value, esd)
- **DRX_33_peak_list_report.csv** - Phase 1 reflection list (hkl, d-spacing, 2θ, intensities)
- **Li4MgWO6_SG12_unit_cell_report.csv** - Phase 2 unit cell parameters with ESDs (3 columns: parameter, value, esd)
- **Li4MgWO6_SG12_peak_list_report.csv** - Phase 2 reflection list (hkl, d-spacing, 2θ, intensities)
- **fit_profile.txt** - Observed/calculated/background/difference intensities


## Known Issues

None. If anisotropic refinement fails to converge or produces unphysical Uij values:
1. Check that starting values are reasonable (diagonal > 0, off-diagonal small)
2. Consider fixing off-diagonal terms (U12, U13, U23) initially
3. Ensure sufficient data quality (high-resolution powder data required for anisotropic ADPs)

## Comparison to Related Examples

| Example | DRX_33 Atom Parameters | Li4MgWO6 Atom Parameters | Key Feature |
|---------|------------------------|--------------------------|-------------|
| example_DRX_33 | ❌ None | ❌ None | Base multi-phase refinement |
| example_DRX_33_atomrefine | ✅ x, Uiso, occupancy (all 5 atoms) | ✅ x only (7 atoms) | Comprehensive atom parameter refinement |
| example_DRX_33_anisoADP | ✅ Uaniso (Li only, 6 components) | ❌ None | **Full anisotropic displacement tensors** |
| example_DRX_33_simulation | ❌ All locked | ❌ All locked | Validate forward calculation only |

## When to Use Anisotropic ADPs

**Use anisotropic ADPs when:**
- High-resolution data available (d-spacing resolution < 1 Å)
- Publication-quality structure refinement needed
- Anisotropic bonding expected (layered materials, molecular crystals)
- Low-temperature data (thermal effects dominate)

**Use isotropic ADPs when:**
- Routine refinements (phase identification, quantification)
- Limited data resolution (powder diffraction with broad peaks)
- Early refinement stages (converge structure first, then add anisotropy)
- Highly disordered materials (anisotropy not physically meaningful)

## Schema 0.26 note: explicit per-parameter refine flags

Since schema 0.26, PowderLine honors each refinement flag individually: a
parameter refines iff it is present with `refine_flag=true`; absent or `false`
means fixed (internally enforced with GSAS-II "Hold" constraints).
Symmetry-linked parameters (e.g. cubic a=b=c) refine together if any member is
requested. This example uses the **explicit style** - every parameter is listed
with a symmetry-consistent flag so the recipe states its full intent. Listing
only the parameters you wish to refine (absence = fixed) is equally valid.

## Data citation

The diffraction data in this example (the disordered-rocksalt cathode
material, "DRX_33") is from the study available at
[doi:10.26434/chemrxiv.15003271/v1](https://doi.org/10.26434/chemrxiv.15003271/v1)
(preprint; to appear in a peer-reviewed publication). Please cite that work
when using this dataset.
