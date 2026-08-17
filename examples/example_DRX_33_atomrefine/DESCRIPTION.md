# Multi-phase refinement with atom parameter refinement (isotropic ADPs)

## Scientific Purpose

This example extends `example_DRX_33` to demonstrate **atom parameter refinement** capabilities. While the base example refines only lattice parameters, scale factors, and peak broadening, this example additionally refines:

- **Atomic coordinates** (x, y, z) - refining crystallographic positions
- **Isotropic displacement parameters (Uiso)** - refining thermal motion
- **Occupancies** - refining site occupancies (useful for disordered materials)

The refinement uses the same disordered rocksalt battery material with two phases:
1. **DRX_33** (main phase): Disordered rock salt (Fm-3m, cubic)
2. **Li4MgWO6_SG12** (secondary phase): Monoclinic (C2/m)

This example validates PowderLine's ability to refine structural parameters in multi-phase systems, which is critical for accurate structure determination.

## Input Parameters

- **Data source**: Same as example_DRX_33 (embedded XRD data in JSON payload)
- **Phases being refined**: DRX_33 (cubic) and Li4MgWO6_SG12 (monoclinic)

### Atom Parameters Being Refined:

**DRX_33 Phase (cubic):**
- **Coordinates (x, y, z)**: Li, Mg, Mn1, O1, W atoms all have x=y=z=true
  - Since schema 0.26, each coordinate flag is honored individually (a false or
    absent flag holds that coordinate). These atoms sit on special positions, so
    only the site-symmetry-allowed components actually vary.
- **Isotropic displacement (Uiso)**: Li, Mg, Mn1, O1, W atoms all have Uiso=true
- **Occupancies**: Li, Mg, Mn1, O1, W atoms all have occupancy=true

**Li4MgWO6_SG12 Phase (monoclinic):**
- **Coordinates (x, y, z)**: Li1, Li2, Li3, Mg, O1, O2, O3, W atoms all have x=y=z=true
  - Explicit flags state the full intent; site symmetry still fixes the
    components that are not free to vary.
- **Isotropic displacement (Uiso)**: All atoms locked (Uiso=false)
- **Occupancies**: All atoms locked (occupancy=false)

### Parameters Constrained:
- **Instrument parameters**: Wavelength, polarization, broadening locked

### Parameters Refined:
- **Lattice parameters**: Both phases
- **Phase scale factors**: Both phases
- **Peak broadening**: Size and strain for both phases
- **Background**: Chebyshev with 6 terms
- **Atomic coordinates**: Both phases (x, y, z all explicitly set to refine; site symmetry limits which components actually vary)
- **Displacement parameters**: DRX_33 phase only (isotropic Uiso for all 5 atoms)
- **Occupancies**: DRX_33 phase only (all 5 atoms)

### Starting Values:
- Atomic coordinates and Uiso from CIF structural data
- Other parameters: Same as example_DRX_33

## Expected Behavior

### Convergence:
- Refinement should converge after 5 cycles
- Atom parameters may show larger shifts in first few cycles
- Coordinate refinement is sensitive to starting values

### Fit Quality:
- **Expected Rwp**: ~19.35% (39 refined parameters)
- Higher Rwp than base example_DRX_33 (10.83%) due to increased parameter count and model complexity
- χ² should still be reasonable given the added structural freedom

### Typical Runtime:
- **Subprocess mode**: 2-4 seconds
- **Server mode**: 0.4-0.6 seconds (10x faster)

### Normal Warnings/Messages:
Typical refinement output includes:
```
Warning: 2 soft (SVD) Hessian singularities
SVD problem(s) likely from:
0:0:Size;mx, 1:0:Size;i

Note highly correlated parameters:
** 0:0:Mustrain;mx and 0:0:Mustrain;i (@100.00%)
** 1:0:Mustrain;mx and 1:0:Mustrain;i (@100.00%)
```

Additional expected warnings with atom refinement:
- High correlations between atom coordinates and displacement parameters (>90% normal)
- Correlations between atomic coordinates of nearby atoms
- Correlations between occupancy and displacement (if both refined)

**These correlations are typical for structural refinements** and indicate that multiple parameters affect peak intensities similarly. As long as the refinement converges and parameters are physically reasonable, this is not a problem.

## Schema Features Demonstrated

This example highlights the schema's **required ADP specification** and the distinction between **ADP type indicator** and **refinement control**:

### Understanding ADP:

The `ADP` field serves two purposes:
1. **Type Indicator**: Tells kicker which displacement parameter to use ("Uiso" or "Uaniso")
2. **NOT a refinement flag**: The `ADP` string itself doesn't control whether parameters are refined

**Refinement is controlled by the `refine_flag`** (second element in `[value, refine_flag, min, max]`):

### Structure Section:
Each atom explicitly declares its ADP type:
```json
"atoms": {
  "Li": {
    "ADP": "Uiso",  // Type indicator - use Uiso for this atom
    "Uiso": 0.0051,
    "element": "Li",
    "x": 0.0, "y": 0.0, "z": 0.0
  }
}
```

### Parameterization Section:
Refinement flags control what's actually refined:
```json
"parameterization": {
  "atoms": {
    "Li": {
      "ADP": "Uiso",  // Type indicator - look for "Uiso" key
      "x": [null, true, null, null],      // REFINED (true)
      "Uiso": [null, true, null, null],   // REFINED (true)
      "occupancy": [null, true, null, null]  // REFINED (true)
    },
    "Li1": {
      "ADP": "Uiso",  // Type indicator - look for "Uiso" key
      "x": [null, true, null, null],       // REFINED (true)
      "Uiso": [null, false, null, null],   // NOT refined (false)
      "occupancy": [null, false, null, null]  // NOT refined (false)
    }
  }
}
```

**Key Points:**
- `ADP` is **required** in both sections
- `ADP` value indicates which parameter key to look for ("Uiso" or "Uaniso")
- Actual refinement controlled by `refine_flag` in the RefinementParameter list
- Structure ADP can differ from parameterization ADP (allows flexibility)

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

None.

## Comparison to Related Examples

| Example | DRX_33 Atom Parameters | Li4MgWO6 Atom Parameters | Key Feature |
|---------|------------------------|--------------------------|-------------|
| example_DRX_33 | ❌ None | ❌ None | Base multi-phase refinement |
| example_DRX_33_atomrefine | ✅ x, Uiso, occupancy (all 5 atoms) | ✅ x only (7 atoms) | Demonstrate comprehensive atom parameter refinement |
| example_DRX_33_anisoADP | ✅ Uaniso (Li only, 6 components) | ❌ None | Demonstrate anisotropic displacement parameters |
| example_DRX_33_simulation | ❌ All locked | ❌ All locked | Validate forward calculation only |

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
