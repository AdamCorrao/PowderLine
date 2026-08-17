# Fit to cathode material diffraction data with multiple phases using the file-less payload

## Scientific Purpose

I am fitting a diffraction pattern collected on a disordered rocksalt battery material.
Both the main phase of interest and a secondary phase are being fit. The main phase is a disordered rock salt (face centered cubic, space group Fm-3m, space group number 225) and the secondary phase is Li4MgWO6 (monoclinic, space group C2/m, space group number 12).
I am refining the phase scale factors, lattice parameterss, phase size broadening including the LG_eta (Lorentzian-Gaussian weighting) parameters, phase strain broadening including the Lg_eta parameters, and a Chebyshev background with 6 terms,
The intention is to quantify the relative phase fractions from the refined scale factors.

## Schema Features Demonstrated

This example demonstrates multi-phase refinement capabilities:

- **Multi-phase refinement**: Two phases with different crystal systems (cubic + monoclinic)
- **Phase-specific parameters**: Independent scale, lattice, and broadening parameters for each phase
- **Peak broadening modeling**: Both size and strain broadening with LG_eta (Lorentzian-Gaussian mixing)
- **Constrained atomic structure**: Atomic positions, occupancies, and ADPs fixed (structure-only refinement)
- **File-less payload**: Complete multi-phase dataset embedded in JSON
- **Phase fraction quantification**: Refined scale factors enable phase fraction determination

## Input Parameters

- What data file(s) are you using?
    None - this is an example of the payload carrying all necessary information including the diffraction data, instrument parameters, and structure information.
- What phases are being refined?
    DRX (space group Fm-3m, cubic) and Li4MgWO6 (space group C2/m, monoclinic)
- What parameters are constrained vs refined?
    Instrument parameters, atomic positions, atomic occupancies, and atomic displacement parameters are constrained.
    Lattice parameters, phase scale factors, phase size broadening, phase strain broadening, and Chebyshev background are refined.
- Starting values for refinement
    Lattice parameters and atomic coordinates from a CIF. Phase lattice strain (mustrain) set to 1 with an LG_eta parameter of 1.
    Phase scale factor defaulted to 1, crystallite size defaulted to 1 (meaning 1 micron crystallite size), and size LG_eta parameter starts at 1.
- Any special GSAS-II settings or options
    None.

## Expected Behavior

- What parameters should converge?
- Expected fit quality (Rwp, χ²)
    Rwp = 10.83%
- Typical runtime
    Less than 10 seconds.
- Any warnings or messages that are normal
    Reported from refinement:
    Warning: 2 soft (SVD) Hessian singularities
    SVD problem(s) likely from:
    0:0:Size;mx, 1:0:Size;i
    Note highly correlated parameters:
    ** 0:0:Mustrain;mx and 0:0:Mustrain;i (@100.00%)
    ** 1:0:Mustrain;mx and 1:0:Mustrain;i (@100.00%)

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
