"""GSAS-II gateway — transient legacy executor for 0.26.0 ``GSASII_*`` recipes (re/04, A39).

Verbatim copies, taken at ``2810c82`` (before the re/04 input-side edits), of
every function the re/04 frozen-edits ledger touches, and of every function on
the 0.26.0 path from ``run_refinement`` that calls one of them, so a 0.26.0
recipe runs only unedited code: the before/after-refactor comparison stays
exact (maintainer, ledger sign-off 2026-10-08). The functions imported below are
unedited and shared. Generated (devkit ``tasks/re04-frozen-edits.md``); deleted
with the 0.26.0 schema in re/07.

Do not edit: the bodies below are byte-identical to their originals.
"""
from __future__ import annotations
import sys
import uuid
import yaml
import json
import argparse
import random as ran
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from pathlib import Path
from pydantic import ValidationError
from GSASII import GSASIIscriptable as G2  # type: ignore
from GSASII import GSASIIlattice as G2lat
from GSASII import GSASIImapvars as G2mv
from GSASII import GSASIIobj as G2obj
from GSASII import GSASIIspc as G2spc
from GSASII import GSASIIElem as G2elem
from GSASII import GSASIIpwd as G2pwd
from powderline.schema import RecipeModel, RefinementControls
from powderline.gateways.gsasii.constraints import atom_refinement_plan, cell_refinement_plan
from powderline._status import CHECK, CROSS, INFO, WARN, emoji
from dataclasses import dataclass

from powderline.gateways.gsasii.helpers import (  # unedited, shared
    DEFAULT_HIST_SCALE_REFINE_FLAG,
    DEFAULT_HIST_SCALE_VAL,
    DEFAULT_SPF_GAMMA_MIN,
    DEFAULT_SPF_SIGMA_MIN,
    OUTPUT_NAMING,
    calculate_peak_widths,
    require_phase,
)
from powderline.gateways.gsasii.project import (  # unedited, shared
    add_powder_histogram_from_arrays,
)
from powderline.gateways.gsasii.extractors import (  # unedited, shared
    build_atom_name_mapping,
    build_phase_name_mapping,
    calculate_cell_esds_from_A_matrix,
    extract_refined_params_from_project,
    get_descriptive_param_name,
    parse_parameter_associations,
)
from powderline.gateways.gsasii.executors import (  # unedited, shared
    execute_rietveld_refinement,
)
from powderline.gateways.gsasii.setters import (  # unedited, shared
    set_chebyshev_background,
    set_fit_range_hist,
    set_hist_scale,
    set_refinement_cycles,
    set_single_peaks,
)


# --- verbatim from gateways/gsasii/helpers.py at 2810c82 ---

def unpack_refinement_parameter(param: list | None, param_name: str = "parameter") -> tuple[Any, Any, Any, Any]:
    """
    Unpack refinement parameter from [value, refine_flag, min, max] format.

    This utility reduces boilerplate for the common pattern of unpacking refinement
    parameters throughout the codebase. Use only in straightforward cases - complex
    conditional logic should inline the unpacking for clarity.

    Args:
        param: RefinementParameter list [value, refine_flag, min, max] or None
        param_name: Name of parameter for error messages (optional)

    Returns:
        Tuple of (value, refine_flag, min_val, max_val)
        If param is None, returns (None, None, None, None)

    Raises:
        ValueError: If param is neither list nor None

    Examples:
        >>> value, refine_flag, min_val, max_val = unpack_refinement_parameter([1.0, True, None, None])
        >>> value, refine_flag, min_val, max_val = unpack_refinement_parameter(None)
    """
    if param is None:
        return (None, None, None, None)
    elif isinstance(param, list):
        if len(param) != 4:
            raise ValueError(
                f"{param_name} must be a list of exactly 4 elements "
                f"[value, refine_flag, min, max], got {len(param)} elements"
            )
        return tuple(param)
    else:
        raise ValueError(
            f"{param_name} must be a list [value, refine_flag, min, max] or None, "
            f"got {type(param).__name__}"
        )


# --- verbatim from gateways/gsasii/helpers.py at 2810c82 ---

def has_active_refinement_parameter(atom_param: dict) -> bool:
    """
    Check if any refinement parameter has non-default values.

    Parameters:
    atom_param : dict
        Atom parameters dict with keys like 'x', 'y', 'z', 'occupancy', 'ADP', 'Uiso', 'Uaniso'

    Returns:
    bool
        True if any parameter should be set (not all [None, False, None, None])

    Notes:
    - Skips 'ADP' field (string, not refinement parameter)
    - Handles 'Uaniso' nested dict separately
    - Checks RefinementParameter lists for non-default values
    """
    for key, value in atom_param.items():
        if key == 'ADP':
            continue  # Skip ADP type string (not a refinement parameter)

        if key == 'Uaniso' and isinstance(value, dict):
            # Check nested Uaniso dict - any U value not [None, False, None, None]?
            if any(v is not None and v != [None, False, None, None] for v in value.values()):
                return True

        elif isinstance(value, list) and len(value) == 4:
            # Check RefinementParameter list [value, refine_flag, min, max]
            if value != [None, False, None, None]:
                return True

    return False


# --- verbatim from gateways/gsasii/project.py at 2810c82 ---

def add_phase_from_cif_dict(proj: Any, cif_data: Dict[str, Any], phase_name: str, histograms: Union[Sequence, str] = 'all') -> Any:
    """
    Add a phase to a GSAS-II project from an embedded structure dictionary.

    This function creates a phase from an in-memory structure data dictionary
    (file-less payload format) without reading from files. It creates a new phase entry
    in the project, sets up the phase's general data (space group, unit_cell) and
    atoms, and links the phase to specified histograms.

    Parameters
    ----------
    proj : GSASIIscriptable.G2Project
        The GSAS-II project object to which the phase will be added.
    cif_data : dict
        The structure dictionary containing phase structural information.
        From the payload this dict is stowed in payload['phases'][phase]['structure']
        (e.g., payload['phases']['LaB6']['structure']).
    phase_name : str
        Name for the new phase. Top level key for a phase in payload['phases'],
        or within payload['phases'][phase]['structure']['phase_name'] from payload.
    histograms : list or 'all', optional
        Which histograms to associate with this phase. Can be:
         - 'all' (default): link to all powder histograms in the project.
         - a list of histogram identifiers (names, objects, or indices).
         - None or empty list for no links.

    Returns
    -------
    phase_obj : GSASIIscriptable.G2Phase
        The newly created phase object (GSAS-II scriptable phase wrapper).

    Raises
    ------
    ValueError
        If the space group in cif_data is invalid or cannot be interpreted.

    Notes
    -----
    The function generates a new phase dictionary using GSASIIobj.SetNewPhase with
    the provided space group and unit_cell parameters. Atomic coordinates and
    occupancies from cif_data are inserted into the phase's atom list. Anisotropic
    displacement parameters, if present, are fully supported.

    By default, the phase will be linked to all existing powder histograms in the
    project. You can specify a subset of histograms by name, index, or object, or
    pass an empty list/None to link none.
    """
    pname = phase_name if phase_name else cif_data.get("name", "New Phase")
    # Ensure phase name is unique in project
    # TODO: handle phase name collisions more gracefully upstream in validation
    existing_names = [p.name for p in proj.phases()]
    pname = G2obj.MakeUniqueLabel(pname, existing_names)

    # Interpret space group symbol into SGData
    sg = cif_data.get("space_group", "P 1")
    err, SGData = G2spc.SpcGroup(sg)

    # Attempt normalization if initial interpretation failed
    normalized_sg = sg
    if err and sg:
        normalized_sg = G2spc.StandardizeSpcName(sg)
        if normalized_sg and normalized_sg != sg:
            err, SGData = G2spc.SpcGroup(normalized_sg)

    # Handle space group interpretation errors
    if err:
        error_msg = G2spc.SGErrors(err)
        raise ValueError(
            f"Space group '{sg}' could not be interpreted.\n"
            f"Error: {error_msg}\n"
            f"Attempted normalization: '{normalized_sg}'\n"
            f"\nCommon issues:\n"
            f"  • Missing spaces in Hermann-Mauguin notation (e.g., use 'R 3 m' not 'R3m')\n"
            f"  • Ambiguous rhombohedral symbols (R-3m normalizes to R-3c rhombohedral)\n"
            f"  • Check structure dict space_group entry matches the crystal structure\n"
        )

    # Create a new phase data structure (similar to reading from file):
    phase_data = G2obj.SetNewPhase(Name=pname, SGData=SGData)
    phase_data['General']['Name'] = pname

    # Document any space group normalization that occurred (silently - verbose mode not available here)
    # if normalized_sg != sg:
    #     print(f"ℹ️  Space group normalization: '{sg}' → '{normalized_sg}'")

    # Set unit cell parameters if available
    unit_cell = cif_data.get("unit_cell", {})
    if unit_cell:
        unit_cell_constants = [unit_cell.get("a"), unit_cell.get("b"), unit_cell.get("c"), unit_cell.get("alpha"), unit_cell.get("beta"), unit_cell.get("gamma")]
        if len(unit_cell_constants) == 6 and None not in unit_cell_constants:
            phase_data['General']['Cell'][1:7] = unit_cell_constants  # insert a,b,c,alpha,beta,gamma
            try:
                # Recalculate volume for consistency
                phase_data['General']['Cell'][7] = G2lat.calc_V(G2lat.cell2A(tuple(unit_cell_constants)))
            except Exception as e:
                raise RuntimeError(
                    f"Unit cell volume calculation failed for phase '{pname}'. "
                    f"Cell parameters: {unit_cell_constants}. "
                    f"Original error: {e}"
                ) from e

    # Add atoms from cif_data into phase_data['Atoms']
    # TODO: review handling of site_sym, multiplicity, and ADPS.
    # The current approach expects minimal info in cif_data, but we are now populating this with the complete info needed.
    # For example, cif_data['atoms'] stows info as shown below, where label is the key for each atom entry.:
        # atoms[label] = {
        #     "element": element,
        #     "Multiplicity": mult,
        #     "x": x, "y": y, "z": z,
        #     "occupancy": occ,
        #     "ADP": ADP,
        #     "Uiso": Uiso_val
        # }
        # # Add Uij if anisotropic and data available
        # if ADP == "Uani" and label in aniso_data:
        #     atoms[label]["Uij"] = aniso_data[label]
    # TODO: in phase 2, assembly of atom_record using GSAS-II helpers will done upstream using "parse_cif_to_dict" function.
    # This code will need to be update accordingly and use "get" with defaults for missing values. E.g., don't interpret site sym or mult.

    for label, atom in cif_data.get("atoms", {}).items():
        x = float(atom.get("x", 0.0))
        y = float(atom.get("y", 0.0))
        z = float(atom.get("z", 0.0))
        occ = float(atom.get("occupancy", 1.0))

        # Get ADP type for this atom (required in schema 0.22)
        adp_type = atom.get("ADP", None)
        if adp_type is None:
            raise ValueError(
                f"Atom '{label}' in phase '{pname}' is missing required 'ADP' field. "
                f"Schema 0.22 requires explicit ADP specification ('Uiso' or 'Uaniso') for all atoms."
            )

        # Prepare atom record list (based on GSAS-II internal format):
        atom_record = ["", "", "", 0.0, 0.0, 0.0, 1.0, "", 0.0, "I", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        atom_record[0] = label
        atom_record[1] = atom.get("element", label.rstrip('0123456789') or "X")
        atom_record[2] = ""  # refinement flags placeholder (left blank)
        atom_record[3] = x
        atom_record[4] = y
        atom_record[5] = z
        atom_record[6] = occ
        # Determine site symmetry and multiplicity for this atom in the given SG
        site_sym, mult = "", 1
        try:
            site_sym, mult = G2spc.SytSym([x, y, z], SGData)[0:2]
        except Exception:
            pass
        atom_record[7] = site_sym if site_sym is not None else ""
        atom_record[8] = mult if mult is not None else 1

        # Set displacement parameters (isotropic or anisotropic)
        if adp_type == "Uaniso":
            # Anisotropic displacement parameters
            atom_record[9] = "A"

            # Get Uaniso dict (supports both "Uij" old name and "Uaniso" new name)
            uaniso_dict = atom.get("Uaniso") or atom.get("Uij")

            if uaniso_dict:
                # Validate all 6 anisotropic values are present
                required_keys = ["U11", "U22", "U33", "U12", "U13", "U23"]
                missing_keys = [k for k in required_keys if k not in uaniso_dict or uaniso_dict[k] is None]
                if missing_keys:
                    raise ValueError(
                        f"Atom '{label}' marked as anisotropic (ADP='Uaniso') but missing required U values: {missing_keys}. "
                        f"All 6 anisotropic parameters (U11, U22, U33, U12, U13, U23) must be provided."
                    )

                # Populate U11, U22, U33, U12, U13, U23 (indices 11-16)
                atom_record[11] = float(uaniso_dict["U11"])
                atom_record[12] = float(uaniso_dict["U22"])
                atom_record[13] = float(uaniso_dict["U33"])
                atom_record[14] = float(uaniso_dict["U12"])
                atom_record[15] = float(uaniso_dict["U13"])
                atom_record[16] = float(uaniso_dict["U23"])

                # Set atom_record[10] to Uiso if provided, else calculate from diagonal elements
                if atom.get("Uiso") is not None:
                    atom_record[10] = float(atom["Uiso"])
                else:
                    # Calculate equivalent isotropic value from diagonal elements
                    atom_record[10] = (atom_record[11] + atom_record[12] + atom_record[13]) / 3.0
            else:
                raise ValueError(
                    f"Atom '{label}' marked as anisotropic (ADP='Uaniso') but no Uaniso dict found. "
                    f"Provide 'Uaniso' dict with U11, U22, U33, U12, U13, U23 values."
                )
        else:
            # Isotropic displacement parameters
            atom_record[9] = "I"
            uiso = atom.get("Uiso")
            if uiso is not None:
                atom_record[10] = float(uiso)
            else:
                atom_record[10] = 0.0
            atom_record[11:17] = [0.0]*6  # six anisotropic Uij (zeros since not used here)

        # Append a random unique ID for the atom
        atom_record.append(ran.randint(0, sys.maxsize))
        phase_data['Atoms'].append(atom_record)

    # Insert the new phase into project data structure
    if 'Phases' not in proj.data:
        proj.data['Phases'] = {'data': None}
    assert pname not in proj.data['Phases'], "Phase name collision despite uniqueness check"
    proj.data['Phases'][pname] = phase_data
    # Update the project tree name list for phases
    for entry in proj.names:
        if entry[0] == 'Phases':
            entry.append(pname); break
    else:
        proj.names.append(['Phases', pname])

    # Initialize phase general data (e.g. set default atom form factors):
    try:
        G2elem.SetupGeneral(phase_data, None)
    except ValueError as err:
        raise ValueError(f"Error in phase initialization: {err}")

    # Link phase to specified histograms
    hist_list = []
    if histograms == 'all':
        hist_list = [h.name for h in proj.histograms()]
    elif histograms:
        for h in histograms:
            if hasattr(h, 'name'):
                hist_list.append(h.name)
            elif isinstance(h, str):
                hist_list.append(h)
            elif isinstance(h, int):
                try:
                    hist_list.append(proj.histogram(h).name)
                except Exception:
                    continue
    for hist_name in hist_list:
        try:
            proj.link_histogram_phase(hist_name, pname)
        except Exception as e:
            print(f"Warning: could not link phase to histogram '{hist_name}': {e}")

    # Refresh internal IDs and return the new phase object
    proj.index_ids()
    proj.update_ids()
    return proj.phase(pname)


# --- verbatim from gateways/gsasii/project.py at 2810c82 ---

def add_phases_from_dict(proj: Any, hist: Any, phases_dict: dict, print_info: bool = False) -> None:
    """
    Add phases to the GSAS-II project from a phases dictionary.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project to add phases to.
    hist : GSAS-II histogram object
        The histogram to associate with the phases.
    phases_dict : dict
        Dictionary containing phase information. (e.g., payload['phases'])
    print_info : bool
        Whether to print information about the added phases.

    Returns:
    None
    """

    for phase_name, phase_info in phases_dict.items():
        structure_info = phase_info.get('structure', {})

        # Validation check (TODO: move this upstream in phase2)
        if phase_name != structure_info.get('phase_name'):
            raise NameError(f"Phase name mismatch for '{phase_name}'. Structure info name is '{structure_info.get('phase_name')}'.")

        # Prevent duplicate phase names - TODO: handle upstream in schema validation / recipe maker
        if phase_name in [p.name for p in proj.phases()]:
            raise ValueError(f"Phase '{phase_name}' already exists in project.")

        try:
            add_phase_from_cif_dict(proj=proj, cif_data=structure_info, phase_name=phase_name, histograms=[hist])
        except Exception as e:
            raise RuntimeError(f"Failed to add phase '{phase_name}': {e}") from e
    if print_info:
        print("\nPhases in proj: ", *[ph.name for ph in proj.phases()], sep='\n\t')


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

def set_single_peak_background(proj: Any, hist: Any, bkg_single_peaks_dict: dict, print_info: bool = False) -> None:
    """
    Set single peak background for histogram using pseudo-Voigt profiles.

    Single peaks are useful for modeling known impurity peaks or other non-background
    features that shouldn't be included in the main phase refinement. Each peak is
    described by a pseudo-Voigt profile (weighted sum of Gaussian and Lorentzian).

    Peak profile: I(2θ) = intensity * [η*L(2θ) + (1-η)*G(2θ)]
    where:
    - G(2θ) is Gaussian with width sigma
    - L(2θ) is Lorentzian with width gamma
    - η (eta) is mixing parameter (0=pure Gaussian, 1=pure Lorentzian)

    Args:
        proj: GSAS-II project object
        hist: Histogram object to set single peaks for
        bkg_single_peaks_dict: Dictionary with keys:
            - positions: List of [[2θ, refine, min, max], ...] for peak positions
            - intensities: List of [[I, refine, min, max], ...] for peak heights
            - pv_gaussian_sigma: List of [[σ, refine, min, max], ...] for Gaussian widths
            - pv_lorentzian_gamma: List of [[γ, refine, min, max], ...] for Lorentzian widths
            All lists must have same length (number of peaks)
        print_info: If True, print peak configuration to stdout

    Returns:
        None

    Raises:
        ValueError: If parameter lists have inconsistent lengths

    Examples:
        >>> # Two single peaks at 2θ=35.5° and 42.0°
        >>> single_peaks = {
        ...     'positions': [[35.5, False, None, None], [42.0, False, None, None]],
        ...     'intensities': [[50.0, True, None, None], [30.0, True, None, None]],
        ...     'pv_gaussian_sigma': [[0.1, False, None, None], [0.1, False, None, None]],
        ...     'pv_lorentzian_gamma': [[0.05, False, None, None], [0.05, False, None, None]]
        ... }
        >>> set_single_peak_background(proj, hist, single_peaks)
    """
    """
    Set the single peak background for a given histogram.

    Parameters:
    proj: object
        The GSAS-II project containing the histogram.
    hist : object
        The histogram to set the background for.
    bkg_single_peaks_dict : dict
        Dictionary containing single peak background parameters:
        - positions : list of [float, bool, float, float]. Indices 2 and 3 can be None.
            List of [position value, refine flag, min, max] for each peak.
            These are two-theta values. Q / d can be passed upstream and converted in validation.
        - intensities: list of [float, bool, float, float]. Indices 2 and 3 can be None.
            List of [intensity value, refine flag, min, max] for each peak.
        - pv_gaussian_sigma: list of [float, bool, float, float]. Indices 2 and 3 can be None.
            List of [gaussian sigma value, refine flag, min, max] for each peak.
        - pv_lorentzian_gamma: list of [float, bool, float, float]. Indices 2 and 3 can be None.
            List of [lorentzian gamma value, refine flag, min, max] for each peak.
    print_info : bool
        Whether to print information about the set background.
    Returns:
    None
    """
    # Parse the bkg_single_peaks_dict
    positions = bkg_single_peaks_dict.get('positions', [])
    intensities = bkg_single_peaks_dict.get('intensities', [])
    gaussian_sigmas = bkg_single_peaks_dict.get('pv_gaussian_sigma', [])
    lorentzian_gammas = bkg_single_peaks_dict.get('pv_lorentzian_gamma', [])

    # Default intensity, sigma, gamma values to use if value input is None
    # TODO: Phase 2 upstream recipe builder — these defaults will be set before calling this function
    # default_intensity = 1.0
    # default_sigma = 0.1
    # default_gamma = 0.1
    # default_refine_flag = False

    # Validate that all input lists have the same length
    if not (len(positions) == len(intensities) == len(gaussian_sigmas) == len(lorentzian_gammas)):
        raise ValueError("All input lists must have the same length.")

    num_peaks = len(positions) # only used for looping

    # Construct the single peak background list
    single_peak_bkg = []

    for i in range(num_peaks):
        pos, pos_refine, pos_min, pos_max = positions[i]
        inten, inten_refine, inten_min, inten_max = intensities[i]
        g_sigma, g_refine, g_sigma_min, g_sigma_max = gaussian_sigmas[i]
        l_gamma, l_refine, l_gamma_min, l_gamma_max = lorentzian_gammas[i]

        # TODO: move default behavior upstream
        # # Use default values if None provided
        # if inten is None:
        #     inten = default_intensity
        # if g_sigma is None:
        #     g_sigma = default_sigma
        # if l_gamma is None:
        #     l_gamma = default_gamma

        # # Use default refine flags if None provided
        # if pos_refine is None:
        #     pos_refine = default_refine_flag
        # if inten_refine is None:
        #     inten_refine = default_refine_flag
        # if g_refine is None:
        #     g_refine = default_refine_flag
        # if l_refine is None:
        #     l_refine = default_refine_flag

        # Min / max not currently available in GSAS-II single peak background definition
        # They can be stored upstream for reference, but not used downstream currently.
        # [placeholder for code to handle min/max if needed in future]

        # Append peak parameters to the single peak background list
        # Each peak is a list:
        # [pos, pos_refine, inten, inten_refine, g_sigma, g_refine, l_gamma, l_refine]
        if None in [pos, pos_refine,
                                inten, inten_refine,
                                g_sigma, g_refine,
                                l_gamma, l_refine]:
            continue

        # We only append if there are no Nones - this is a temp fix before phase2
        # TODO: update this behavior in the future when this list will only be passed with valid contents
        single_peak_bkg.append([pos, pos_refine,
                                inten, inten_refine,
                                g_sigma, g_refine,
                                l_gamma, l_refine])

    # Append the single peak background to the histogram's background list
    if proj.data[hist.name]['Background'][1]['peaksList'] is None:
        proj.data[hist.name]['Background'][1]['peaksList'] = []

    # Since single_peak_bkg is a list of peaks, we need to append each peak individually
    for peak in single_peak_bkg:
        proj.data[hist.name]['Background'][1]['peaksList'].append(peak)

    # Update the number of peaks
    proj.data[hist.name]['Background'][1]['nPeaks'] = len(proj.data[hist.name]['Background'][1]['peaksList'])

    # if print_info, then print details about the set background using the proj data, not the inputs
    if print_info:
        print(f"Single peak background set with {proj.data[hist.name]['Background'][1]['nPeaks']} peaks:")
        for peak in proj.data[hist.name]['Background'][1]['peaksList']:
            print(f"  Position: {peak[0]} (refine: {peak[1]}), Intensity: {peak[2]} (refine: {peak[3]}), "
                  f"Gaussian Sigma: {peak[4]} (refine: {peak[5]}), Lorentzian Gamma: {peak[6]} (refine: {peak[7]})")


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

@require_phase("Cannot set scale.")
def set_phase_scale(proj: Any, hist: Any, phase_name: str, scale_param: list, print_info: bool = False) -> None:
    """
    Set the phase scale factor in the GSAS-II project.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the phase.
    hist : GSAS-II histogram object
        The histogram associated with the phase.
    phase_name : str
        The name of the phase to set the scale for.
    scale_param : list
        List containing [value, refine_flag, min, max] for the scale factor.

    Returns:
    None
    """
    value, refine_flag, min_val, max_val = unpack_refinement_parameter(scale_param, "scale_param")

    # Set the scale factor value - defaults will be handled upstream in future in validation
    if value is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Scale'][0] = value

    # Set refinement flag
    if refine_flag is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Scale'][1] = refine_flag
    # Set min and max if provided - CURRENTLY NOT IMPLEMENETED. Unclear if allowed in GSAS-II.
    # if min_val is not None:
    #     phase.set_parameter_min('Scale', min_val)
    # if max_val is not None:
    #     phase.set_parameter_max('Scale', max_val)

    if print_info:
        print(f"Set scale for phase '{phase_name}' to {value} with refine_flag={refine_flag}")


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

@require_phase("Cannot set unit cell parameters.")
def set_phase_unit_cell(proj: Any, phase_name: str, unit_cell_dict: dict, print_info: bool = False) -> list[str]:
    """
    Set the unit cell parameters in the GSAS-II project for a given phase.

    Schema 0.26 semantics: each cell parameter refines iff it is present with
    refine_flag=true; absent or false means fixed. Because GSAS-II has a single
    whole-cell refine flag, per-parameter control is achieved by setting that
    flag when any symmetry degree-of-freedom (DOF) group is requested and
    returning "Hold" constraint variable names (e.g. '0::A2') for the DOF
    groups that were not requested. Symmetry-linked parameters (e.g. cubic
    a=b=c, or the coupled monoclinic {a, c, beta}) refine together if any
    member is requested — see powderline.constraints.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the phase.
    phase_name : str
        The name of the phase to set the unit cell for.
    unit_cell_dict : dict
        Dictionary containing unit cell parameters and their parameterization.
    print_info : bool
        Whether to print information about the set unit cell.

    Returns:
    list[str]
        GSAS-II variable names to Hold (empty when the whole cell refines or
        is entirely fixed). The caller applies them via proj.add_HoldConstr.
    """

    # Note: cell values are set verbatim; validating them against the phase's
    # crystal system is intentionally NOT done here (PowderLine is the engine,
    # not the arbiter of recipe correctness — see docs/known_issues.md, KI-01).

    # Note2: hist not required since unit cell is phase-specific, not histogram-specific.
    # The delineation of phase and histogram parameters is confusing in GSAS-II documentation.
    # This makes sticking with a project based approach more straightforward.

    phase_names = [p.name for p in proj.phases()]
    cell_changed = False
    cell_params = ['a', 'b', 'c', 'alpha', 'beta', 'gamma'] # must match schema, and match order in GSAS-II

    for i, param in enumerate(cell_params):
        if unit_cell_dict.get(param) is not None:
            value, refine_flag, min_val, max_val = unit_cell_dict[param]

            # Set the unit cell parameter value - a=1, b=2, c=3, alpha=4, beta=5, gamma=6
            if value is not None:
                proj.data['Phases'][phase_name]['General']['Cell'][i+1] = value
                cell_changed = True

            # Set min and max if provided - CURRENTLY NOT IMPLEMENTED (deferred;
            # GSAS-II's own min/max semantics differ from simple bounds).

    # Calculate and set volume from unit cell parameters only if cell was changed
    if cell_changed:
        cell = proj.data['Phases'][phase_name]['General']['Cell'][1:7] # get list of a, b, c, alpha, beta, gamma
        proj.data['Phases'][phase_name]['General']['Cell'][7] = G2lat.calc_V(G2lat.cell2A(cell)) # index 7 is volume

    # Translate per-parameter refine flags into the whole-cell flag + holds
    SGData = proj.data['Phases'][phase_name]['General']['SGData']
    phase_idx = phase_names.index(phase_name)
    cell_plan = cell_refinement_plan(SGData, unit_cell_dict, phase_idx)
    proj.data['Phases'][phase_name]['General']['Cell'][0] = cell_plan.refine_cell

    if print_info:
        curr_cell = proj.data['Phases'][phase_name]['General']['Cell']
        if cell_changed:
            cell_prms_dict = {"a": curr_cell[1],
                            "b": curr_cell[2],
                            "c": curr_cell[3],
                            "alpha": curr_cell[4],
                            "beta": curr_cell[5],
                            "gamma": curr_cell[6],
                            "volume": curr_cell[7]}

            print(f"Set unit cell parameters for phase '{phase_name}' to {[(key, value) for key, value in cell_prms_dict.items()]} with refine_flag={curr_cell[0]}")
        else:
            print(f"No unit cell parameters were changed for phase '{phase_name}'. Refine_flag={curr_cell[0]}")
        if cell_plan.holds:
            print(f"Holding fixed unit cell DOFs for phase '{phase_name}': {cell_plan.holds}")

    return cell_plan.holds


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

@require_phase("Cannot set size broadening parameters.")
def set_phase_size_broadening(proj: Any, hist: Any, phase_name: str, size_broadening_dict: dict, print_info: bool = False) -> None:
    """
    Set size broadening parameters in the GSAS-II project for a given phase.

    Supports model branching for isotropic, uniaxial, and ellipsoidal models.
    Currently only isotropic model is implemented - others raise NotImplementedError.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the phase.
    hist : GSAS-II histogram object
        The histogram associated with the phase.
    phase_name : str
        The name of the phase to set the size broadening for.
    size_broadening_dict : dict
        Dictionary containing size broadening parameters with 'model' key.

    Raises:
        NotImplementedError: For uniaxial or ellipsoidal models
        ValueError: For unknown model types or invalid parameters

    Returns:
    None
    """
    # Extract model type (default to isotropic for backward compatibility)
    model = size_broadening_dict.get('model', 'isotropic')

    # Validate model type
    valid_models = ['isotropic', 'uniaxial', 'ellipsoidal']
    if model not in valid_models:
        raise ValueError(
            f"Unknown size broadening model '{model}'. "
            f"Valid options: {valid_models}"
        )

    # Set model type in GSAS-II data structure
    proj.data['Phases'][phase_name]['Histograms'][hist.name]['Size'][0] = model

    # Branch on model type
    if model == 'isotropic':
        _set_isotropic_size_broadening(proj, hist, phase_name, size_broadening_dict, print_info)
    elif model == 'uniaxial':
        raise NotImplementedError(
            "Uniaxial size broadening model is not yet implemented. "
            "Support planned for future release. Use 'isotropic' model instead."
        )
    elif model == 'ellipsoidal':
        raise NotImplementedError(
            "Ellipsoidal size broadening model is not yet implemented. "
            "Support planned for future release. Use 'isotropic' model instead."
        )


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

def _set_isotropic_size_broadening(proj: Any, hist: Any, phase_name: str, size_broadening_dict: dict, print_info: bool = False) -> None:
    """
    Set isotropic size broadening parameters (internal helper).

    Refactored from original set_phase_size_broadening function.
    """

    # Example structure of proj.data['Phases'][phase_name]['Histograms'][hist.name]['Size']:
    # ['isotropic', # [0]: str for broadening type. 'isotropic', 'uniaxial', or 'ellipsoidal'
    # [1.0, 1.0, 1.0], # [1][0]: iso size or equatorial size, [1][1]: axial size (if uniaxial used, else ignored), [1][2]: LG_mix (eta parameter, 1 = Lorentzian, 0 = Gaussian)
    # [False, False, False], # boolean refinement flags. [2][0]: isotropic or uniaxial equatorial refine, [2][1]: uniaxial axial refine, [2][2]: LG_mix refine
    # [0, 0, 1], # hkl direction for uniaxial broadening. [3][0]: h, [3][1]: k, [3][2]: l
    # [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], # ellipsoidal sizes. S11, S22, S33, S12, S13, S23 for [4][0:6]
    # [False, False, False, False, False, False]]  # ellipsoidal size refine flags. S11, S22, S33, S12, S13, S23 for [5][0:6]
    ############################
    # Set isotropic size parameters
    ############################
    value, refine_flag, min_val, max_val = unpack_refinement_parameter(
        size_broadening_dict.get('isotropic_size'), "isotropic_size"
    )

    # Set the size parameter value
    if value is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Size'][1][0] = value

    # Set refinement flag
    if refine_flag is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Size'][2][0] = refine_flag

    # Set min and max if provided - CURRENTLY NOT IMPLEMENTED. Unclear if allowed in GSAS-II.

    if print_info:
        print(f"Set size broadening size parameters for phase '{phase_name}' to {value} with refine_flag={refine_flag}")

    ############################
    # Set size LG_eta parameter
    ############################
    try:
        value, refine_flag, min_val, max_val = unpack_refinement_parameter(
            size_broadening_dict.get('LG_eta'), "LG_eta"
        )
    except (ValueError, TypeError) as e:
        # Preserve original error message format
        raise ValueError(f"Size broadening 'LG_eta' parameter must be a list or None.\nType for phase {phase_name} is {type(size_broadening_dict['LG_eta'])}.") from e

    # Set the LG_eta parameter value
    if value is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Size'][1][2] = value

    # Set refinement flag
    if refine_flag is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Size'][2][2] = refine_flag

    # Set min and max if provided - CURRENTLY NOT IMPLEMENTED. Unclear if allowed in GSAS-II.

    if print_info:
        print(f"Set size broadening LG_eta parameters for phase '{phase_name}' to {value} with refine_flag={refine_flag}")


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

@require_phase("Cannot set strain broadening parameters.")
def set_phase_strain_broadening(proj: Any, hist: Any, phase_name: str, strain_broadening_dict: dict, print_info: bool = False) -> None:
    """
    Set strain broadening parameters in the GSAS-II project for a given phase.

    Supports model branching for isotropic, uniaxial, and generalized (Stephens) models.
    Currently only isotropic model is implemented - others raise NotImplementedError.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the phase.
    hist : GSAS-II histogram object
        The histogram associated with the phase.
    phase_name : str
        The name of the phase to set the strain broadening for.
    strain_broadening_dict : dict
        Dictionary containing strain broadening parameters with 'model' key.

    Raises:
        NotImplementedError: For uniaxial or generalized models
        ValueError: For unknown model types or invalid parameters

    Returns:
    None
    """
    # Extract model type (default to isotropic for backward compatibility)
    model = strain_broadening_dict.get('model', 'isotropic')

    # Validate model type
    valid_models = ['isotropic', 'uniaxial', 'generalized']
    if model not in valid_models:
        raise ValueError(
            f"Unknown strain broadening model '{model}'. "
            f"Valid options: {valid_models}"
        )

    # Set model type in GSAS-II data structure
    proj.data['Phases'][phase_name]['Histograms'][hist.name]['Mustrain'][0] = model

    # Branch on model type
    if model == 'isotropic':
        _set_isotropic_strain_broadening(proj, hist, phase_name, strain_broadening_dict, print_info)
    elif model == 'uniaxial':
        raise NotImplementedError(
            "Uniaxial strain broadening model is not yet implemented. "
            "Support planned for future release. Use 'isotropic' model instead."
        )
    elif model == 'generalized':
        raise NotImplementedError(
            "Generalized (Stephens) strain broadening model is not yet implemented. "
            "This model requires complex symmetry-dependent parameterization. "
            "Support planned for Phase 2. Use 'isotropic' model instead."
        )


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

def _set_isotropic_strain_broadening(proj: Any, hist: Any, phase_name: str, strain_broadening_dict: dict, print_info: bool = False) -> None:
    """
    Set isotropic strain broadening parameters (internal helper).

    Refactored from original set_phase_strain_broadening function.
    """
    ############################
    # Set isotropic strain parameters
    ############################
    value, refine_flag, min_val, max_val = unpack_refinement_parameter(
        strain_broadening_dict.get('isotropic_strain'), "isotropic_strain"
    )

    # Set the strain parameter value
    if value is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Mustrain'][1][0] = value

    # Set refinement flag
    if refine_flag is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Mustrain'][2][0] = refine_flag

    # Set min and max if provided - CURRENTLY NOT IMPLEMENTED. Unclear if allowed in GSAS-II.

    if print_info:
        print(f"Set strain broadening strain parameters for phase '{phase_name}' to {value} with refine_flag={refine_flag}")

    ############################
    # Set strain LG_eta parameter
    ############################
    try:
        value, refine_flag, min_val, max_val = unpack_refinement_parameter(
            strain_broadening_dict.get('LG_eta'), "LG_eta"
        )
    except (ValueError, TypeError) as e:
        # Preserve original error message format
        raise ValueError(f"Strain broadening dict key 'LG_eta' must hold a value that is a list or None.\nType for phase {phase_name} is {type(strain_broadening_dict['LG_eta'])}.") from e

    # Set the LG_eta parameter value
    if value is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Mustrain'][1][2] = value

    # Set refinement flag
    if refine_flag is not None:
        proj.data['Phases'][phase_name]['Histograms'][hist.name]['Mustrain'][2][2] = refine_flag

    # Set min and max if provided - CURRENTLY NOT IMPLEMENTED. Unclear if allowed in GSAS-II.

    if print_info:
        print(f"Set strain broadening LG_eta parameters for phase '{phase_name}' to {value} with refine_flag={refine_flag}")


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

@require_phase("Cannot set atom parameters.")
def set_phase_atom_parameters(proj: Any, phase_name: str, atom_parameters_dict: dict, print_info: bool = False) -> list[str]:
    """
    Set atom parameters for a phase in the GSAS-II project.

    This function parses atom-specific refinement parameters (coordinates, occupancy, displacement)
    and constructs GSAS-II's refine_flags string ('F', 'X', 'U' combinations).

    Schema 0.26 semantics: each coordinate (x/y/z) and each anisotropic Uij
    component refines iff present with refine_flag=true; absent or false means
    fixed. GSAS-II's 'X'/'U' flags are per-atom, so per-component control is
    achieved by returning "Hold" constraint variable names (e.g. '0::dAy:3',
    '0::AU22:4') for the site-symmetry DOF groups that were not requested —
    see powderline.constraints.
    Symmetry-linked components (e.g. x=y on an (x,x,z) site) refine together
    if any member is requested.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the phase.
    phase_name : str
        The name of the phase to set the atom parameters for.
    atom_parameters_dict : dict
        Dictionary containing atom parameters and their parameterization.
        E.g., payload['phases'][phase_name]['parameterization']['atoms']

        Expected structure:
        {
            'atom_label': {
                'x': [value, refine_flag, min, max],
                'y': [value, refine_flag, min, max],
                'z': [value, refine_flag, min, max],
                'occupancy': [value, refine_flag, min, max],
                'ADP': 'Uiso' or 'Uaniso',
                'Uiso': [value, refine_flag, min, max],  # if ADP='Uiso'
                'Uaniso': {  # if ADP='Uaniso'
                    'U11': [value, refine_flag, min, max],
                    'U22': [value, refine_flag, min, max],
                    ...
                }
            }
        }
    print_info : bool, optional
        If True, print detailed information about the atom parameters being set. Default is False.

    Returns:
    list[str]
        GSAS-II variable names to Hold (empty when every requested DOF group
        refines in full). The caller applies them via proj.add_HoldConstr.

    Notes:
    proj.data['Phases'][phase_name]['Atoms'] is a list of lists, where each sublist represents an atom and its parameters.
    Sublist structure in atom_param_index_mapping below.

    Example: proj.data['Phases'][phase_name]['Atoms'][0] from LaB6 example returns:
    ['La', 'La', '', 0.0, 0.0, 0.0, 1.0, 'm3m', 1, 'I', 0.00858, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 844046454091148456]

    GSAS-II refine_flags string format (index 2):
    - 'F': refine occupancy
    - 'X': refine all coordinates (x, y, z together)
    - 'U': refine displacement parameters (isotropic or anisotropic)
    - Combinations: 'FXU', 'XU', 'F', etc.
    """

    # Dictionary of atom parameter names to their indices in the atom list
    atom_param_index_mapping = {
        'label': 0, # atom label
        'element': 1, # element symbol
        'refine_flags': 2, # string of refine flags e.g., 'F', 'X', 'U' or any combination such as 'FXU'. F for occupancy, X for xyz coords, U for displacement parameters
        'x': 3, # fractional coordinate x value
        'y': 4, # fractional coordinate y value
        'z': 5, # fractional coordinate z value
        'occupancy': 6, # float, referred to as frac in GSAS-II syntax
        'site_symmetry': 7, # string representation of site symmetry
        'multiplicity': 8, # multiplicity of the site, integer
        'Utype': 9, # 'I' for isotropic, 'A' for anisotropic
        'Uiso': 10, # value if isotropic
        'U11': 11, # value if anisotropic
        'U22': 12, # value if anisotropic
        'U33': 13, # value if anisotropic
        'U12': 14, # value if anisotropic
        'U13': 15, # value if anisotropic
        'U23': 16 # value if anisotropic
        # Note: index 17 is a random unique ID for the atom, not a parameter to set
    }

    phase_names = [p.name for p in proj.phases()]
    phase_idx = phase_names.index(phase_name)
    SGData = proj.data['Phases'][phase_name]['General']['SGData']
    atom_labels_present = [x[0] for x in proj.data['Phases'][phase_name]['Atoms']]
    holds: list[str] = []

    for atom_label, atom_params in atom_parameters_dict.items():
        if atom_label not in atom_labels_present:
            print(f"Atom '{atom_label}' not found in phase '{phase_name}'. Skipping.")
            continue

        # Get the atom record
        atom_index = atom_labels_present.index(atom_label)
        atom_record = proj.data['Phases'][phase_name]['Atoms'][atom_index]

        # Get ADP type for this atom (required in schema 0.22)
        adp_type = atom_params.get('ADP')

        # Process coordinate parameters (x, y, z) — values only; refine flags
        # are translated by atom_refinement_plan below
        for coord in ['x', 'y', 'z']:
            if coord in atom_params and atom_params[coord] is not None:
                param_value = atom_params[coord]
                if param_value != [None, False, None, None]:
                    value, refine_flag, min_val, max_val = param_value

                    # Update coordinate value if provided
                    if value is not None:
                        atom_record[atom_param_index_mapping[coord]] = value

        # Process occupancy
        if 'occupancy' in atom_params and atom_params['occupancy'] is not None:
            param_value = atom_params['occupancy']
            if param_value != [None, False, None, None]:
                value, refine_flag, min_val, max_val = param_value

                # Update occupancy value if provided
                if value is not None:
                    atom_record[6] = value

        # Process displacement parameters based on ADP type
        if adp_type == 'Uiso':
            # Isotropic displacement parameters
            if 'Uiso' in atom_params and atom_params['Uiso'] is not None:
                param_value = atom_params['Uiso']
                if param_value != [None, False, None, None]:
                    value, refine_flag, min_val, max_val = param_value

                    # Update Uiso value if provided
                    if value is not None:
                        atom_record[9] = 'I'  # Set to isotropic
                        atom_record[10] = value

        elif adp_type == 'Uaniso':
            # Anisotropic displacement parameters
            if 'Uaniso' in atom_params and atom_params['Uaniso'] is not None:
                uaniso_dict = atom_params['Uaniso']

                # Track if any anisotropic values are being updated
                aniso_values_provided = []

                # Process each anisotropic parameter
                for u_key in ['U11', 'U22', 'U33', 'U12', 'U13', 'U23']:
                    if u_key in uaniso_dict and uaniso_dict[u_key] is not None:
                        param_value = uaniso_dict[u_key]
                        if param_value != [None, False, None, None]:
                            value, refine_flag, min_val, max_val = param_value

                            # Update anisotropic value if provided
                            if value is not None:
                                atom_record[9] = 'A'  # Set to anisotropic
                                atom_record[atom_param_index_mapping[u_key]] = value
                                aniso_values_provided.append(u_key)

                # Validate that if any anisotropic values are provided, all 6 must be provided
                if aniso_values_provided:
                    required_keys = ['U11', 'U22', 'U33', 'U12', 'U13', 'U23']
                    missing_keys = [k for k in required_keys if k not in aniso_values_provided]
                    if missing_keys:
                        raise ValueError(
                            f"Atom '{atom_label}' in phase '{phase_name}': "
                            f"When updating anisotropic displacement parameters, all 6 values (U11-U23) must be provided. "
                            f"Missing: {missing_keys}"
                        )

        # Translate per-parameter refine flags into GSAS-II's per-atom flag
        # string plus holds for the unrefined site-symmetry DOF groups.
        # xyz/SGData let the plan recompute a stale/blank site symmetry the
        # same way GSAS-II itself does (GetPhaseData's KeyError patch).
        atom_plan = atom_refinement_plan(
            atom_params,
            atom_record[7],
            phase_idx,
            atom_index,
            xyz=atom_record[3:6],
            SGData=SGData,
        )
        atom_record[2] = atom_plan.refine_flags
        holds.extend(atom_plan.holds)

        # Verbose logging
        if print_info and atom_plan.refine_flags:
            adp_display = adp_type if adp_type else "inherited"
            print(f"  Set refine flags for atom '{atom_label}' ({adp_display}): '{atom_plan.refine_flags}'")
            if atom_plan.holds:
                print(f"  Holding fixed DOFs for atom '{atom_label}': {atom_plan.holds}")

    return holds


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

def set_phase_parameterization(proj: Any, hist: Any, phases_dict: dict, print_info: bool = False) -> list[str]:
    """
    Set phase parameterization in the GSAS-II project from a phases dictionary.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the phases.
    hist : GSAS-II histogram object
        The histogram associated with the phases.
    phases_dict : dict
        Dictionary containing phase parameterization information. E.g., payload['phases']
    print_info : bool, optional
        If True, print information about the parameterization being set. Default is False.

    Returns:
    list[str]
        Accumulated GSAS-II "Hold" variable names from the unit-cell and atom
        setters (schema 0.26 per-parameter refine flags). The caller applies
        them once via proj.add_HoldConstr.
    """
    holds: list[str] = []

    for phase_name, phase_info in phases_dict.items():

        # In schema 0.21, phase_name is the dict key itself
        # (structure.phase_name should match, but we use the key for consistency)

        # Check that phase exists in project - this would indicate an error in phase addition earlier
        if phase_name not in [p.name for p in proj.phases()]:
            print(f"Phase '{phase_name}' not found in project. Skipping parameterization.")
            continue

        # Now that we have the phase name, we can add phase-specific parameterization here
        # This will call on a few helper functions to set scale, unit cell, peak broadening, atom parameters, etc.
        # Each of these helper functions will take the proj, phase name (rather than phase object), and relevant parameterization dict

        # First, get parameterization dict from phase_info (already phase-specific)
        param_dict = phase_info.get('parameterization', {})

        # Keys in param dict are 'scale', 'unit_cell', 'peak_broadening', and 'atoms' for now
        # It is not as simple as looping over keys since some have nested dicts and others hold values or lists

        # Set scale factor if provided
        scale_param = param_dict.get('scale', None)
        if scale_param is not None:
            if any(value is not None for value in scale_param):
                set_phase_scale(proj, hist, phase_name, scale_param, print_info=False)

        # Set unit cell parameters if provided. The key values can be None,
        # so here we cannot just check for unit_cell_dict not being None.
        # We need to check if any of hte keys in unit_cell_dict are not None.
        unit_cell_dict = param_dict.get('unit_cell', None)

        if unit_cell_dict is not None:
            if any(value is not None for value in unit_cell_dict.values()):
                holds.extend(set_phase_unit_cell(proj, phase_name, unit_cell_dict, print_info=False))

        # Set peak broadening if provided. Keys are "size_broadening" and "strain_broadening"
        # and each holds a dict with relevant parameters (size or strain) and the LG_eta
        # ("or {}" because model_dump emits an explicit None when the section is absent)
        peak_broadening_dict = param_dict.get('peak_broadening') or {}
        size_broadening_dict = peak_broadening_dict.get('size_broadening', None)
        strain_broadening_dict = peak_broadening_dict.get('strain_broadening', None)

        if size_broadening_dict is not None:
            if any(value is not None for value in size_broadening_dict.values()):
                set_phase_size_broadening(proj, hist, phase_name, size_broadening_dict, print_info=False)

        if strain_broadening_dict is not None:
            if any(value is not None for value in strain_broadening_dict.values()):
                set_phase_strain_broadening(proj, hist, phase_name, strain_broadening_dict, print_info=False)

        atoms_parameters_dict = param_dict.get('atoms', {})
        if atoms_parameters_dict and any(has_active_refinement_parameter(ap) for ap in atoms_parameters_dict.values()):
            holds.extend(set_phase_atom_parameters(proj, phase_name, atoms_parameters_dict, print_info=False))

        if print_info:
            print(f"Completed parameterization for phase '{phase_name}'.")

    if print_info:
        print("All phase parameterization complete. Phases in proj: ", *[p.name for p in proj.phases()], sep='\n\t')
        if holds:
            print("Hold constraints to apply (per-parameter refine flags): ", *holds, sep='\n\t')

    return holds


# --- verbatim from gateways/gsasii/setters.py at 2810c82 ---

def set_instrument_parameterization(proj: Any, hist: Any, instrument_param_dict: dict, print_info: bool = False) -> None:
    """
    Set instrument parameterization in the GSAS-II project from an instrument parameterization dictionary.

    Parameters:
    proj : GSAS-II project object
        The GSAS-II project containing the histogram.
    hist : GSAS-II histogram object
        The histogram associated with the instrument.
    instrument_param_dict : dict
        Dictionary containing instrument parameterization information. E.g., payload['instrument']
    print_info : bool, optional
        If True, print information about the parameterization being set. Default is False.

    Returns:
    None
    """

    # Note: GSAS-II's instrument parameters hold values in a list.
    # The indices for each parameter (e.g., 'Lam' for wavelength) are as follows:
    # [0] default/starting value from iprms, [1] = value, [2] = refinement flag

    # Wavelength
    wavelength_param = instrument_param_dict.get('wavelength', None)
    if wavelength_param is not None:
        value, refine_flag, min_val, max_val = unpack_refinement_parameter(wavelength_param, "wavelength")
        wavelength_changed = False # track if wavelength value changed

        # Set wavelength value
        curr_wavelength = proj.data[hist.name]['Instrument Parameters'][0]['Lam'][1]

        if value is not None and value != curr_wavelength:
            proj.data[hist.name]['Instrument Parameters'][0]['Lam'][1] = value
            wavelength_changed = True

        # Set refinement flag
        if refine_flag is not None:
            proj.data[hist.name]['Instrument Parameters'][0]['Lam'][2] = refine_flag

        if print_info:
            if wavelength_changed:
                print(f"Set instrument wavelength to {value} with refine_flag={refine_flag}")
            else:
                print(f"No change to instrument wavelength. Refine_flag={refine_flag}")

    # Additional instrument parameters (polarization, broadening, corrections) would be set similarly
    # Implementing those follows the same pattern as above

    # Polarization
    polarization_param = instrument_param_dict.get('polarization', None)
    if polarization_param is not None:
        value, refine_flag, min_val, max_val = unpack_refinement_parameter(polarization_param, "polarization")
        polarization_changed = False # track if polarization value changed

        curr_polarization = proj.data[hist.name]['Instrument Parameters'][0]['Polariz.'][1]

        # Set polarization value
        if value is not None and value != curr_polarization:
            proj.data[hist.name]['Instrument Parameters'][0]['Polariz.'][1] = value
            polarization_changed = True

        # Set refinement flag
        if refine_flag is not None:
            proj.data[hist.name]['Instrument Parameters'][0]['Polariz.'][2] = refine_flag

        if print_info:
            if polarization_changed:
                print(f"Set instrument polarization to {value} with refine_flag={refine_flag}")
            else:
                print(f"No change to instrument polarization. Refine_flag={refine_flag}")

    # Broadening parameters (U, V, W, X, Y, Z)
    broadening_dict = instrument_param_dict.get('broadening', {})
    for param_key in broadening_dict.keys():
        broadening_param = broadening_dict.get(param_key, None)
        if broadening_param is not None:
            value, refine_flag, min_val, max_val = unpack_refinement_parameter(broadening_param, f"broadening.{param_key}")

            broadening_changed = False

            # Set broadening parameter value
            if value is not None:
                proj.data[hist.name]['Instrument Parameters'][0][param_key][1] = value
                broadening_changed = True

            # Set refinement flag
            if refine_flag is not None:
                proj.data[hist.name]['Instrument Parameters'][0][param_key][2] = refine_flag

            if print_info:
                if broadening_changed:
                    # Print changed parameter
                    print(f"Set instrument broadening parameter '{param_key}' to {value} with refine_flag={refine_flag}")

                # Always print refinement flag even if value not changed
                else:
                    print(f"No change to instrument broadening parameter '{param_key}'. Refine_flag={refine_flag}")


    # Corrections parameters (zero_shift, axial_divergence, sample_height_displacement)
    corrections_dict = instrument_param_dict.get('corrections', {})
    for param_key in corrections_dict.keys():
        correction_param = corrections_dict.get(param_key, None)
        if correction_param is not None: # only loop over provided params

            if param_key == 'zero_shift':
                gsas_key = 'Zero'
            elif param_key == 'axial_divergence':
                gsas_key = 'SH/L'
            #elif param_key == 'sample_height_displacement': # irrelevant to area detector data, future work for Bragg-Brentano geometry
            #    gsas_key = 'SampHt'
            else:
                raise ValueError(f"Unknown instrument correction parameter key '{param_key}'.")

            value, refine_flag, min_val, max_val = correction_param
            correction_changed = False

            # Set correction parameter value
            if value is not None:
                proj.data[hist.name]['Instrument Parameters'][0][gsas_key][1] = value
                correction_changed = True

            # Set refinement flag
            if refine_flag is not None:
                proj.data[hist.name]['Instrument Parameters'][0][gsas_key][2] = refine_flag

            if print_info:
                if correction_changed:
                    # Print changed parameter
                    print(f"Set instrument correction parameter '{param_key}' to {value} with refine_flag={refine_flag}")

                # Always print refinement flag even if value not changed
                else:
                    print(f"No change to instrument correction parameter '{param_key}'. Refine_flag={refine_flag}")

    # Print instrument parameters set and flags if requested
    if print_info:
        # Check if 1st and 2nd indices are different for any instrument parameteters
        # If they are all the same, then no parameters were changed
        params_changed = False
        for param in proj.data[hist.name]['Instrument Parameters'][0].values():
            if param[0] != param[1]:
                params_changed = True
                break
        if not params_changed:
            print("No instrument parameters were changed from their default values.")

        # print refine flags for all instrument parameters
        print("Instrument parameter refinement flags:")
        for param_key, param in proj.data[hist.name]['Instrument Parameters'][0].items():
            print(f"  {param_key}: refine_flag={param[2]}")


# --- verbatim from gateways/gsasii/executors.py at 2810c82 ---

def execute_spf_refinement(
    proj: Any,
    hist: Any,
    recipe: RecipeModel,
    verbose: bool
) -> dict:
    """
    Execute single peak fitting refinement.

    This replaces the previous "peaks_only" strategy.

    Args:
        proj: GSAS-II project object
        hist: Histogram object
        recipe: Validated recipe model
        verbose: Print detailed progress

    Returns:
        Result dict with success, rwp, elapsed_time
    """
    controls = recipe.payload.refinement_controls
    spf_mode = "useIP" if controls.single_peak_fitting_mode.use_instrument_profile else "hold"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Executing Single Peak Fitting")
        print(f"  Mode: {spf_mode} ({'use instrument profile' if spf_mode == 'useIP' else 'refine peak widths'})")
        print(f"  Cycles: {controls.refinement_cycles}")
        print(f"{'='*60}\n")

    # Execute single peak fitting (cycles already set in step 10)
    peak_result = hist.refine_peaks(mode=spf_mode)

    # Extract Rwp from peak_result
    rwp_final = peak_result[3].get("Rwp") if len(peak_result) > 3 else None

    if rwp_final is None:
        return {
            'success': False,
            'rwp': None,
            'error': "Single peak fitting produced no Rwp — hist.refine_peaks() may have failed silently",
        }

    if verbose:
        print(f"Single peak fitting complete. Final Rwp: {rwp_final:.3f}%\n")

    return {
        'success': True,
        'rwp': rwp_final
    }


# --- verbatim from gateways/gsasii/executors.py at 2810c82 ---

SCHEMA_EXECUTORS = {
    'GSASII_Rietveld': execute_rietveld_refinement,
    'GSASII_SPF': execute_spf_refinement
}


# --- verbatim from gateways/gsasii/executors.py at 2810c82 ---

def run_refinement(recipe: RecipeModel, output_dir: Path, verbose: bool = False, method: str = 'server') -> dict:
    """
    Internal execution engine: run a GSAS-II refinement using already-loaded libraries.

    This function is the single execution entry point used by all paths:
    - ``gsas_server.py`` calls it directly after receiving an HTTP request
      (``method='server'``).
    - ``GSASClient._submit_via_subprocess()`` calls it in-process as the
      subprocess fallback (``method='subprocess'``). Despite the method name,
      no OS subprocess is spawned; see :meth:`GSASClient._submit_via_subprocess`.

    For programmatic use, prefer ``powderline.run()`` which validates the recipe,
    dispatches to the appropriate backend, and normalises all structured data
    fields to pandas DataFrames.

    Args:
        recipe: Validated RecipeModel instance (call ``powderline.validate()``
            first if starting from a raw dict).
        output_dir: Directory where output files are written (``dummy.gpx``,
            ``dummy.lst``, CSV reports, ``fit_profile.txt``). Must exist or be
            creatable.
        verbose: If True, print detailed progress to stdout.
        method: Execution method identifier string stored in the return dict.
            Callers set this to ``'server'`` or ``'subprocess'`` so downstream
            consumers know how the run was executed. Default ``'server'``.

    Returns:
        dict with keys:

        - ``success`` (bool)
        - ``run_id`` (str) — UUID4 string
        - ``rwp`` (float | None)
        - ``elapsed_time`` (float) — wall-clock seconds
        - ``method`` (str) — value of the ``method`` argument
        - ``output_files`` (list[str]) — paths of all files written to
          ``output_dir`` (informational)
        - ``fit_profile`` (dict) — column-oriented dict (JSON-serializable);
          normalised to ``pd.DataFrame`` by ``run()``
        - ``unit_cell_data`` (dict) — ``{phase: list-of-records}``
        - ``peak_list_data`` (dict) — ``{phase: list-of-records}``
        - ``refined_parameters`` (list) — list-of-records (9-column schema)
        - ``spf_peaks`` (dict) — column-oriented dict; populated for
          ``GSASII_SPF`` runs, empty dict otherwise
        - ``spf_convergence_diagnostics`` (dict) — column-oriented dict;
          populated when SPF peaks have convergence issues, empty dict otherwise
        - ``error`` (str) — present when ``success=False``
        - ``traceback`` (str) — present when an unhandled exception occurred

    Example::

        from powderline.kicker import run_refinement
        from powderline.schema import RecipeModel
        recipe = RecipeModel.model_validate(recipe_dict)
        result = run_refinement(recipe, Path('output'), verbose=True)
        print(f"Rwp: {result['rwp']:.3f}%  [run_id={result['run_id']}]")
    """
    # Assign a unique ID for this refinement run immediately, before any work begins,
    # so every exit path (success, GSAS-II failure, unexpected exception) can include
    # it in the returned dict and server logs.
    run_id = str(uuid.uuid4())
    import time
    start_time = time.time()

    try:
        # 1. Get xrd_data from validated model (file-less: no sample_name)
        xrd_data = recipe.payload.xrd_data

        # 2. Construct name and initialize project using standard naming
        gpx_path = output_dir / OUTPUT_NAMING.gpx_filename
        proj = G2.G2Project(newgpx=str(gpx_path))

        # 3. Instrument initialization - now a list of dicts [Iparm1, Iparm2]
        instrument_init = recipe.payload.instrument.initialization

        # 4. Add histogram - phase1B update: xrd_data is now a dict with arrays, instrument_init is list of dicts
        try:
            hist = add_powder_histogram_from_arrays(
                proj=proj,
                tth_array=xrd_data.tth,
                intensity_array=xrd_data.Itth,
                intensity_weights_array=xrd_data.Itth_weights,
                histogram_name=OUTPUT_NAMING.histogram_name,
                instrument_prm_dict=instrument_init[0],  # Use Iparm1 (first dict)
                comments=None,
                phases=None, # Optionally could link to phases if order of operations
            )
        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            elapsed_time = time.time() - start_time
            return {
                'success': False,
                'rwp': None,
                'run_id': run_id,
                'error': f"Failed to load XRD data or instrument parameters: {str(e)}",
                'traceback': tb,
                'elapsed_time': elapsed_time,
                'method': method
            }

        # After adding histogram, set hist scale and refine flag to defaults
        set_hist_scale(proj, hist, hist_scale_val=DEFAULT_HIST_SCALE_VAL, hist_scale_refine_flag=DEFAULT_HIST_SCALE_REFINE_FLAG, print_info=verbose)

        # 5. Set fit range
        fit_range = recipe.payload.fit_range if recipe.payload.fit_range else (None, None)
        if fit_range != (None, None):
            set_fit_range_hist(hist, fit_range, print_info=verbose)

        # 6. Set background - Chebyshev and single peaks
        if recipe.payload.background is not None:
            background_dict = recipe.payload.background.model_dump(mode='json')
            if recipe.payload.background.chebyshev is not None:
                chebyshev_dict = background_dict.get('chebyshev')
                set_chebyshev_background(proj, hist, chebyshev_dict, print_info=verbose)
            if recipe.payload.background.single_peaks is not None:
                bkg_single_peaks_dict = background_dict.get('single_peaks')
                set_single_peak_background(proj, hist, bkg_single_peaks_dict, print_info=verbose)

        # 6b. Set single peaks (Peak List mode for non-background peaks)
        if recipe.payload.single_peaks is not None:
            single_peaks_dict = recipe.payload.single_peaks.model_dump(mode='json')
            # Check if dict has any actual peak data before calling setter
            has_peak_data = False
            for key in ['positions', 'intensities', 'pv_gaussian_sigma_sq', 'pv_lorentzian_gamma']:
                if key in single_peaks_dict and single_peaks_dict[key] and len(single_peaks_dict[key]) > 0:
                    has_peak_data = True
                    break
            if has_peak_data:
                set_single_peaks(proj, hist, single_peaks_dict, print_info=verbose)

        # 7. Add phases (if present)
        if recipe.payload.phases is not None and len(recipe.payload.phases) > 0:
            phases_dict = recipe.payload.model_dump(mode='json')['phases']
            add_phases_from_dict(proj, hist, phases_dict, print_info=verbose)

            # 8. Parameterize phases
            holds = set_phase_parameterization(proj, hist, phases_dict, print_info=verbose)

            # 8b. Apply Hold constraints for per-parameter refine flags
            # (schema 0.26: fixed members of partially-refined DOF groups).
            # The isinstance guard keeps mocked-setter unit tests inert.
            if isinstance(holds, list) and holds:
                proj.add_HoldConstr(holds)
                if verbose:
                    print(f"Applied {len(holds)} Hold constraint(s): {holds}")

        # 9. Set instrument parameterization
        instrument_param_dict = recipe.payload.instrument.parameterization.model_dump(mode='json') if recipe.payload.instrument.parameterization else None
        if instrument_param_dict is not None:
            instrument_param_changes = False
            for key, value in instrument_param_dict.items():
                if isinstance(value, dict):
                    if any(v is not None for v in value.values()):
                        instrument_param_changes = True
                        break
                elif value is not None:
                    instrument_param_changes = True
                    break
            if instrument_param_changes:
                set_instrument_parameterization(proj, hist, instrument_param_dict, print_info=verbose)

        # 10. Get refinement controls and set refinement cycles
        controls = recipe.payload.refinement_controls

        # 10a. Validate single_peak_fitting_mode requirements (only for GSASII_SPF)
        if recipe.schema_name == 'GSASII_SPF':
            if controls.single_peak_fitting_mode is None:
                raise ValueError(
                    f"Schema '{recipe.schema_name}' requires "
                    f"'single_peak_fitting_mode' to be configured in refinement_controls."
                )

        num_cycles = controls.refinement_cycles
        if num_cycles != proj.get_Controls('cycles'):
            set_refinement_cycles(proj, num_cycles, print_info=verbose)

        # 11. Execute refinement using schema-based dispatch
        executor = SCHEMA_EXECUTORS.get(recipe.schema_name)
        if executor is None:
            raise ValueError(f"Unknown schema_name: {recipe.schema_name}. Valid options: {list(SCHEMA_EXECUTORS.keys())}")

        try:
            result = executor(proj, hist, recipe, verbose)

            # Check if executor succeeded
            if not result.get('success', False):
                elapsed_time = time.time() - start_time
                return {
                    'success': False,
                    'rwp': None,
                    'run_id': run_id,
                    'error': result.get(
                        'error',
                        f"Refinement executor '{recipe.schema_name}' returned success=False without an error message",
                    ),
                    'traceback': result.get('traceback'),
                    'elapsed_time': elapsed_time,
                    'method': method
                }

        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            elapsed_time = time.time() - start_time
            return {
                'success': False,
                'rwp': None,
                'run_id': run_id,
                'error': f"Refinement failed: {str(e)}",
                'traceback': tb,
                'elapsed_time': elapsed_time,
                'method': method
            }

        # Save GPX file after successful refinement
        proj.save(str(gpx_path))

        # 12. Extract final Rwp from result
        rwp = result.get('rwp')

        # --- Post-refinement data extraction ---
        # Each helper writes its own output files and returns JSON-serializable data.
        # run() normalises all dicts/lists to DataFrames for programmatic callers.

        fit_profile_data = _extract_fit_profile(hist, output_dir)

        spf_peaks_data, spf_diagnostics_data = _extract_spf_peak_report(
            proj, hist, recipe, output_dir, verbose
        )

        # Extract refined parameters (needed for unit cell ESDs in _extract_phase_reports)
        if verbose:
            print("\nExtracting refined parameters with ESDs...")

        # NOTE: .lst fallback (extract_refined_params_from_lst) is preserved for Phase 2
        # reference but its call site is intentionally disabled: .lst parsing is incomplete
        # (no atom/HAP params; different naming scheme) and obscured covariance failures.
        param_dict = extract_refined_params_from_project(proj, verbose=verbose)

        # Only raise if GSAS-II populated a varyList (i.e. parameters were actually varied)
        # but we still got nothing back. An empty param_dict with an empty varyList is
        # expected and correct for simulation mode (refinement_cycles=1, all params locked)
        # and for GSASII_SPF (no standard covariance matrix produced).
        cov_vary_list = (
            proj.data.get('Covariance', {}).get('data', {}).get('varyList', [])
        )
        if len(param_dict) == 0 and len(cov_vary_list) > 0:
            raise RuntimeError(
                "GSAS-II populated a varyList but refined parameter extraction returned "
                "empty results; covariance data may be corrupt. "
                "Ensure refinement completed successfully. "
                "Check the .lst file for GSAS-II error details."
            )

        # --- .lst fallback (disabled) ---
        # NOTE: preserved for Phase 2 reference — .lst parsing is incomplete
        # (no atom/HAP params; different naming scheme); covariance extraction is the
        # only supported path.
        # if len(param_dict) == 0:
        #     lst_file = output_dir / OUTPUT_NAMING.lst_filename
        #     if lst_file.exists():
        #         if verbose:
        #             print("  Covariance data not available, parsing .lst file...")
        #         param_dict = extract_refined_params_from_lst(lst_file)

        unit_cell_data, peak_list_data = _extract_phase_reports(
            proj, hist, recipe, param_dict, output_dir
        )

        refined_parameters_data = _extract_refined_parameters(
            param_dict, output_dir, proj, verbose
        )

        elapsed_time = time.time() - start_time
        return {
            'success': True,
            'run_id': run_id,
            'rwp': rwp,
            'elapsed_time': elapsed_time,
            'method': method,
            'output_files': [str(f) for f in output_dir.glob('*')],
            'fit_profile': fit_profile_data,
            'unit_cell_data': unit_cell_data,
            'peak_list_data': peak_list_data,
            'refined_parameters': refined_parameters_data,
            'spf_peaks': spf_peaks_data,
            'spf_convergence_diagnostics': spf_diagnostics_data,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        import logging
        logging.getLogger(__name__).error("run_refinement failed:\n%s", tb)
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'rwp': None,
            'run_id': run_id,
            'error': str(e),
            'traceback': tb,
            'elapsed_time': elapsed_time,
            'method': method
        }


# --- verbatim from gateways/gsasii/extractors.py at 2810c82 ---

def export_refined_parameters_csv(
    param_dict: Dict[str, Dict[str, Any]],
    output_file: Path,
    proj: Any = None,
    include_category: bool = True
) -> Optional[pd.DataFrame]:
    """
    Export refined parameters to CSV file and return the DataFrame.

    Args:
        param_dict: Dictionary from extract_refined_params_* functions
        output_file: Path to output CSV file
        proj: GSAS-II project object (optional, needed for phase/atom names)
        include_category: If True, add 'category' and 'descriptive_name' columns (default: True)

    Returns:
        DataFrame with the exported parameters, or None if param_dict is empty.

    Output CSV / DataFrame columns:
        - parameter_name: GSAS-II internal parameter name
        - descriptive_name: Human-readable parameter description (if include_category=True)
        - phase_name: Name of associated phase (None if not phase-specific)
        - phase_idx: Index of associated phase (None if not phase-specific)
        - atom_name: Label of associated atom (None if not atom-specific)
        - atom_idx: Index of associated atom (None if not atom-specific)
        - value: Refined value
        - esd: Estimated standard deviation (None for fixed parameters)
        - category: Parameter category (instrument, background, cell, etc.) if include_category=True
    """
    if len(param_dict) == 0:
        return None

    # Build phase and atom mappings if proj is available
    phase_mapping = {}
    atom_mapping = {}
    if proj is not None:
        phase_mapping = build_phase_name_mapping(proj)
        atom_mapping = build_atom_name_mapping(proj)

    rows = []
    for param_name, param_data in param_dict.items():
        # Parse parameter associations
        associations = parse_parameter_associations(param_name)
        phase_idx = associations["phase_idx"]
        atom_idx = associations["atom_idx"]

        # Lookup names from mappings
        phase_name = phase_mapping.get(phase_idx) if phase_idx is not None else None
        atom_name = atom_mapping.get(phase_idx, {}).get(atom_idx) if phase_idx is not None and atom_idx is not None else None

        row = {
            "parameter_name": param_name,
            "value": param_data["value"],
            "esd": param_data.get("esd"),
            "phase_name": phase_name,
            "phase_idx": phase_idx,
            "atom_name": atom_name,
            "atom_idx": atom_idx
        }

        if include_category:
            # Add descriptive name
            row["descriptive_name"] = get_descriptive_param_name(param_name)

            # Determine category from parameter name
            if '::A' in param_name and len(param_name.split('::')[1]) <= 2:
                # Reciprocal metric tensor (A0-A5)
                category = 'reciprocal_metric_tensor'
            elif '::' in param_name:
                # Other phase parameters (atoms, etc.)
                param_type = param_name.split('::')[1]
                if 'frac' in param_type:
                    category = 'atom_occupancy'
                elif 'Uiso' in param_type:
                    category = 'atom_displacement_isotropic'
                elif param_type.startswith('AU') and len(param_type) > 2:
                    category = 'atom_displacement_anisotropic'
                elif param_type.startswith('A') and param_type[1] in ['x', 'y', 'z']:
                    category = 'atom_position'
                else:
                    category = 'phase_other'
            elif param_name.startswith('cell:'):
                category = 'unit_cell'
            elif ':' in param_name:
                parts = param_name.split(':')
                if len(parts) >= 3:
                    param_part = parts[2]
                    if param_part.startswith('Back;'):
                        category = 'background'
                    elif param_part.startswith('BkPk'):
                        category = 'background_peak'
                    elif param_part == 'Scale':
                        category = 'scale'
                    elif param_part in ['U', 'V', 'W', 'X', 'Y', 'Z']:
                        category = 'instrument_broadening'
                    elif param_part in ['Lam', 'Zero', 'SH/L', 'Polariz']:
                        category = 'instrument'
                    else:
                        category = 'other'
                else:
                    category = 'other'
            else:
                category = 'other'

            row["category"] = category

        rows.append(row)

    # Sort by category, then parameter name
    if include_category:
        rows.sort(key=lambda x: (x["category"], x["parameter_name"]))
    else:
        rows.sort(key=lambda x: x["parameter_name"])

    # Reorder columns: parameter_name, descriptive_name, phase_name, phase_idx, atom_name, atom_idx, value, esd, category
    if include_category:
        df = pd.DataFrame(rows, columns=["parameter_name", "descriptive_name", "phase_name", "phase_idx", "atom_name", "atom_idx", "value", "esd", "category"])
    else:
        df = pd.DataFrame(rows, columns=["parameter_name", "phase_name", "phase_idx", "atom_name", "atom_idx", "value", "esd"])

    # Convert index columns to nullable integer type (Int64) to avoid scientific notation formatting
    df['phase_idx'] = df['phase_idx'].astype('Int64')
    df['atom_idx'] = df['atom_idx'].astype('Int64')

    # Format float columns independently: values need 6 decimal places, ESDs need 8
    df['value'] = df['value'].apply(lambda x: f"{x:.6e}" if pd.notna(x) else "")
    df['esd'] = df['esd'].apply(lambda x: f"{x:.8e}" if pd.notna(x) else "")

    df.to_csv(output_file, index=False, lineterminator="\n")
    return df


# --- verbatim from gateways/gsasii/extractors.py at 2810c82 ---

def _extract_fit_profile(hist: Any, output_dir: Path) -> dict:
    """Extract fit profile arrays from histogram and save fit_profile.txt.

    Args:
        hist: GSAS-II histogram object after refinement.
        output_dir: Directory to write ``fit_profile.txt``.

    Returns:
        dict: Column-oriented data (JSON-serializable) with keys ``two_theta``,
        ``y_obs``, ``y_weights``, ``y_calc``, ``y_diff``, ``y_bkg``, ``q_values``,
        ``d_spacings``.
    """
    two_theta = hist.getdata(datatype="X")
    q_values = hist.getdata(datatype="Q")
    d_spacings = hist.getdata(datatype="d")
    y_obs = hist.getdata(datatype="Yobs")
    y_weights = hist.getdata(datatype="Yweight")
    y_calc = hist.getdata(datatype="Ycalc")
    y_bkg = hist.getdata(datatype="Background")
    y_diff = hist.getdata(datatype="Residual")

    fit_profile_df = pd.DataFrame({
        "two_theta": two_theta,
        "y_obs": y_obs,
        "y_weights": y_weights,
        "y_calc": y_calc,
        "y_diff": y_diff,
        "y_bkg": y_bkg,
        "q_values": q_values,
        "d_spacings": d_spacings,
    })
    fit_profile_df.to_csv(
        output_dir / "fit_profile.txt", sep="\t", float_format="%.8f",
        header=True, index=False, lineterminator="\n"
    )
    return {col: fit_profile_df[col].tolist() for col in fit_profile_df.columns}


# --- verbatim from gateways/gsasii/extractors.py at 2810c82 ---

def _extract_spf_peak_report(
    proj: Any, hist: Any, recipe: RecipeModel, output_dir: Path, verbose: bool
) -> tuple[dict, dict]:
    """Extract single peak fitting results and save report files.

    Called only for ``GSASII_SPF`` runs where ``recipe.payload.single_peaks``
    is set. Returns two column-oriented dicts (JSON-serializable) that are
    normalised to DataFrames by ``run()``.

    Also writes:
    - ``single_peaks_report.txt`` — per-peak widths and convergence status
    - ``peak_convergence_diagnostics.txt`` — only when peaks have issues

    Args:
        proj: GSAS-II project object after refinement.
        hist: GSAS-II histogram object.
        recipe: Validated RecipeModel.
        output_dir: Directory to write report files.
        verbose: If True, print convergence warnings to stdout.

    Returns:
        ``(spf_peaks_data, spf_diagnostics_data)`` where each is a
        column-oriented dict, or ``({}, {})`` when single peaks are not used.
    """
    if recipe.payload.single_peaks is None:
        return {}, {}

    peak_list = proj.data.get(hist.name, {}).get('Peak List', {}).get('peaks', [])
    if not peak_list:
        return {}, {}

    peak_data = []
    convergence_diagnostics = []
    converged_count = 0
    aphysical_count = 0

    for i, peak in enumerate(peak_list):
        pos, pos_flag, intensity, int_flag, sig_sq, sig_sq_flag, gamma, gam_flag = peak

        # Determine convergence status
        status = "converged"
        if np.isnan(sig_sq) or np.isnan(gamma):
            status = "NaN_failed"
        elif sig_sq <= 0 and gamma < 0:
            status = "negative_sigma_sq_and_gamma_warning"
            aphysical_count += 1
        elif sig_sq <= 0:
            status = "zero_or_negative_sigma_sq_warning"
            aphysical_count += 1
        elif gamma < 0:
            status = "negative_gamma_warning"
            aphysical_count += 1
        else:
            converged_count += 1

        # Copy GSAS-II behavior for PV calc — use a small positive default for
        # aphysical (zero/negative) sigma_sq or gamma values
        sigma = np.sqrt(sig_sq) if sig_sq > 0 else DEFAULT_SPF_SIGMA_MIN
        # NOTE: GSAS-II gamma = FWHM_L (not HWHM), so we use gamma directly
        gamma_calc = gamma if gamma > 0 else DEFAULT_SPF_GAMMA_MIN

        # Calculate widths in degrees (GSAS-II stores in centidegrees)
        fwhm_g, fwhm_l, fwhm_pv, ib_g, ib_l, ib_pv, valid, warning_msg = calculate_peak_widths(
            sigma / 100, gamma_calc * 0.5 / 100
        )

        # Get GSAS-II verification (may fail for aphysical values)
        try:
            # GSAS-II uses gamma directly as FWHM_L (differs from scipy, TOPAS, literature)
            fwhm_gsas = G2pwd.getgamFW(gamma_calc / 100, sigma / 100)
        except Exception:
            fwhm_gsas = np.nan

        converged_bool = status == "converged"
        peak_data.append([
            pos, intensity, sigma, sig_sq, gamma_calc,
            fwhm_g, fwhm_l, fwhm_pv, ib_g, ib_l, ib_pv,
            fwhm_gsas, converged_bool, status
        ])

        if status != "converged":
            convergence_diagnostics.append({
                'peak_index': i,
                'position_2theta': pos,
                'final_sigma_sq': sig_sq,
                'final_gamma': gamma,
                'status': status,
                'notes': warning_msg if warning_msg else ""
            })

    # Build DataFrames
    spf_cols = [
        "position_2theta", "intensity", "sigma", "sigma_squared", "gamma",
        "fwhm_gaussian", "fwhm_lorentzian", "fwhm_pseudovoigt",
        "integral_breadth_gaussian", "integral_breadth_lorentzian",
        "integral_breadth_pseudovoigt", "fwhm_gsas_verification",
        "converged", "convergence_detail"
    ]
    spf_peaks_df = pd.DataFrame(peak_data, columns=spf_cols)
    diag_df = pd.DataFrame(convergence_diagnostics) if convergence_diagnostics else pd.DataFrame(
        columns=['peak_index', 'position_2theta', 'final_sigma_sq', 'final_gamma', 'status', 'notes']
    )

    # Write single_peaks_report.txt (custom mixed-type formatting)
    header_comment = (
        f"{converged_count} of {len(peak_list)} peaks converged; "
        f"{aphysical_count} peaks have aphysical values (warnings)"
    )
    header_cols = "\t".join(spf_cols)
    peak_array = np.array(peak_data, dtype=object)
    with open(output_dir / "single_peaks_report.txt", 'w', newline="\n") as f:
        f.write(f"# {header_comment}\n")
        f.write(f"{header_cols}\n")
        for row in peak_array:
            formatted_row = []
            for j, val in enumerate(row):
                col_name = spf_cols[j]
                if col_name == "converged":
                    formatted_row.append(str(val).lower())
                elif col_name == "convergence_detail":
                    formatted_row.append(str(val))
                else:
                    formatted_row.append("nan" if np.isnan(val) else f"{val:.8f}")
            f.write("\t".join(formatted_row) + "\n")

    # Write diagnostics file if there are issues
    if convergence_diagnostics:
        diag_df.to_csv(
            output_dir / "peak_convergence_diagnostics.txt", sep="\t",
            float_format="%.8f", header=True, index=False, lineterminator="\n"
        )
        if verbose:
            print(
                f"\nWarning: {len(convergence_diagnostics)} of {len(peak_list)} "
                f"peaks have convergence issues"
            )
            print(f"  See {output_dir / 'peak_convergence_diagnostics.txt'} for details\n")

    return (
        {col: spf_peaks_df[col].tolist() for col in spf_peaks_df.columns},
        {col: diag_df[col].tolist() for col in diag_df.columns},
    )


# --- verbatim from gateways/gsasii/extractors.py at 2810c82 ---

def _extract_phase_reports(
    proj: Any, hist: Any, recipe: RecipeModel, param_dict: dict, output_dir: Path
) -> tuple[dict, dict]:
    """Extract unit cell and peak list reports for all phases.

    Writes ``{phase}_unit_cell_report.csv`` and ``{phase}_peak_list_report.csv``
    for each phase.

    Args:
        proj: GSAS-II project object after refinement.
        hist: GSAS-II histogram object.
        recipe: Validated RecipeModel (used to check if phases exist).
        param_dict: Refined parameter dict from
            ``extract_refined_params_from_project()`` (needed for cell ESDs).
        output_dir: Directory to write CSV files.

    Returns:
        ``(unit_cell_data, peak_list_data)`` — each is a
        ``{phase_name: list-of-records}`` dict (JSON-serializable).
    """
    unit_cell_data: dict = {}
    peak_list_data: dict = {}

    if recipe.payload.phases is None or len(recipe.payload.phases) == 0:
        return unit_cell_data, peak_list_data

    phase_names = [p.name for p in proj.phases() if p is not None]

    for phase_idx, phase_name in enumerate(phase_names):
        try:
            unit_cell = proj.data['Phases'][phase_name]['General']['Cell'][1:8]
        except (KeyError, IndexError):
            continue

        cell_params = ["cell_a", "cell_b", "cell_c", "cell_alpha", "cell_beta", "cell_gamma", "cell_volume"]
        esds = calculate_cell_esds_from_A_matrix(phase_idx, proj, phase_name)

        # NOTE: .lst fallback for cell ESDs is disabled — preserved for Phase 2 reference.
        # calculate_cell_esds_from_A_matrix raises RuntimeError on failure rather than
        # returning [None]*7, so the fallback block below is unreachable.
        # if all(esd is None for esd in esds):
        #     cell_param_keys = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume']
        #     esds = [param_dict.get(f"cell:{k}", {}).get("esd") for k in cell_param_keys]

        unit_cell_df = pd.DataFrame({
            "parameter": cell_params,
            "value": unit_cell,
            "esd": esds,
        })
        unit_cell_df.to_csv(
            output_dir / f"{phase_name}_unit_cell_report.csv",
            float_format="%.8f", index=False, lineterminator="\n"
        )
        unit_cell_data[phase_name] = json.loads(unit_cell_df.to_json(orient='records'))

    # Peak list for each phase
    reflection_lists = proj.data.get(hist.name, {}).get('Reflection Lists', {})
    for phase_name in phase_names:
        if phase_name not in reflection_lists:
            continue
        phase_ref = reflection_lists[phase_name]
        if 'RefList' not in phase_ref:
            continue
        reflection_list = phase_ref['RefList']
        headers = [
            "h", "k", "l", "multiplicity", "d_spacing", "2theta",
            "sigma_squared", "gamma", "F_obs_squared", "F_calc_squared",
            "phase", "I_corr", "Prfo", "Trans", "ExtP"
        ]
        peak_list_df = pd.DataFrame(reflection_list, columns=headers)
        peak_list_df.to_csv(
            output_dir / f"{phase_name}_peak_list_report.csv",
            float_format="%.8f", index=False, lineterminator="\n"
        )
        peak_list_data[phase_name] = json.loads(peak_list_df.to_json(orient='records'))

    return unit_cell_data, peak_list_data


# --- verbatim from gateways/gsasii/extractors.py at 2810c82 ---

def _extract_refined_parameters(
    param_dict: dict, output_dir: Path, proj: Any, verbose: bool
) -> list:
    """Export refined parameters to CSV and return as list-of-records.

    Args:
        param_dict: From ``extract_refined_params_from_project()``.
        output_dir: Directory to write ``refined_parameters.csv``.
        proj: GSAS-II project object (for phase/atom name mappings).
        verbose: If True, print export status to stdout.

    Returns:
        List of dicts (records) with the 9-column schema. Returns ``[]``
        when ``param_dict`` is empty (simulation mode, SPF, etc.).
    """
    if not param_dict:
        if verbose:
            print("  No refined parameters found to export")
        return []

    try:
        refined_params_csv = output_dir / "refined_parameters.csv"
        refined_params_df = export_refined_parameters_csv(
            param_dict, refined_params_csv, proj=proj
        )
        if refined_params_df is not None:
            if verbose:
                print(f"  Exported {len(param_dict)} refined parameters to {refined_params_csv.name}")
            return refined_params_df.to_dict('records')
    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to export refined parameters CSV: {e}")
    return []
