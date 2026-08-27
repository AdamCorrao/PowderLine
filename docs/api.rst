API Reference
=============

This page provides detailed documentation for PowderLine's Python API.

Schema Module
-------------

Pydantic models for recipe validation.

.. automodule:: powderline.schema
   :members:
   :undoc-members:
   :show-inheritance:

Kicker Module
-------------

Main refinement workflow and parameter setting functions.

Key Functions
~~~~~~~~~~~~~~

Template and Validation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: powderline.kicker.is_template_file

Output Generation
^^^^^^^^^^^^^^^^^^

Functions for exporting refinement results with uncertainty quantification.

.. autofunction:: powderline.kicker.export_refined_parameters_csv

.. autofunction:: powderline.kicker.calculate_cell_esds_from_A_matrix

.. autofunction:: powderline.kicker.extract_refined_params_from_project

.. note::

   These helpers document the GSAS-II engine's extraction path. ESDs (estimated
   standard deviations) are read from GSAS-II's covariance matrix
   (``proj.data['Covariance']``). Unit cell ESDs require conversion from the reciprocal
   metric tensor (A-matrix) to direct lattice parameters via ``calculate_cell_esds_from_A_matrix()``.

Post-Refinement Extraction Helpers (Private)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The four private ``_extract_*`` helpers encapsulate the post-refinement data
extraction and file-writing logic. They are not part of the public API but are
documented here for maintainers.

.. autofunction:: powderline.kicker._extract_fit_profile

.. autofunction:: powderline.kicker._extract_spf_peak_report

.. autofunction:: powderline.kicker._extract_phase_reports

.. autofunction:: powderline.kicker._extract_refined_parameters

Background Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: powderline.kicker.set_chebyshev_background

.. autofunction:: powderline.kicker.set_single_peak_background

Fit Range
^^^^^^^^^

.. autofunction:: powderline.kicker.set_fit_range_hist


Types
-----

Refinement Parameter Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Throughout the API, parameters use the format::

    [value, refine_flag, min, max]

Where:

- ``value`` (float): Current parameter value
- ``refine_flag`` (bool): True to refine, False to hold fixed
- ``min`` (float | None): Minimum bound (placeholder, not enforced yet)
- ``max`` (float | None): Maximum bound (placeholder, not enforced yet)

Example::

    "wavelength": [0.45236, false, null, null]  # Fixed at 0.45236 Å
    "scale": [1.0, true, null, null]            # Refined starting from 1.0

Phase vs Histogram Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Phase parameters**: Belong to crystal structure (unit cell, atoms)
- **Histogram parameters**: Belong to measurement (scale, broadening, background)

In multi-phase refinements, phase parameters are phase-specific,
while histogram parameters are shared across all phases (except scale factors,
which are phase-histogram pairs). Multi-phase refinement is fully supported.
