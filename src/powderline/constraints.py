"""Deprecated import path — use ``powderline.gateways.gsasii.constraints``.

Kept for backward compatibility during the multi-engine refactor (re/01);
removed in a later branch. Re-exports every top-level name of the moved module.
"""
from powderline.gateways.gsasii.constraints import (  # noqa: F401
    CellPlan,
    AtomPlan,
    _CELL_PARAMS,
    _COORD_PARAMS,
    _DA_NAMES,
    _UIJ_PARAMS,
    _AU_NAMES,
    _is_requested,
    cell_dof_groups,
    cell_refinement_plan,
    _site_ids,
    _free_groups,
    atom_refinement_plan,
)
