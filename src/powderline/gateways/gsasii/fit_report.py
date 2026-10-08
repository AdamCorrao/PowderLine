"""GSAS-II gateway — standard fit statistics, engine details and engine warnings after a run (A23, A41, A45, A125).

Read after the refinement from what GSAS-II stored:

- the profile arrays (``hist.data['data'][1]``: 2theta, yobs, weights, ycalc)
  and the fit window (``Limits``; excluded regions are masked), for core
  :mod:`powderline.fitstats` (standard keys ``rwp``, ``r_exp``, ``gof``,
  ``chi2_red``);
- ``P``, the varied-parameter count GSAS-II used: Rietveld
  ``Covariance['data']['Rvals']['Nvars']`` (it includes parameters GSAS-II then
  dropped as singular, EB-33), Peak List fit ``len(Peak List['sigDict'])``;
- GSAS-II's own values (``Rwp``; ``GOF``, which for a Rietveld refinement is
  sqrt(reduced chi^2), EB-52; ``chisq``; ``Nobs``)
  under ``engine_details``, with ``parameters_requested`` (the recipe's
  independently refined parameters) next to ``parameters_varied`` (P);
- GSAS-II's refinement message (``Rvals['msg']``: parameters dropped as
  singular, SVD problems, limits), which ``G2strMain.Refine`` only prints
  (EB-32), as a structured warning.

Runtime layer (called with GSAS-II objects); imports no GSAS-II module itself.
"""

from __future__ import annotations

import re

import numpy as np

from powderline.exceptions import StructuredWarning
from powderline.fitstats import compute_fit_statistics
from powderline.schema_core import ChebyshevBackground, Phase, RefinableParameter
from powderline.symmetry import cell_tie_groups, coupling_groups


def parameters_requested(recipe) -> int:
    """Independently refined parameters the recipe asks for: one per symmetry tie group (A94), per flag otherwise."""
    count = 0

    def walk(obj) -> None:
        nonlocal count
        if isinstance(obj, RefinableParameter):
            count += obj.refine_flag
        elif isinstance(obj, ChebyshevBackground):
            count += obj.num_coefficients if obj.refine_flag else 0
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif hasattr(obj, "model_fields"):
            for name in type(obj).model_fields:
                walk(getattr(obj, name))
            if isinstance(obj, Phase):  # a refined tie group is one engine parameter
                tied = [(obj.unit_cell, g.members) for g in cell_tie_groups(obj.space_group).groups]
                for atom in obj.atoms.values():
                    ties = coupling_groups(obj.space_group, (atom.x.value, atom.y.value, atom.z.value))
                    tied += [(atom, g.members) for g in ties.xyz.groups]
                    if atom.Uaniso is not None:
                        tied += [(atom.Uaniso, g.members) for g in ties.uij.groups]
                for holder, members in tied:
                    refined = sum(getattr(holder, m).refine_flag for m in members)
                    count -= max(refined - 1, 0)

    walk(recipe.payload)
    return count


def _dropped(msg: str) -> int | None:
    match = re.search(r"(\d+)\s+Parameter\(s\) dropped", msg or "")
    return int(match.group(1)) if match else None


def fit_report(proj, hist, recipe, engine_rwp) -> dict:
    """``rwp``, ``r_exp``, ``gof``, ``chi2_red``, ``engine_details`` and ``warnings`` for the result dict.

    ``engine_rwp`` is the Rwp GSAS-II reported for the run (the executor's value).
    """
    x, yobs, weights, ycalc = (hist.data["data"][1][i] for i in range(4))
    lo, hi = hist.data["Limits"][1]
    xs = np.ma.getdata(x)
    mask = (xs >= lo) & (xs <= hi) & ~np.ma.getmaskarray(x)
    spf = recipe.schema_name == "gsasii.spf"
    if spf:
        rvals = {}
        n_params = len(hist.data.get("Peak List", {}).get("sigDict") or {})
    else:
        rvals = proj.data.get("Covariance", {}).get("data", {}).get("Rvals", {}) or {}
        n_params = int(rvals.get("Nvars", 0))
    stats = compute_fit_statistics(np.ma.getdata(yobs), np.ma.getdata(ycalc), np.ma.getdata(weights),
                                   n_params, mask=mask)
    message = " ".join(str(rvals.get("msg", "")).split())
    num = lambda v: None if v is None else float(v)  # plain floats: the result crosses JSON (server mode)
    details = {
        "engine": "GSAS-II",
        "rwp": num(engine_rwp),
        "gof": num(rvals.get("GOF")),
        "chi2": num(rvals.get("chisq")),
        "n_obs": None if rvals.get("Nobs") is None else int(rvals["Nobs"]),
        "n_points": stats.n_points,
        "parameters_requested": parameters_requested(recipe),
        "parameters_varied": n_params,
        "message": message or None,
    }
    warnings = []
    if message:
        dropped = _dropped(message)
        warnings.append(StructuredWarning(
            code="gsasii_refinement_message",
            message=(f"GSAS-II reported: {message}"
                     + (f" ({dropped} parameter(s) dropped as singular are still counted in "
                        "parameters_varied, GSAS-II's own count, EB-33)" if dropped else "")),
            field_path=None))
    return {"rwp": stats.rwp, "r_exp": stats.rexp, "gof": stats.gof, "chi2_red": stats.chi2_red,
            "engine_details": details, "warnings": warnings}
