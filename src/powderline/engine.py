"""Engine dispatch for ``powderline.run`` — GSAS-II, TOPAS, or easydiffraction by one argument.

``run(recipe, output_dir, engine="gsasii")`` preserves today's GSAS-II behaviour
exactly (D9); ``engine="topas"`` routes to the GSAS-II-free TOPAS adapter and
returns the identical result dict; ``engine="easydiffraction"`` routes to the
GSAS-II-free easydiffraction adapter. This module imports **no GSAS-II** at load
time — the GSAS-II backend is imported lazily only for ``engine="gsasii"``, so
``powderline.run(engine="topas")`` or ``powderline.run(engine="easydiffraction")``
works on machines without GSAS-II.

Dispatch goes through the gateway registry (``powderline.registry``, entry-point
group ``powderline.gateways``); each old ``engine=`` name maps 1:1 to a registry
name. Each gateway's runtime layer imports its engine lazily.
"""

from __future__ import annotations

from powderline import registry

_ENGINES = ("gsasii", "topas", "easydiffraction")
#: Old ``engine=`` names -> registry names (1:1; the ``engine=`` argument is removed in re/07).
_ENGINE_TO_GATEWAY = {"gsasii": "gsasii", "topas": "topas", "easydiffraction": "easydiffraction"}


def run(
    recipe,
    output_dir,
    *,
    engine: str = "gsasii",
    verbose: bool = False,
    validate_only: bool = False,
    execution_mode: str = "auto",
    topas_dir=None,
    topas_version=None,
) -> dict:
    """Run a refinement with the selected engine; return the standardized dict.

    Args:
        recipe: a ``RecipeModel`` or recipe dict.
        output_dir: directory for output files.
        engine: ``"gsasii"`` (default), ``"topas"``, or ``"easydiffraction"``.
            ``engine="easydiffraction"`` routes to the GSAS-II-free easydiffraction
            adapter; requires the optional ``easydiff`` pixi environment; ignores
            ``execution_mode`` and ``topas_*``.
        verbose / validate_only: as for the GSAS-II ``run``.
        execution_mode: GSAS-II backend mode (``auto``/``server``/``subprocess``);
            ignored by the TOPAS and easydiffraction engines.
        topas_dir / topas_version: TOPAS install selection; ignored by GSAS-II
            and easydiffraction.

    The return dict shape is identical across engines (see ``tests/test_api.py``).
    """
    if engine not in _ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {_ENGINES}")
    gateway = registry.get(_ENGINE_TO_GATEWAY[engine])
    if engine == "gsasii":
        return gateway.run(
            recipe,
            output_dir,
            verbose=verbose,
            validate_only=validate_only,
            execution_mode=execution_mode,
        )
    if engine == "topas":
        return gateway.run(
            recipe,
            output_dir,
            verbose=verbose,
            validate_only=validate_only,
            topas_dir=topas_dir,
            topas_version=topas_version,
        )
    return gateway.run(recipe, output_dir, verbose=verbose, validate_only=validate_only)
