"""Engine dispatch for ``powderline.run`` — GSAS-II or TOPAS by one argument.

``run(recipe, output_dir, engine="gsasii")`` preserves today's GSAS-II behaviour
exactly (D9); ``engine="topas"`` routes to the GSAS-II-free TOPAS adapter and
returns the identical result dict. This module imports **no GSAS-II** at load
time — the GSAS-II backend is imported lazily only for ``engine="gsasii"``, so
``powderline.run(engine="topas")`` works on machines without GSAS-II.
"""

from __future__ import annotations

_ENGINES = ("gsasii", "topas")


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
        engine: ``"gsasii"`` (default) or ``"topas"``.
        verbose / validate_only: as for the GSAS-II ``run``.
        execution_mode: GSAS-II backend mode (``auto``/``server``/``subprocess``);
            ignored by the TOPAS engine.
        topas_dir / topas_version: TOPAS install selection; ignored by GSAS-II.

    The return dict shape is identical across engines (see ``tests/test_api.py``).
    """
    if engine == "gsasii":
        from powderline.kicker import run as _gsas_run  # lazy: imports GSAS-II

        return _gsas_run(
            recipe,
            output_dir,
            verbose=verbose,
            validate_only=validate_only,
            execution_mode=execution_mode,
        )
    if engine == "topas":
        from powderline.topas.engine import run_topas_recipe  # GSAS-II-free

        return run_topas_recipe(
            recipe,
            output_dir,
            verbose=verbose,
            validate_only=validate_only,
            topas_dir=topas_dir,
            topas_version=topas_version,
        )
    raise ValueError(f"unknown engine {engine!r}; expected one of {_ENGINES}")
