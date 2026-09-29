"""Engine-free factory for the TOPAS gateway (entry point ``powderline.gateways:topas``).

- **Schema layer** — ``validate`` / ``capabilities``: builds the ``RecipeModel``
  then renders the INP in memory, exactly the check ``validate_only`` runs
  today (raises ``TopasTranslationError`` on untranslatable input).
- **Runtime layer** — ``run``: a thin wrapper over
  ``powderline.topas.engine.run_topas_recipe`` (behavior unchanged).

TOPAS has no Python engine dependency, so ``EngineNotAvailableError`` never
applies: without ``tc.exe`` the run degrades to generate-only, as today. TOPAS
also has no queryable runtime version: the version check uses the *configured*
version (``topas_version`` argument, else ``.powderline_config.yaml``
``topas.version``) and is skipped when none is configured (A53) — it checks what
the user configured, not the binary.
"""

from __future__ import annotations

from powderline.registry import Gateway, check_engine_version
from powderline.schema import RecipeModel

NAME = "topas"
#: Supported TOPAS versions (A41/A44).
ENGINE_VERSION_SPEC = ">=7,<8"
#: Workflows served (0.26.0 schema names until re/05 introduces ``topas.*``).
WORKFLOWS = ("GSASII_Rietveld", "GSASII_SPF")


def capabilities() -> dict:
    return {"name": NAME, "workflows": WORKFLOWS, "engine_version_spec": ENGINE_VERSION_SPEC}


def validate(recipe, *, verbose: bool = False):
    """Validate a recipe (dict or ``RecipeModel``); return the ``RecipeModel`` (A50, A59).

    ``verbose`` is part of the uniform gateway contract; this gateway prints
    nothing extra today.
    """
    from powderline.topas.writer import render_topas

    model = recipe if isinstance(recipe, RecipeModel) else RecipeModel.model_validate(recipe)
    render_topas(model.model_dump(), "topas")  # validates translatability; may raise
    return model


def configured_engine_version(topas_version=None):
    """The configured TOPAS version, or ``None`` when none is configured."""
    if topas_version is not None:
        return topas_version
    from powderline.topas.runner import _read_topas_config

    return _read_topas_config().get("version")


def run(recipe, output_dir, *, verbose: bool = False, validate_only: bool = False,
        topas_dir=None, topas_version=None,
        allow_unsupported_engine_version: bool = False) -> dict:
    """Run through ``run_topas_recipe``; see its docstring for the result dict.

    ``allow_unsupported_engine_version`` is the test-only bypass of the engine
    version check (A53); the check is skipped for ``validate_only``.
    """
    if not validate_only:
        version = configured_engine_version(topas_version)
        if version is not None:
            check_engine_version(gateway=NAME, engine="TOPAS", installed=version,
                                 spec=ENGINE_VERSION_SPEC,
                                 allow_unsupported=allow_unsupported_engine_version)
    from powderline.topas.engine import run_topas_recipe

    return run_topas_recipe(recipe, output_dir, verbose=verbose, validate_only=validate_only,
                            topas_dir=topas_dir, topas_version=topas_version)


def get_gateway() -> Gateway:
    return Gateway(name=NAME, capabilities=capabilities, validate=validate, run=run)
