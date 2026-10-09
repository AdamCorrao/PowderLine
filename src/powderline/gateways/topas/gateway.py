"""Engine-free factory for the TOPAS gateway (entry point ``powderline.gateways:topas``).

- **Schema layer** — ``validate`` / ``capabilities``: a ``topas.*`` recipe
  validates against the native schemas (:mod:`.schema`, re/05); a 0.26.0
  ``GSASII_*`` recipe builds the ``RecipeModel`` and renders the INP in memory,
  exactly the check ``validate_only`` runs today (raises
  ``TopasTranslationError`` on untranslatable input), until re/07.
- **Runtime layer** — ``run``: a native recipe runs through
  :func:`.native_run.run_native`; a 0.26.0 recipe through
  ``powderline.gateways.topas.engine.run_topas_recipe`` (unchanged, A146).

TOPAS has no Python engine dependency, so ``EngineNotAvailableError`` never
applies: without ``tc.exe`` a run is generate-only. TOPAS also has no
queryable runtime version (``tc.exe`` carries no version metadata and prints
its banner only to its console window; devkit ``re05-topas6-compat.md`` C29),
so version checks use the **declared** version (``topas_version`` argument,
else ``.powderline_config.yaml`` ``topas.version``):

- native recipes (A144, A145): a run that will execute TOPAS needs a declared
  version, equal to the recipe's ``engine_version`` and within
  ``supported_engine_versions``; a missing declaration is an error;
- 0.26.0 recipes (A53, A146): ``ENGINE_VERSION_SPEC``, skipped when none is
  configured.
"""

from __future__ import annotations

from packaging.version import Version

from powderline.exceptions import EngineVersionError
from powderline.gateways.topas import schema
from powderline.registry import Gateway, check_engine_version
from powderline.schema import RecipeModel

NAME = "topas"
#: TOPAS versions of the 0.26.0 translation path (A41/A44; unchanged until re/07, A146).
#: Native ``topas.*`` recipes declare theirs in ``schema.SUPPORTED_ENGINE_VERSIONS`` (A144).
ENGINE_VERSION_SPEC = ">=7,<8"
#: Workflows served: the native schemas, and the 0.26.0 names until re/07 removes them.
WORKFLOWS = (*schema.SCHEMA_NAMES, "GSASII_Rietveld", "GSASII_SPF")


def capabilities() -> dict:
    return {"name": NAME, "workflows": WORKFLOWS, "engine_version_spec": ENGINE_VERSION_SPEC,
            **schema.DECLARATIONS}


def validate(recipe, *, verbose: bool = False):
    """Validate a recipe (dict or model); return the validated model (A50, A59).

    A ``topas.*`` recipe gives its native model (``TopasRietveldRecipe`` /
    ``TopasSpfRecipe``); a 0.26.0 one the legacy ``RecipeModel``. ``verbose`` is
    part of the uniform gateway contract; this gateway prints nothing extra.
    """
    if schema.is_native_recipe(recipe):
        return schema.validate_recipe(recipe)
    from powderline.gateways.topas.writer import render_topas

    model = recipe if isinstance(recipe, RecipeModel) else RecipeModel.model_validate(recipe)
    render_topas(model.model_dump(), "topas")  # validates translatability; may raise
    return model


def configured_engine_version(topas_version=None):
    """The configured TOPAS version, or ``None`` when none is configured."""
    if topas_version is not None:
        return topas_version
    from powderline.gateways.topas.runner import _read_topas_config

    return _read_topas_config().get("version")


def check_native_engine_version(model, topas_version=None, *, allow_unsupported: bool = False) -> None:
    """A145: the declared installed TOPAS version must exist and equal the recipe's ``engine_version``."""
    if allow_unsupported:
        return
    declared = configured_engine_version(topas_version)
    supported = schema.SUPPORTED_ENGINE_VERSIONS
    if declared is None:
        raise EngineVersionError(
            gateway=NAME, engine="TOPAS", installed="undeclared", supported=supported,
            message=(f"the TOPAS version to run is not declared: PowderLine cannot read it from tc.exe, so set "
                     f"topas_version (run argument) or topas.version in .powderline_config.yaml; this recipe is "
                     f"written for TOPAS {model.engine_version}"))
    check_engine_version(gateway=NAME, engine="TOPAS", installed=declared, spec=supported)
    if Version(str(declared)) != Version(model.engine_version):
        raise EngineVersionError(
            gateway=NAME, engine="TOPAS", installed=str(declared), supported=f"=={model.engine_version}",
            message=(f"the recipe is written for TOPAS {model.engine_version} (engine_version), but TOPAS "
                     f"{declared} is declared as installed"))


def run(recipe, output_dir, *, verbose: bool = False, validate_only: bool = False,
        topas_dir=None, topas_version=None,
        allow_unsupported_engine_version: bool = False) -> dict:
    """Run a recipe; see :func:`.native_run.run_native` (native) or ``run_topas_recipe`` (0.26.0).

    ``allow_unsupported_engine_version`` is the test-only bypass of the engine
    version check (A53); the check is skipped for ``validate_only``.
    """
    if schema.is_native_recipe(recipe):
        from powderline.gateways.topas import native_run

        model = schema.validate_recipe(recipe)  # the model validated from the user's recipe (A115)
        if validate_only:
            return native_run.run_native_validate(model)
        return native_run.run_native(
            model, output_dir, topas_dir=topas_dir, topas_version=topas_version,
            engine_version_check=lambda _tc: check_native_engine_version(
                model, topas_version, allow_unsupported=allow_unsupported_engine_version))
    if not validate_only:
        version = configured_engine_version(topas_version)
        if version is not None:
            check_engine_version(gateway=NAME, engine="TOPAS", installed=version,
                                 spec=ENGINE_VERSION_SPEC,
                                 allow_unsupported=allow_unsupported_engine_version)
    from powderline.gateways.topas.engine import run_topas_recipe

    return run_topas_recipe(recipe, output_dir, verbose=verbose, validate_only=validate_only,
                            topas_dir=topas_dir, topas_version=topas_version)


def get_gateway() -> Gateway:
    return Gateway(name=NAME, capabilities=capabilities, validate=validate, run=run)
