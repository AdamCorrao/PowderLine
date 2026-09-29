"""Engine-free factory for the GSAS-II gateway (entry point ``powderline.gateways:gsasii``).

- **Schema layer** — ``validate`` / ``capabilities``: engine-free; runs with
  GSAS-II not installed (``gateways.gsasii.validation`` is recipe-only).
- **Runtime layer** — ``run``: a thin wrapper over ``powderline.kicker.run``
  (behavior unchanged), imported inside the function. On a real run it first
  reads the installed GSAS-II version and enforces ``ENGINE_VERSION_SPEC``.

Never import ``powderline.kicker`` or the gateway's runtime modules (helpers,
project, setters, executors, extractors) at module top: they import GSAS-II.
Enforced by an import-block test.
"""

from __future__ import annotations

from powderline.exceptions import EngineNotAvailableError
from powderline.gateways.gsasii.validation import validate as _validate_recipe
from powderline.registry import Gateway, check_engine_version

NAME = "gsasii"
#: Supported GSAS-II runtime versions (A41/A44); pixi pins align in re/08.
ENGINE_VERSION_SPEC = "==5.7.9"
#: Workflows served (0.26.0 schema names until re/04 introduces ``gsasii.*``).
WORKFLOWS = ("GSASII_Rietveld", "GSASII_SPF")
#: pixi environment providing GSAS-II + how to install it (names final in re/08).
ENV = "default"
INSTALL_COMMAND = "pixi install"


def capabilities() -> dict:
    return {"name": NAME, "workflows": WORKFLOWS, "engine_version_spec": ENGINE_VERSION_SPEC}


def validate(recipe, *, verbose: bool = False):
    """Validate a recipe (dict or ``RecipeModel``); return the ``RecipeModel`` (A50, A59)."""
    return _validate_recipe(recipe, verbose=verbose)


def installed_engine_version() -> str:
    """The GSAS-II **runtime** version (``GSASII.__version__``), never distribution metadata.

    The pixi-built conda distribution reports ``gsas-ii 2.0.0`` (the fork recipe's
    version); the runtime reports ``5.7.9``. Fallback: ``GSASIIpath.GetVersionTag()``
    (``v5.7.9``) with the ``v`` stripped.
    """
    try:
        import GSASII
    except ImportError as exc:
        raise EngineNotAvailableError(gateway=NAME, dependency="GSAS-II", env=ENV,
                                      install_command=INSTALL_COMMAND) from exc
    version = getattr(GSASII, "__version__", None)
    if not version:
        from GSASII import GSASIIpath
        version = GSASIIpath.GetVersionTag().lstrip("v")
    return str(version)


def run(recipe, output_dir, *, verbose: bool = False, validate_only: bool = False,
        execution_mode: str = "auto", allow_unsupported_engine_version: bool = False) -> dict:
    """Run through ``powderline.kicker.run``; see its docstring for the result dict.

    ``allow_unsupported_engine_version`` is the test-only bypass of the engine
    version check (A53); the check is skipped for ``validate_only``.
    """
    if not validate_only:
        check_engine_version(gateway=NAME, engine="GSAS-II", installed=installed_engine_version(),
                             spec=ENGINE_VERSION_SPEC,
                             allow_unsupported=allow_unsupported_engine_version)
    from powderline.kicker import run as _gsas_run  # lazy: imports GSAS-II

    return _gsas_run(recipe, output_dir, verbose=verbose, validate_only=validate_only,
                     execution_mode=execution_mode)


def get_gateway() -> Gateway:
    return Gateway(name=NAME, capabilities=capabilities, validate=validate, run=run)
