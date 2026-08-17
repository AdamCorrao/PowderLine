"""
Example: Using PowderLine for schema-only validation (no GSAS-II required).

Demonstrates validating a PowderLine recipe JSON using RecipeModel without
importing GSAS-II or any heavy refinement machinery.

Typical use cases:
  - Recipe builder / editor tools that construct input.json files
  - Pre-flight validation before submitting a refinement job
  - CI checks that recipe files conform to the schema

REQUIREMENTS:
  - Environment built from docs/integration/pixi_schema_only.toml
  - PowderLine editable-installed (pixi install handles this automatically)

EDIT the path constants below before running.
"""

import json
from pathlib import Path
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Edit this path for your setup
# ---------------------------------------------------------------------------

# Absolute path to your local PowderLine clone (used only to locate example files)
POWDERLINE_REPO = Path("/path/to/PowderLine")

RECIPE_PATH = POWDERLINE_REPO / "examples" / "example_LaB6" / "input.json"

# ---------------------------------------------------------------------------
# Fail-fast: verify paths before doing any imports or real work.
# If you see this error, edit POWDERLINE_REPO above to point at your clone.
# ---------------------------------------------------------------------------

if not POWDERLINE_REPO.is_dir():
    raise SystemExit(
        f"POWDERLINE_REPO does not exist: {POWDERLINE_REPO}\n"
        f"Edit the POWDERLINE_REPO constant at the top of this script."
    )
if not RECIPE_PATH.is_file():
    raise SystemExit(
        f"Recipe file not found: {RECIPE_PATH}\n"
        f"Check that POWDERLINE_REPO points to a valid PowderLine clone "
        f"containing examples/example_LaB6/input.json."
    )

# ---------------------------------------------------------------------------
# Schema validation — this is the only import needed.
# GSAS-II is never touched in this path.
# ---------------------------------------------------------------------------

from powderline.schema import RecipeModel


def validate_recipe(path: Path) -> RecipeModel | None:
    """Load and validate a PowderLine recipe JSON. Returns the model or None on failure."""
    try:
        recipe_dict = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read {path}: {exc}")
        return None

    try:
        recipe = RecipeModel.model_validate(recipe_dict)
        return recipe
    except ValidationError as exc:
        print(f"Validation errors in {path.name}:")
        for error in exc.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            print(f"  [{error['type']}] {location}: {error['msg']}")
        return None


# ---------------------------------------------------------------------------
# Example: validate a single recipe and inspect the result
# ---------------------------------------------------------------------------

recipe = validate_recipe(RECIPE_PATH)

if recipe is not None:
    print(f"Valid recipe: {RECIPE_PATH.name}")
    print(f"  Schema:  {recipe.schema_name}  v{recipe.schema_version}")
    print(f"  Phases:  {list(recipe.payload.phases.keys())}")
    print(f"  Cycles:  {recipe.payload.refinement_controls.refinement_cycles}")

    # Access any payload field — all Pydantic types, fully typed
    wl = recipe.payload.instrument.parameterization.wavelength
    if wl is not None:
        print(f"  Wavelength: {wl[0]} Å  (refine={wl[1]})")

# ---------------------------------------------------------------------------
# Example: batch-validate multiple recipes
# ---------------------------------------------------------------------------

examples_dir = POWDERLINE_REPO / "examples"
recipe_files = sorted(examples_dir.glob("*/input.json"))

print(f"\nBatch validating {len(recipe_files)} example recipes:")
passed, failed = 0, 0
for recipe_path in recipe_files:
    result = validate_recipe(recipe_path)
    status = "OK" if result is not None else "FAIL"
    print(f"  [{status}] {recipe_path.parent.name}")
    if result is not None:
        passed += 1
    else:
        failed += 1

print(f"\n{passed} passed, {failed} failed.")
