"""
Example: Using PowderLine for schema validation and refinement execution.

Demonstrates the main import patterns for consuming projects:
  1. Schema validation via RecipeModel
  2. Programmatic refinement via powderline.run() (recommended entry point)
  3. Server-based refinement via GSASClient (preferred for repeated/batch runs)

REQUIREMENTS:
  - Environment built from docs/integration/pixi_full.toml
  - PowderLine editable-installed (pixi install handles this automatically)

EDIT the two path constants below before running.
"""

import json
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Edit these paths for your setup
# ---------------------------------------------------------------------------

# Absolute path to your local PowderLine clone
POWDERLINE_REPO = Path("/path/to/PowderLine")

# Directory where output files (*.gpx, *.lst, *.csv, fit_profile.txt) will be written
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="powderline_example_"))

# Recipe to use — any valid input.json from the PowderLine examples directory works here
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
# 1. Schema validation
#    Import only powderline.schema — GSAS-II is NOT imported at this point.
# ---------------------------------------------------------------------------

from powderline.schema import RecipeModel
from pydantic import ValidationError

recipe_dict = json.loads(RECIPE_PATH.read_text())

try:
    recipe = RecipeModel.model_validate(recipe_dict)
    print(f"Recipe validated successfully.")
    print(f"  Schema:   {recipe.schema_name}  v{recipe.schema_version}")
    print(f"  Phases:   {list(recipe.payload.phases.keys())}")
    print(f"  Cycles:   {recipe.payload.refinement_controls.refinement_cycles}")
except ValidationError as exc:
    print("Recipe validation failed:")
    for error in exc.errors():
        print(f"  {' -> '.join(str(loc) for loc in error['loc'])}: {error['msg']}")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 2. Programmatic refinement (powderline.run)
#    Recommended entry point: validates recipe, dispatches to server or
#    in-process fallback, and returns all outputs as pandas DataFrames.
#    GSAS-II is imported here — expect ~10 s on first call.
# ---------------------------------------------------------------------------

import powderline

print(f"\nRunning refinement via powderline.run() (output -> {OUTPUT_DIR})")
result = powderline.run(
    recipe=recipe,
    output_dir=OUTPUT_DIR,
)

if result["success"]:
    print(f"  Refinement complete.  Rwp = {result['rwp']:.3f}%")
    print(f"  Output files: {[Path(p).name for p in result['output_files']]}")
else:
    print(f"  Refinement failed: {result.get('error')}")

# ---------------------------------------------------------------------------
# 3. Server-based refinement (GSASClient)
#    Best for: repeated or batch refinements where GSAS-II startup cost matters.
#    GSASClient keeps GSAS-II loaded in a background process (~10 s savings per call).
#    The client auto-starts the server if it is not already running.
# ---------------------------------------------------------------------------

from powderline.gsas_client import GSASClient

server_output_dir = OUTPUT_DIR / "server_run"
server_output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nRunning server-based refinement (output -> {server_output_dir})")
client = GSASClient(fallback_to_subprocess=True)
server_result = client.submit_simulation(
    recipe_path=RECIPE_PATH,
    output_dir=server_output_dir,
    verbose=False,
    auto_start_server=True,
)

if server_result["success"]:
    print(f"  Refinement complete.  method={server_result['method']}")
    print(f"  Output files: {[Path(p).name for p in server_result['output_files']]}")
else:
    print(f"  Refinement failed: {server_result.get('error')}")

