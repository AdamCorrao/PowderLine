"""Test that GSAS-II gateway module code changes are accompanied by a version bump.

WHY THIS EXISTS:
    The GSAS-II server loads the GSAS-II gateway modules
    (src/powderline/gateways/gsasii/) once at startup and caches them in memory.
    If those modules change on disk, running servers still use the old code. The
    schema_version in each recipe is validated against EXPECTED_SCHEMA_VERSION
    in schema.py, so bumping the version forces stale servers to reject new
    recipes and restart with fresh code. Without this, tests pass locally
    (subprocess mode uses current code) but fail in production (server mode
    uses stale cached code).

WHEN THIS TEST FAILS:
    A GSAS-II gateway module was modified but _code_hash.json was not updated. Fix it:

    1. If output behavior changed (new files, format changes, column additions):
       bump EXPECTED_SCHEMA_VERSION in schema.py and update schema_version
       in all examples/*/input.json files.
    2. Run: pixi run update-code-hash
    3. Commit _code_hash.json alongside your changes.

    See docs/DEVELOPMENT.md 'Schema Version Discipline' for full details.
"""

import hashlib
import json
from pathlib import Path
import pytest


SRC_DIR = Path(__file__).parent.parent / "src" / "powderline"
GATEWAY_DIR = SRC_DIR / "gateways" / "gsasii"
HASH_FILE = SRC_DIR / "_code_hash.json"

# Expected modules in the manifest (independent tripwire)
EXPECTED_MODULES = (
    "helpers.py",
    "project.py",
    "setters.py",
    "executors.py",
    "extractors.py",
    "constraints.py",
)


def test_gateway_manifest_covers_expected_modules():
    """Verify the manifest covers exactly the expected GSAS-II gateway modules."""
    assert HASH_FILE.exists(), f"_code_hash.json not found at {HASH_FILE}"

    stored = json.loads(HASH_FILE.read_text(encoding="utf-8"))
    manifest = stored.get("gsasii_gateway_hashes", {})

    assert set(manifest.keys()) == set(EXPECTED_MODULES), (
        f"\nManifest modules mismatch!\n"
        f"  Expected: {set(EXPECTED_MODULES)}\n"
        f"  Got:      {set(manifest.keys())}\n"
    )


@pytest.mark.parametrize("module_name", EXPECTED_MODULES)
def test_gateway_hashes_match(module_name):
    """Verify each GSAS-II gateway module hash matches the stored hash.

    If this test fails, it means a gateway module was modified without updating
    the code hash. To fix:

    1. If output behavior changed: bump EXPECTED_SCHEMA_VERSION in schema.py
       and update schema_version in all examples/*/input.json files.
    2. Run: pixi run update-code-hash
    3. Commit the updated _code_hash.json along with your changes.
    """
    module_path = GATEWAY_DIR / module_name
    assert module_path.exists(), f"{module_name} not found at {module_path}"
    assert HASH_FILE.exists(), f"_code_hash.json not found at {HASH_FILE}"

    current_hash = hashlib.md5(module_path.read_bytes()).hexdigest()
    stored = json.loads(HASH_FILE.read_text(encoding="utf-8"))
    manifest = stored.get("gsasii_gateway_hashes", {})

    assert module_name in manifest, f"{module_name} not in manifest"
    stored_hash = manifest[module_name]

    assert current_hash == stored_hash, (
        f"\n{module_name} has changed but _code_hash.json was not updated!\n"
        f"  Stored hash:  {stored_hash}\n"
        f"  Current hash: {current_hash}\n\n"
        f"If gateway module output behavior changed, bump EXPECTED_SCHEMA_VERSION in schema.py.\n"
        f"Then run: pixi run update-code-hash\n"
        f"See docs/DEVELOPMENT.md 'Schema Version Discipline' for details."
    )
