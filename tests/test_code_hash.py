"""Test that kicker.py code changes are accompanied by a version bump.

WHY THIS EXISTS:
    The GSAS-II server loads kicker.py once at startup and caches it in memory.
    If kicker.py changes on disk, running servers still use the old code. The
    schema_version in each recipe is validated against EXPECTED_SCHEMA_VERSION
    in schema.py, so bumping the version forces stale servers to reject new
    recipes and restart with fresh code. Without this, tests pass locally
    (subprocess mode uses current code) but fail in production (server mode
    uses stale cached code).

WHEN THIS TEST FAILS:
    kicker.py was modified but _code_hash.json was not updated. Fix it:

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


SRC_DIR = Path(__file__).parent.parent / "src" / "powderline"
KICKER_PATH = SRC_DIR / "kicker.py"
HASH_FILE = SRC_DIR / "_code_hash.json"


def test_kicker_hash_matches():
    """Verify kicker.py hash matches the stored hash in _code_hash.json.

    If this test fails, it means kicker.py was modified without updating
    the code hash. To fix:

    1. If output behavior changed: bump EXPECTED_SCHEMA_VERSION in schema.py
       and update schema_version in all examples/*/input.json files.
    2. Run: pixi run update-code-hash
    3. Commit the updated _code_hash.json along with your changes.
    """
    assert KICKER_PATH.exists(), f"kicker.py not found at {KICKER_PATH}"
    assert HASH_FILE.exists(), f"_code_hash.json not found at {HASH_FILE}"

    current_hash = hashlib.md5(KICKER_PATH.read_bytes()).hexdigest()
    stored = json.loads(HASH_FILE.read_text())

    assert current_hash == stored["kicker_hash"], (
        f"\nkicker.py has changed but _code_hash.json was not updated!\n"
        f"  Stored hash:  {stored['kicker_hash']}\n"
        f"  Current hash: {current_hash}\n\n"
        f"If kicker.py output behavior changed, bump EXPECTED_SCHEMA_VERSION in schema.py.\n"
        f"Then run: pixi run update-code-hash\n"
        f"See docs/DEVELOPMENT.md 'Schema Version Discipline' for details."
    )
