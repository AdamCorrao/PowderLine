"""Regenerate _code_hash.json after kicker.py changes.

Usage: pixi run update-code-hash
"""

import hashlib
import json
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src" / "powderline"
KICKER_PATH = SRC_DIR / "kicker.py"
HASH_FILE = SRC_DIR / "_code_hash.json"


def main():
    current = json.loads(HASH_FILE.read_text())
    new_hash = hashlib.md5(KICKER_PATH.read_bytes()).hexdigest()

    if new_hash == current["kicker_hash"]:
        print(f"Hash unchanged: {new_hash}")
        return

    current["kicker_hash"] = new_hash
    HASH_FILE.write_text(json.dumps(current, indent=2) + "\n")
    print(f"Updated _code_hash.json: {new_hash}")


if __name__ == "__main__":
    main()
