"""Deprecated import path — use ``powderline.gateways.gsasii.client``.

Kept for backward compatibility during the multi-engine refactor (re/01);
removed in a later branch. Re-exports every top-level name of the moved module.
"""
from powderline.gateways.gsasii.client import (  # noqa: F401
    HOST,
    DEFAULT_PORT,
    PORT,
    PORT_FILE,
    GSASClient,
    run_simulation,
)
