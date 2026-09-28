"""Deprecated import path — use ``powderline.gateways.gsasii.server``.

Kept for backward compatibility during the multi-engine refactor (re/01);
removed in a later branch. Re-exports every top-level name of the moved module.
"""
from powderline.gateways.gsasii.server import (  # noqa: F401
    HOST,
    DEFAULT_PORT,
    PORT,
    PORT_FILE,
    PID_FILE,
    LOG_FILE,
    SimulationRequest,
    SimulationResponse,
    setup_logging,
    GSASServer,
    _pid_alive,
    is_server_running,
    stop_server,
    get_server_info,
    main,
)

if __name__ == "__main__":
    main()
