"""GSAS-II persistent server for fast simulations.

This server keeps GSAS-II loaded in memory to eliminate startup overhead.
Listens on an HTTP port via FastAPI and processes simulation requests sequentially.

Security notes:
    - The server binds exclusively to ``127.0.0.1`` (loopback) on an ephemeral
      port (or ``POWDERLINE_SERVER_PORT`` if set). It is not accessible from
      remote hosts.
    - Every endpoint requires ``Authorization: Bearer <token>``. The token is
      generated at startup and stored, with the port and PID, in a per-user
      state directory (see :func:`state_dir`) that only the owning user can
      read. On a shared host other local users therefore cannot submit jobs
      (which would write files as the server's owner) or even probe the
      server. ``/health`` additionally proves to the client that the server
      holds the token, so a client never talks to another user's listener.
    - Set ``POWDERLINE_NO_SERVER=1`` (``true``/``yes``) to disable the server
      entirely: clients run in-process and ``gsas-server start`` refuses.
    - No rate limiting is applied.
    - The server resolves ``output_dir`` in ITS OWN filesystem view. A client
      on another node — or a server started inside a sandbox/container with a
      private /tmp — will never see the output files even though the run
      succeeds. ``GSASClient._server_output_visible`` guards clients against
      this and falls back to in-process execution.

Performance:
- First simulation: ~12s (server startup + GSAS-II import)
- Subsequent simulations: ~2-3s each (no startup overhead)
- 4-6x speedup for batch operations

Usage:
    # Start server (daemon mode)
    pixi run gsas-server start

    # Stop server
    pixi run gsas-server stop

    # Check status (also prints the log file location)
    pixi run gsas-server status

    # View logs
    pixi run gsas-server logs

    # Restart server
    pixi run gsas-server restart
"""

import json
import sys
import os
import signal
import socket
import stat
import time
import subprocess
import logging
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import atexit
from datetime import datetime

from pydantic import BaseModel

# Server configuration
HOST = "127.0.0.1"
PORT_ENV = 'POWDERLINE_SERVER_PORT'
NO_SERVER_ENV = 'POWDERLINE_NO_SERVER'
NONCE_HEADER = 'X-PowderLine-Nonce'
NO_SERVER_MESSAGE = (
    f"The GSAS-II server is disabled ({NO_SERVER_ENV} is set). "
    f"Run in-process instead (execution_mode='subprocess' / --no-server), "
    f"or unset {NO_SERVER_ENV}."
)


class ServerStateError(RuntimeError):
    """The per-user server state directory is unsafe or unusable."""


def server_disabled() -> bool:
    """True if ``POWDERLINE_NO_SERVER`` is set to a truthy value (1/true/yes)."""
    return os.environ.get(NO_SERVER_ENV, '').strip().lower() in ('1', 'true', 'yes')


def _private_parent(path: str) -> bool:
    """True if ``path`` is a directory owned by us that others cannot write to."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    return (stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid()
            and not st.st_mode & 0o022)


def _state_dir_path() -> Path:
    """Location of the per-user state directory (not created or checked).

    POSIX: ``$XDG_RUNTIME_DIR/powderline`` when that directory is ours, else
    ``<tempdir>/powderline-<uid>`` — host-local, never the (possibly
    NFS-shared) home directory, because the server is per-host.
    Windows: ``%LOCALAPPDATA%\\powderline`` (protected by the profile ACL).
    """
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        return Path(base) / 'powderline'
    runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
    if runtime_dir and _private_parent(runtime_dir):
        return Path(runtime_dir) / 'powderline'
    return Path(tempfile.gettempdir()) / f'powderline-{os.getuid()}'


def state_dir() -> Path:
    """Return the per-user server state directory, creating it if needed.

    Holds the PID, endpoint (port + token) and log files. On POSIX it is
    created with mode 0700 and refused if it is a symlink, not owned by the
    current user, or accessible to group/other — a directory pre-created by
    another user in a shared /tmp must never be trusted.

    Raises:
        ServerStateError: If the directory exists but is unsafe to use.
    """
    path = _state_dir_path()
    try:
        if os.name == 'nt':
            path.mkdir(parents=True, exist_ok=True)
            return path
        try:
            os.mkdir(path, 0o700)
            os.chmod(path, 0o700)  # umask may have stripped owner bits
        except FileExistsError:
            pass
        st = os.lstat(path)
    except OSError as e:
        raise ServerStateError(
            f"Cannot create GSAS-II server state directory {path}: {e}") from e
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        problem = "is not a directory (or is a symlink)"
    elif st.st_uid != os.getuid():
        problem = f"is owned by uid {st.st_uid}, not by you (uid {os.getuid()})"
    elif st.st_mode & 0o077:
        problem = f"is accessible to other users (mode {stat.S_IMODE(st.st_mode):o})"
    else:
        return path
    raise ServerStateError(
        f"Refusing to use GSAS-II server state directory {path}: it {problem}. "
        f"Remove it so it can be recreated privately."
    )


def _pid_file() -> Path:
    return state_dir() / 'server.pid'


def _endpoint_file() -> Path:
    return state_dir() / 'server.json'


def log_file() -> Path:
    """Path of this user's server log file."""
    return state_dir() / 'server.log'


def _write_private(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path``, readable by the owner only."""
    tmp = path.with_name(path.name + '.tmp')
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def read_endpoint() -> Optional[Dict[str, Any]]:
    """Return this user's server endpoint ``{'pid', 'port', 'token'}``, or None.

    Raises:
        ServerStateError: If the state directory is unsafe to use.
    """
    try:
        data = json.loads(_endpoint_file().read_text())
        return {'pid': int(data['pid']), 'port': int(data['port']),
                'token': str(data['token'])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def health_proof(token: str, nonce: str) -> str:
    """HMAC proving knowledge of ``token`` for a client-chosen ``nonce``."""
    return hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()


def _requested_port() -> int:
    """Port from ``POWDERLINE_SERVER_PORT``, else 0 (an ephemeral port)."""
    value = os.environ.get(PORT_ENV, '').strip()
    return int(value) if value else 0


def _bind_listener(port: int) -> socket.socket:
    """Bind and listen on ``HOST:port`` (0 = ephemeral)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        # Windows: forbid other sockets from binding the same port.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    elif port:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((HOST, port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    return sock


def _claim_pid_file() -> bool:
    """Atomically create the PID file; False if a live server already holds it.

    This is the start lock: with ephemeral ports, two concurrent starts would
    otherwise both succeed and orphan one server.
    """
    pid_path = _pid_file()
    for _ in range(2):
        try:
            fd = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if is_server_running():
                return False
            continue  # stale PID file was removed; try again
        with os.fdopen(fd, 'w') as f:
            f.write(str(os.getpid()))
        return True
    return False


def _release_state_files() -> None:
    """Remove the PID and endpoint files if they belong to this process."""
    try:
        pid_path = _pid_file()
        if int(pid_path.read_text()) != os.getpid():
            return
        _endpoint_file().unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
    except (OSError, ValueError, ServerStateError):
        pass


# --- Pydantic request/response models ---

class SimulationRequest(BaseModel):
    recipe_data: dict
    output_dir: str
    verbose: bool = False


class SimulationResponse(BaseModel):
    success: bool
    run_id: Optional[str] = None
    rwp: Optional[float] = None
    elapsed_time: Optional[float] = None
    method: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    output_files: Optional[List[str]] = None
    fit_profile: Optional[dict] = None
    unit_cell_data: Optional[dict] = None
    peak_list_data: Optional[dict] = None
    refined_parameters: Optional[list] = None
    spf_peaks: Optional[dict] = None
    spf_convergence_diagnostics: Optional[dict] = None


# --- Logging ---

def setup_logging(log_file: Path) -> logging.Logger:
    """Set up logging to both console and file."""
    logger = logging.getLogger('powderline.server')
    logger.setLevel(logging.DEBUG)

    # File handler (always log everything)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    # Console handler (info level for cleaner output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter with timestamps
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# --- Server class ---

class GSASServer:
    """Persistent GSAS-II server for fast simulations."""

    def __init__(self, logger: logging.Logger = None):
        """Initialize server (GSAS-II import happens here).

        Args:
            logger: Optional logger instance. If not provided, one will be created.
        """
        self.running = False
        self.request_count = 0
        self.start_time = None
        # Per-start credential; clients read it from the owner-only state dir.
        self.token = secrets.token_urlsafe(32)

        # Set up logging
        self.logger = logger or setup_logging(log_file())

        # Import GSAS-II (this is the slow part we want to do once)
        self.logger.info("Loading GSAS-II libraries...")
        import_start = time.time()

        # Set PYTHONPATH to include src directory
        src_dir = Path(__file__).parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            # Import kicker module which has all the GSAS-II integration
            from powderline import kicker
            self.kicker = kicker
            import_time = time.time() - import_start
            self.logger.info(f"GSAS-II loaded in {import_time:.1f}s")
        except ImportError as e:
            self.logger.error(f"Failed to import GSAS-II: {e}")
            sys.exit(1)

    def create_app(self):
        """Build and return the FastAPI application.

        Every route — including ``/health`` — requires
        ``Authorization: Bearer <self.token>``; anything else gets 401.
        """
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        # No interactive docs/schema routes: nothing is served unauthenticated.
        app = FastAPI(title="PowderLine GSAS-II Server",
                      docs_url=None, redoc_url=None, openapi_url=None)
        server = self  # capture for closures
        expected_auth = f"Bearer {self.token}".encode()

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            # Starlette decodes headers as latin-1, so this round-trips exactly.
            provided = request.headers.get('authorization', '').encode('latin-1')
            if not secrets.compare_digest(provided, expected_auth):
                return JSONResponse(status_code=401, content={'detail': 'Unauthorized'},
                                    headers={'WWW-Authenticate': 'Bearer'})
            return await call_next(request)

        @app.post("/simulate", response_model=SimulationResponse)
        def simulate(req: SimulationRequest):
            server.request_count += 1
            schema_name = req.recipe_data.get('schema_name', 'unknown') if isinstance(req.recipe_data, dict) else 'unknown'
            server.logger.info(f"Request #{server.request_count}: schema={schema_name}")

            result = server._run_simulation(req.model_dump())

            elapsed = result.get('elapsed_time', 0)
            run_id = result.get('run_id', 'n/a')
            if result.get('success'):
                server.logger.info(f"Request #{server.request_count} (run_id={run_id}) completed in {elapsed:.1f}s")
            else:
                server.logger.warning(
                    f"Request #{server.request_count} (run_id={run_id}) failed: {result.get('error', 'Unknown error')}"
                )
                if result.get('traceback'):
                    server.logger.debug(
                        f"Request #{server.request_count} traceback:\n{result['traceback']}"
                    )
            return SimulationResponse(**result)

        @app.get("/health")
        def health(request: Request):
            uptime = (datetime.now() - server.start_time).total_seconds() if server.start_time else 0
            body = {
                "status": "ok",
                "pid": os.getpid(),
                "uptime_seconds": uptime,
                "request_count": server.request_count,
            }
            # Let the client verify it reached the holder of its token.
            nonce = request.headers.get(NONCE_HEADER)
            if nonce:
                body["proof"] = health_proof(server.token, nonce)
            return body

        return app

    def start(self):
        """Start the server with uvicorn."""
        import uvicorn

        self.start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("GSAS-II Server Starting")
        self.logger.info("=" * 60)

        # Bind first so the endpoint file records the real (ephemeral) port
        # and connections queue from the moment clients can discover it.
        sock = _bind_listener(_requested_port())
        port = sock.getsockname()[1]
        _write_private(_endpoint_file(), json.dumps(
            {'pid': os.getpid(), 'port': port, 'token': self.token}))

        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.running = True
        self.logger.info(f"GSAS-II server started")
        self.logger.info(f"   URL: http://{HOST}:{port}")
        self.logger.info(f"   Log file: {log_file()}")
        self.logger.info(f"   PID: {os.getpid()}")
        self.logger.info(f"   Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"   Listening for simulation requests...")
        self.logger.info("-" * 60)

        app = self.create_app()
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])

    def _run_simulation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run a GSAS-II simulation using the loaded kicker module.

        Args:
            request: Dictionary with 'recipe_data' (inline recipe dict), 'output_dir',
                     and optional 'verbose'

        Returns:
            Dictionary with keys: 'success', 'run_id', 'rwp', 'elapsed_time',
            'method', 'output_files', 'fit_profile' (dict), 'unit_cell_data' (dict),
            'peak_list_data' (dict), 'refined_parameters' (list), 'spf_peaks' (dict),
            'spf_convergence_diagnostics' (dict), and optional 'error', 'traceback'.
        """
        from powderline.kicker import run_refinement
        from powderline.schema import RecipeModel
        from pydantic import ValidationError

        recipe_dict = request.get('recipe_data')
        output_dir = Path(request['output_dir'])
        verbose = request.get('verbose', False)

        if recipe_dict is None:
            return {
                'success': False,
                'error': 'Missing recipe_data in request',
                'method': 'server'
            }

        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate recipe
        try:
            recipe = RecipeModel.model_validate(recipe_dict)
        except ValidationError as e:
            error_lines = [f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                          for err in e.errors()]
            return {
                'success': False,
                'error': f"Recipe validation failed:\n" + "\n".join(error_lines),
                'method': 'server'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to parse recipe: {str(e)}",
                'method': 'server'
            }

        # Run refinement directly using loaded GSAS-II libraries
        result = run_refinement(recipe, output_dir, verbose=verbose, method='server')

        return result

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up the endpoint and PID files."""
        _release_state_files()

        if self.start_time:
            uptime = datetime.now() - self.start_time
            self.logger.info(f"Server stopped")
            self.logger.info(f"   Total requests: {self.request_count}")
            self.logger.info(f"   Uptime: {uptime}")
            self.logger.info("=" * 60)


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists (cross-platform).

    ``os.kill(pid, 0)`` is a valid liveness probe on POSIX but not on Windows,
    where ``os.kill`` terminates the target (or raises) rather than checking it;
    use the Win32 API there instead.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_server_running() -> bool:
    """Check if this user's server is currently running (stale files are removed)."""
    pid_path = _pid_file()
    if not pid_path.exists():
        return False

    try:
        pid = int(pid_path.read_text())
    except ValueError:
        pid = None

    if pid is not None and _pid_alive(pid):
        return True
    # PID file is unreadable or the process is gone
    pid_path.unlink(missing_ok=True)
    _endpoint_file().unlink(missing_ok=True)
    return False


def stop_server() -> bool:
    """Stop the running server."""
    if not is_server_running():
        print("Server is not running")
        return False

    pid = int(_pid_file().read_text())
    print(f"Stopping server (PID {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)

        # Wait for graceful shutdown
        for _ in range(50):  # 5 seconds timeout
            time.sleep(0.1)
            if not is_server_running():
                print("Server stopped")
                return True

        # Force kill if still running. SIGKILL does not exist on Windows;
        # there os.kill(pid, SIGTERM) maps to TerminateProcess (a hard kill),
        # which is the correct escalation.
        print("Forcing shutdown...")
        force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, force_signal)
        time.sleep(0.5)

        if not is_server_running():
            print("Server stopped (forced)")
            return True
        else:
            print("Failed to stop server")
            return False

    except ProcessLookupError:
        print("Server process not found")
        _pid_file().unlink(missing_ok=True)
        return False


def _server_url() -> str:
    endpoint = read_endpoint()
    if endpoint is None:
        return "(not listening yet)"
    return f"http://{HOST}:{endpoint['port']}"


def get_server_info() -> Optional[Dict[str, Any]]:
    """Get information about the running server.

    Returns:
        Dictionary with server info (pid, url, log_file, uptime) or None if not running
    """
    if not is_server_running():
        return None

    try:
        pid = int(_pid_file().read_text())

        # Try to get process info
        import psutil
        proc = psutil.Process(pid)
        uptime = datetime.now() - datetime.fromtimestamp(proc.create_time())

        return {
            'pid': pid,
            'url': _server_url(),
            'log_file': str(log_file()),
            'uptime': str(uptime).split('.')[0],  # Remove microseconds
            'cpu_percent': proc.cpu_percent(interval=0.1),
            'memory_mb': proc.memory_info().rss / 1024 / 1024,
        }
    except Exception:
        # psutil not available, return basic info
        return {
            'pid': pid,
            'url': _server_url(),
            'log_file': str(log_file()),
        }


def _start_foreground() -> None:
    """Claim the PID file, load GSAS-II and serve until stopped."""
    if not _claim_pid_file():
        print("Server is already running")
        print(f"   PID: {_pid_file().read_text()}")
        print(f"   URL: {_server_url()}")
        print(f"\n   To view logs: pixi run gsas-server logs")
        sys.exit(1)
    # Release the claim even if the GSAS-II import below fails.
    atexit.register(_release_state_files)
    # No client may use a leftover endpoint while GSAS-II loads.
    _endpoint_file().unlink(missing_ok=True)

    print("Starting GSAS-II server...")
    logger = setup_logging(log_file())
    server = GSASServer(logger)
    server.start()


def main():
    """Main entry point for server management."""
    import argparse

    parser = argparse.ArgumentParser(description="GSAS-II persistent server")
    subparsers = parser.add_subparsers(dest='action', help='Server action')

    subparsers.add_parser('start', help='Start the server')
    subparsers.add_parser('stop', help='Stop the server')
    subparsers.add_parser('status', help='Show server status')
    subparsers.add_parser('restart', help='Restart the server')
    subparsers.add_parser('logs', help='Show recent server logs')

    args = parser.parse_args()
    action = args.action or 'status'  # Default to status if no action

    if action in ('start', 'restart') and server_disabled():
        print(f"Not starting: {NO_SERVER_MESSAGE}")
        sys.exit(1)

    try:
        _run_action(action)
    except ServerStateError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _run_action(action: str) -> None:
    if action == 'start':
        _start_foreground()

    elif action == 'stop':
        stop_server()

    elif action == 'status':
        info = get_server_info()
        if info:
            print(f"Server is running")
            print(f"   PID: {info['pid']}")
            print(f"   URL: {info['url']}")
            print(f"   Log file: {info['log_file']}")
            if 'uptime' in info:
                print(f"   Uptime: {info['uptime']}")
            if 'cpu_percent' in info:
                print(f"   CPU: {info['cpu_percent']:.1f}%")
            if 'memory_mb' in info:
                print(f"   Memory: {info['memory_mb']:.1f} MB")
            print(f"\n   To view logs: pixi run gsas-server logs")
        else:
            print("Server is not running")
            print(f"\n   To start: pixi run gsas-server start")
            print(f"   Log file: {log_file()}")
            sys.exit(1)

    elif action == 'restart':
        if is_server_running():
            stop_server()
            time.sleep(1)

        _start_foreground()

    elif action == 'logs':
        path = log_file()
        if not path.exists():
            print(f"Log file not created yet: {path}")
            sys.exit(0)

        print(f"Recent logs from {path}:")
        print("   (showing last 50 lines)")
        print("-" * 60)

        try:
            # Show last 50 lines
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"Failed to read log file: {e}")
            sys.exit(1)

        print("-" * 60)
        print(f"   To follow logs in real-time: tail -f {path}")


if __name__ == '__main__':
    main()
