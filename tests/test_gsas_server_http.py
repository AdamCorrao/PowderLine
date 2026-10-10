"""Tests for GSAS-II server HTTP endpoints.

These tests verify that the FastAPI server exposes the correct HTTP API,
handles requests properly, and rejects callers that do not hold its token.

These are integration tests: the module fixture starts a real server in a
private, per-test state directory (via ``XDG_RUNTIME_DIR`` / ``LOCALAPPDATA``),
so it never touches — or reuses — the developer's own server, and reads the
server's port and bearer token from that directory. They skip if the server
cannot start (e.g. GSAS-II unavailable).
"""

import json
import pytest
import time
import subprocess
import sys
from pathlib import Path
import tempfile
import os

from powderline import gsas_server
from powderline.gsas_client import GSASClient


def _endpoint_in(state_root: Path):
    """The endpoint (pid/port/token) the server wrote under ``state_root``."""
    try:
        return json.loads((state_root / 'powderline' / 'server.json').read_text())
    except (OSError, ValueError):
        return None


def _health_ok(port: int, token: str) -> bool:
    import httpx
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0,
                         headers={"Authorization": f"Bearer {token}"}, trust_env=False)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


@pytest.fixture(scope="module")
def server_state_root(tmp_path_factory):
    """A private (0700) directory standing in for this user's runtime dir."""
    root = tmp_path_factory.mktemp("server_state")
    root.chmod(0o700)
    return root


@pytest.fixture(scope="module")
def test_server(server_state_root):
    """Start a server in an isolated state dir; yield ``(port, token)``."""
    src_dir = Path(__file__).parent.parent / 'src'
    server_script = src_dir / 'powderline' / 'gsas_server.py'

    if not server_script.exists():
        pytest.skip(f"Server script not found: {server_script}")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(src_dir) + os.pathsep + env.get('PYTHONPATH', '')
    env['XDG_RUNTIME_DIR'] = str(server_state_root)   # POSIX state dir
    env['LOCALAPPDATA'] = str(server_state_root)      # Windows state dir
    env.pop('POWDERLINE_NO_SERVER', None)
    env.pop('POWDERLINE_SERVER_PORT', None)           # exercise the ephemeral port

    proc = subprocess.Popen(
        [sys.executable, str(server_script), 'start'],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Wait for server to start (up to 30 seconds for GSAS-II import)
    endpoint = None
    for attempt in range(300):
        time.sleep(0.1)
        endpoint = _endpoint_in(server_state_root)
        if endpoint and _health_ok(endpoint['port'], endpoint['token']):
            break
        if proc.poll() is not None:
            pytest.skip(f"Test server exited unexpectedly (exit code: {proc.returncode})")
    else:
        proc.kill()
        pytest.skip("Test server failed to start within 30 seconds")

    yield endpoint['port'], endpoint['token']

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint(test_server):
    """Test /health endpoint returns correct format."""
    import httpx

    port, token = test_server
    resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=_auth(token), timeout=2.0,
                     trust_env=False)

    assert resp.status_code == 200, "Health endpoint should return 200"

    data = resp.json()
    assert "status" in data, "Response should contain 'status' field"
    assert data["status"] == "ok", "Status should be 'ok'"
    assert "pid" in data, "Response should contain 'pid' field"
    assert "uptime_seconds" in data, "Response should contain 'uptime_seconds' field"
    assert "request_count" in data, "Response should contain 'request_count' field"

    assert isinstance(data["pid"], int), "PID should be integer"
    assert data["pid"] > 0, "PID should be positive"
    assert data["uptime_seconds"] >= 0, "Uptime should be non-negative"
    assert data["request_count"] >= 0, "Request count should be non-negative"


def test_health_endpoint_format(test_server):
    """Verify health endpoint returns proper JSON structure."""
    import httpx

    port, token = test_server
    resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=_auth(token), timeout=2.0,
                     trust_env=False)

    # Verify response is valid JSON
    data = resp.json()

    # Verify all expected fields are present
    required_fields = {"status", "pid", "uptime_seconds", "request_count"}
    assert required_fields.issubset(data.keys()), \
        f"Missing required fields: {required_fields - data.keys()}"


def test_simulate_endpoint_missing_fields(test_server):
    """Test /simulate endpoint validates required fields."""
    import httpx

    port, token = test_server

    # Missing recipe_data
    resp = httpx.post(
        f"http://127.0.0.1:{port}/simulate",
        headers=_auth(token),
        trust_env=False,
        json={"output_dir": "/tmp/test"},
        timeout=5.0
    )
    assert resp.status_code == 422, "Should return validation error for missing recipe_data"

    # Missing output_dir
    resp = httpx.post(
        f"http://127.0.0.1:{port}/simulate",
        headers=_auth(token),
        trust_env=False,
        json={"recipe_data": {}},
        timeout=5.0
    )
    assert resp.status_code == 422, "Should return validation error for missing output_dir"

    # Empty request
    resp = httpx.post(
        f"http://127.0.0.1:{port}/simulate",
        headers=_auth(token),
        trust_env=False,
        json={},
        timeout=5.0
    )
    assert resp.status_code == 422, "Should return validation error for empty request"


def test_simulate_endpoint_invalid_recipe(test_server):
    """Test /simulate endpoint handles invalid recipe_data gracefully."""
    import httpx

    port, token = test_server

    # Invalid (empty) recipe_data — server should return 200 with success=False
    with tempfile.TemporaryDirectory() as tmpdir:
        resp = httpx.post(
            f"http://127.0.0.1:{port}/simulate",
            headers=_auth(token),
            trust_env=False,
            json={
                "recipe_data": {},
                "output_dir": tmpdir
            },
            timeout=5.0
        )

        assert resp.status_code == 200, "Should return 200 even for errors (error in response body)"
        data = resp.json()

        assert "success" in data, "Response should contain 'success' field"
        assert data["success"] is False, "Should indicate failure"
        assert "error" in data, "Response should contain 'error' field"
        assert "recipe" in data["error"].lower() or "validation" in data["error"].lower(), \
            "Error message should mention recipe validation failure"


def test_simulate_endpoint_optional_verbose(test_server):
    """Test /simulate endpoint accepts optional verbose parameter."""
    import httpx

    port, token = test_server

    with tempfile.TemporaryDirectory() as tmpdir:
        # Request with verbose=true — should accept and return 200 (even if recipe invalid)
        resp = httpx.post(
            f"http://127.0.0.1:{port}/simulate",
            headers=_auth(token),
            trust_env=False,
            json={
                "recipe_data": {},
                "output_dir": tmpdir,
                "verbose": True
            },
            timeout=5.0
        )

        assert resp.status_code == 200, "Should accept verbose parameter"

        # Request with verbose=false
        resp = httpx.post(
            f"http://127.0.0.1:{port}/simulate",
            headers=_auth(token),
            trust_env=False,
            json={
                "recipe_data": {},
                "output_dir": tmpdir,
                "verbose": False
            },
            timeout=5.0
        )

        assert resp.status_code == 200, "Should accept verbose parameter"


def test_health_endpoint_performance(test_server):
    """Test that health endpoint responds quickly."""
    import httpx

    port, token = test_server

    start = time.time()
    resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=_auth(token), timeout=2.0,
                     trust_env=False)
    elapsed = time.time() - start

    assert resp.status_code == 200
    assert elapsed < 0.1, f"Health check should be fast, took {elapsed:.2f}s"


def test_multiple_health_checks(test_server):
    """Test that multiple health checks work correctly."""
    import httpx

    port, token = test_server

    # Multiple rapid health checks
    for _ in range(10):
        resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=_auth(token), timeout=2.0,
                     trust_env=False)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


def test_server_increments_request_count(test_server):
    """Test that server tracks request count correctly."""
    import httpx

    port, token = test_server

    # Get initial count (might not be 0 due to startup checks)
    resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=_auth(token), timeout=2.0,
                     trust_env=False)
    initial_count = resp.json()["request_count"]

    # Make a simulation request (will fail validation but should still increment counter)
    with tempfile.TemporaryDirectory() as tmpdir:
        httpx.post(
            f"http://127.0.0.1:{port}/simulate",
            headers=_auth(token),
            trust_env=False,
            json={
                "recipe_data": {},
                "output_dir": tmpdir
            },
            timeout=5.0
        )

    # Check count increased
    resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=_auth(token), timeout=2.0,
                     trust_env=False)
    new_count = resp.json()["request_count"]

    assert new_count > initial_count, "Request count should increment after simulation request"


# --- Multi-user safety: only the token holder can use the server ---

def test_requests_without_token_rejected(test_server, tmp_path):
    """No token (or a wrong one) -> 401 on every route, and nothing is written."""
    import httpx

    port, _ = test_server
    out = tmp_path / "other_users_output"
    for headers in ({}, {"Authorization": "Bearer not-the-token"}):
        resp = httpx.post(f"http://127.0.0.1:{port}/simulate", headers=headers,
                          json={"recipe_data": {}, "output_dir": str(out)},
                          timeout=5.0, trust_env=False)
        assert resp.status_code == 401
        resp = httpx.get(f"http://127.0.0.1:{port}/health", headers=headers,
                         timeout=2.0, trust_env=False)
        assert resp.status_code == 401
        assert "pid" not in resp.text
    assert not out.exists(), "an unauthenticated request must not create output_dir"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_state_files_are_owner_only(test_server, server_state_root):
    state = server_state_root / "powderline"
    assert (state.stat().st_mode & 0o777) == 0o700
    assert (state / "server.json").stat().st_mode & 0o777 == 0o600


def test_client_uses_own_server(test_server, server_state_root, monkeypatch):
    monkeypatch.delenv("POWDERLINE_NO_SERVER", raising=False)
    monkeypatch.setattr(gsas_server, "state_dir", lambda: server_state_root / "powderline")
    assert GSASClient().is_server_available() is True


def test_client_ignores_another_users_server(test_server, tmp_path, monkeypatch):
    """A client whose state dir lacks the token never treats the server as its own,
    even when it knows the port (here: a forged endpoint with a guessed token)."""
    port, _ = test_server
    monkeypatch.setattr(gsas_server, "state_dir", lambda: tmp_path)
    client = GSASClient()
    assert client.is_server_available() is False

    (tmp_path / "server.json").write_text(
        json.dumps({"pid": os.getpid(), "port": port, "token": "guessed"}))
    assert client.is_server_available() is False
    with pytest.raises(Exception):
        client._submit_to_server({}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refinement_through_authenticated_server(test_server, server_state_root,
                                                 monkeypatch, recipe_LaB6_dict, tmp_path):
    """End to end: the owner's client runs a real refinement via the server."""
    monkeypatch.delenv("POWDERLINE_NO_SERVER", raising=False)
    monkeypatch.setattr(gsas_server, "state_dir", lambda: server_state_root / "powderline")
    client = GSASClient(fallback_to_subprocess=False)
    result = client.submit_simulation(recipe_LaB6_dict, tmp_path, auto_start_server=False)
    assert result["success"] is True, result.get("error")
    assert result["method"] == "server"
    assert (tmp_path / "fit_profile.txt").exists()
