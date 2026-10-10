"""Unit tests for gsas_server process-management helpers.

Covers the cross-platform liveness probe (_pid_alive), is_server_running(),
and stop_server() PID-file handling, plus the multi-user safety layer: the
per-user state directory checks, the PID-file start lock, bearer-token auth on
every route, and the POWDERLINE_NO_SERVER switch. All os.kill / PID-file
interactions are mocked with monkeypatch + tmp_path so NO real server is ever
started and no signals reach real processes.

Import note: gsas_server.py does NOT import GSAS-II at module top (only stdlib
+ pydantic); GSAS-II is imported lazily inside GSASServer.__init__ / the
request handlers. A plain module import is therefore safe without GSAS-II.
"""
import os
import signal
import subprocess
import sys

import pytest

pytest.importorskip("pydantic")

from powderline import gsas_server


# The ProcessLookupError/PermissionError semantics below are specific to the
# POSIX os.kill(pid, 0) implementation of _pid_alive; on Windows _pid_alive uses
# a ctypes/OpenProcess path and never calls os.kill, so monkeypatching os.kill
# is a no-op there. Those cases are POSIX-only; cross-platform behaviour is
# covered by the real-PID contract tests (current process alive / exited process
# dead), which run the actual platform branch with no mocking.
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="tests the POSIX os.kill branch of _pid_alive"
)


@pytest.fixture
def isolated_pid_files(monkeypatch, tmp_path):
    """Point the state dir at a throwaway tmp dir (state files absent by default)."""
    monkeypatch.setattr(gsas_server, "state_dir", lambda: tmp_path)
    return tmp_path / "server.pid", tmp_path / "server.json"


# --- _pid_alive: cross-platform contract (real PIDs, no mocking) ---

def test_pid_alive_true_for_current_process():
    """The running test process is alive on any platform."""
    assert gsas_server._pid_alive(os.getpid()) is True


def test_pid_alive_false_for_exited_process():
    """A process that has exited is reported dead on any platform."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert gsas_server._pid_alive(proc.pid) is False


# --- _pid_alive: POSIX os.kill branch (skipped on Windows) ---

@posix_only
def test_pid_alive_true_when_kill_succeeds(monkeypatch):
    monkeypatch.setattr(gsas_server.os, "kill", lambda pid, sig: None)
    assert gsas_server._pid_alive(12345) is True


@posix_only
def test_pid_alive_false_when_process_lookup_error(monkeypatch):
    def _raise(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(gsas_server.os, "kill", _raise)
    assert gsas_server._pid_alive(12345) is False


@posix_only
def test_pid_alive_true_on_permission_error(monkeypatch):
    # Process exists but we lack permission to signal it -> still "alive".
    def _raise(pid, sig):
        raise PermissionError()

    monkeypatch.setattr(gsas_server.os, "kill", _raise)
    assert gsas_server._pid_alive(1) is True


# --- is_server_running ---

def test_is_server_running_false_without_pid_file(isolated_pid_files):
    pid_file, _ = isolated_pid_files
    assert not pid_file.exists()
    assert gsas_server.is_server_running() is False


def test_is_server_running_true_for_live_pid(isolated_pid_files, monkeypatch):
    pid_file, _ = isolated_pid_files
    pid_file.write_text("4242")
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: True)
    assert gsas_server.is_server_running() is True
    # Live PID file is left in place.
    assert pid_file.exists()


def test_is_server_running_stale_pid_removes_file(isolated_pid_files, monkeypatch):
    pid_file, _ = isolated_pid_files
    pid_file.write_text("4242")
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: False)
    assert gsas_server.is_server_running() is False
    # Stale PID file is cleaned up.
    assert not pid_file.exists()


def test_is_server_running_garbage_pid_removes_file(isolated_pid_files):
    pid_file, _ = isolated_pid_files
    pid_file.write_text("not-an-int")
    assert gsas_server.is_server_running() is False
    assert not pid_file.exists()


# --- stop_server ---

def test_stop_server_when_not_running(isolated_pid_files, capsys):
    # No PID file at all -> graceful no-op, returns False, no exception.
    result = gsas_server.stop_server()
    assert result is False
    assert "not running" in capsys.readouterr().out.lower()


def test_stop_server_stale_pid_graceful(isolated_pid_files, monkeypatch, capsys):
    # PID file present but process dead: is_server_running() detects the stale
    # PID, removes the file, and stop_server() reports "not running" -> False.
    pid_file, _ = isolated_pid_files
    pid_file.write_text("999999")
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: False)

    # Guard: os.kill must never be invoked on a dead PID here.
    def _boom(pid, sig):
        raise AssertionError("os.kill should not be called for a stale PID")

    monkeypatch.setattr(gsas_server.os, "kill", _boom)

    result = gsas_server.stop_server()
    assert result is False
    assert "not running" in capsys.readouterr().out.lower()
    assert not pid_file.exists()


def test_stop_server_sends_sigterm_to_live_pid(isolated_pid_files, monkeypatch, capsys):
    # Live PID: stop_server should signal it, then observe it gone on next poll.
    pid_file, _ = isolated_pid_files
    pid_file.write_text("4242")

    calls = {"signals": []}

    def fake_kill(pid, sig):
        calls["signals"].append((pid, sig))

    # First is_server_running() (guard) True; after SIGTERM, report dead.
    alive_states = iter([True, False])
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: next(alive_states))
    monkeypatch.setattr(gsas_server.os, "kill", fake_kill)
    monkeypatch.setattr(gsas_server.time, "sleep", lambda s: None)

    result = gsas_server.stop_server()
    assert result is True
    assert calls["signals"], "expected at least one signal to be sent"
    assert calls["signals"][0][0] == 4242
    assert calls["signals"][0][1] == signal.SIGTERM
    assert "stopped" in capsys.readouterr().out.lower()


# The escalation signal stop_server() uses when the graceful SIGTERM window
# expires: SIGKILL where it exists; on Windows (no SIGKILL) os.kill(pid,
# SIGTERM) maps to TerminateProcess, so SIGTERM *is* the hard kill there.
_FORCE_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)


def test_stop_server_force_kill_escalation(isolated_pid_files, monkeypatch, capsys):
    # PID ignores SIGTERM through the whole graceful 5s window and dies only
    # after the force kill: exactly SIGTERM then the force signal are sent.
    pid_file, _ = isolated_pid_files
    pid_file.write_text("4242")

    calls = {"signals": []}

    def fake_kill(pid, sig):
        calls["signals"].append((pid, sig))

    # Alive until the second (force) signal has been sent, then dead.
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: len(calls["signals"]) < 2)
    monkeypatch.setattr(gsas_server.os, "kill", fake_kill)
    monkeypatch.setattr(gsas_server.time, "sleep", lambda s: None)

    result = gsas_server.stop_server()
    assert result is True
    assert calls["signals"] == [(4242, signal.SIGTERM), (4242, _FORCE_SIGNAL)]
    assert "forced" in capsys.readouterr().out.lower()


def test_stop_server_force_kill_never_dies_returns_false(isolated_pid_files, monkeypatch, capsys):
    # PID survives even the force kill: stop_server() gives up and reports False
    # after sending exactly SIGTERM then the force signal.
    pid_file, _ = isolated_pid_files
    pid_file.write_text("4242")

    calls = {"signals": []}

    def fake_kill(pid, sig):
        calls["signals"].append((pid, sig))

    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(gsas_server.os, "kill", fake_kill)
    monkeypatch.setattr(gsas_server.time, "sleep", lambda s: None)

    result = gsas_server.stop_server()
    assert result is False
    assert calls["signals"] == [(4242, signal.SIGTERM), (4242, _FORCE_SIGNAL)]
    assert "failed to stop" in capsys.readouterr().out.lower()


# --- POWDERLINE_NO_SERVER ---

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " Yes "])
def test_server_disabled_truthy(monkeypatch, value):
    monkeypatch.setenv("POWDERLINE_NO_SERVER", value)
    assert gsas_server.server_disabled() is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_server_disabled_falsy(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("POWDERLINE_NO_SERVER", raising=False)
    else:
        monkeypatch.setenv("POWDERLINE_NO_SERVER", value)
    assert gsas_server.server_disabled() is False


@pytest.mark.parametrize("action", ["start", "restart"])
def test_main_start_refuses_when_disabled(monkeypatch, capsys, action):
    monkeypatch.setenv("POWDERLINE_NO_SERVER", "1")
    monkeypatch.setattr(sys, "argv", ["gsas_server.py", action])
    monkeypatch.setattr(gsas_server, "GSASServer",
                        lambda *a, **k: pytest.fail("server must not be created"))
    monkeypatch.setattr(gsas_server, "stop_server",
                        lambda: pytest.fail("restart must not stop the server first"))
    with pytest.raises(SystemExit) as exc:
        gsas_server.main()
    assert exc.value.code == 1
    assert "POWDERLINE_NO_SERVER" in capsys.readouterr().out


# --- Per-user state directory ---

@posix_only
def test_state_dir_uses_private_xdg_runtime_dir(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    path = gsas_server.state_dir()
    assert path == runtime / "powderline"
    assert (path.stat().st_mode & 0o777) == 0o700


@posix_only
def test_state_dir_falls_back_to_per_uid_tempdir(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir", lambda: str(tmp_path))
    path = gsas_server.state_dir()
    assert path == tmp_path / f"powderline-{os.getuid()}"
    assert (path.stat().st_mode & 0o777) == 0o700


@posix_only
def test_state_dir_ignores_shared_xdg_runtime_dir(monkeypatch, tmp_path):
    # An XDG_RUNTIME_DIR others can write to (e.g. inherited after su) is not used.
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(shared))
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir", lambda: str(tmp_path))
    assert gsas_server.state_dir() == tmp_path / f"powderline-{os.getuid()}"


@posix_only
def test_state_dir_path_is_distinct_per_uid(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir", lambda: str(tmp_path))
    paths = set()
    for uid in (1001, 1002):
        monkeypatch.setattr(gsas_server.os, "getuid", lambda uid=uid: uid)
        paths.add(gsas_server._state_dir_path())
    assert paths == {tmp_path / "powderline-1001", tmp_path / "powderline-1002"}


@posix_only
def test_state_dir_refuses_dir_owned_by_another_user(monkeypatch, tmp_path):
    # The dir exists (created by "someone else"): we are not its owner.
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir", lambda: str(tmp_path))
    other_uid = os.getuid() + 1
    (tmp_path / f"powderline-{other_uid}").mkdir(mode=0o700)
    monkeypatch.setattr(gsas_server.os, "getuid", lambda: other_uid)
    with pytest.raises(gsas_server.ServerStateError, match="owned by uid"):
        gsas_server.state_dir()


@posix_only
def test_state_dir_refuses_symlink(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir", lambda: str(tmp_path))
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    (tmp_path / f"powderline-{os.getuid()}").symlink_to(target)
    with pytest.raises(gsas_server.ServerStateError, match="symlink"):
        gsas_server.state_dir()


@posix_only
@pytest.mark.parametrize("mode", [0o750, 0o705, 0o777])
def test_state_dir_refuses_group_or_other_access(monkeypatch, tmp_path, mode):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir", lambda: str(tmp_path))
    path = tmp_path / f"powderline-{os.getuid()}"
    path.mkdir()
    path.chmod(mode)
    with pytest.raises(gsas_server.ServerStateError, match="accessible to other users"):
        gsas_server.state_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows state-dir location")
def test_state_dir_windows_uses_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert gsas_server.state_dir() == tmp_path / "powderline"


@posix_only
def test_write_private_is_owner_only(tmp_path):
    target = tmp_path / "server.json"
    target.write_text("old")
    target.chmod(0o644)
    gsas_server._write_private(target, "new")
    assert target.read_text() == "new"
    assert (target.stat().st_mode & 0o777) == 0o600


def test_read_endpoint_absent_or_garbage_is_none(isolated_pid_files):
    _, endpoint_file = isolated_pid_files
    assert gsas_server.read_endpoint() is None
    endpoint_file.write_text("{not json")
    assert gsas_server.read_endpoint() is None
    endpoint_file.write_text('{"pid": 1, "port": 2}')  # no token
    assert gsas_server.read_endpoint() is None
    endpoint_file.write_text('{"pid": 1, "port": 2, "token": "t"}')
    assert gsas_server.read_endpoint() == {"pid": 1, "port": 2, "token": "t"}


# --- Start lock / state-file lifecycle ---

def test_claim_pid_file_when_absent(isolated_pid_files):
    pid_file, _ = isolated_pid_files
    assert gsas_server._claim_pid_file() is True
    assert pid_file.read_text() == str(os.getpid())


def test_claim_pid_file_refused_while_server_alive(isolated_pid_files, monkeypatch):
    pid_file, _ = isolated_pid_files
    pid_file.write_text("4242")
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: True)
    assert gsas_server._claim_pid_file() is False
    assert pid_file.read_text() == "4242"


def test_claim_pid_file_replaces_stale_claim(isolated_pid_files, monkeypatch):
    pid_file, endpoint_file = isolated_pid_files
    pid_file.write_text("4242")
    endpoint_file.write_text('{"pid": 4242, "port": 1, "token": "old"}')
    monkeypatch.setattr(gsas_server, "_pid_alive", lambda pid: False)
    assert gsas_server._claim_pid_file() is True
    assert pid_file.read_text() == str(os.getpid())
    assert not endpoint_file.exists()  # stale endpoint removed with it


def test_release_state_files_only_removes_own(isolated_pid_files):
    pid_file, endpoint_file = isolated_pid_files
    pid_file.write_text("4242")
    endpoint_file.write_text("{}")
    gsas_server._release_state_files()
    assert pid_file.exists() and endpoint_file.exists()

    pid_file.write_text(str(os.getpid()))
    gsas_server._release_state_files()
    assert not pid_file.exists() and not endpoint_file.exists()


# --- Bearer-token auth on the FastAPI app ---

@pytest.fixture
def app_client():
    """TestClient for the real app, without importing GSAS-II or binding a port."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import logging
    from fastapi.testclient import TestClient

    server = gsas_server.GSASServer.__new__(gsas_server.GSASServer)
    server.request_count = 0
    server.start_time = None
    server.token = "s3cret-token"
    server.logger = logging.getLogger("powderline.server.test")
    calls = []
    server._run_simulation = lambda req: calls.append(req) or {
        "success": False, "error": "stub", "method": "server"}
    return TestClient(server.create_app()), server.token, calls


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer wrong"},
    {"Authorization": "s3cret-token"},          # missing scheme
    {"Authorization": "Bearer s3cret-token "},  # not an exact match
    {"Authorization": "Bearer s3cret-tokén".encode("latin-1")},  # non-ASCII: no crash
])
def test_unauthenticated_requests_rejected(app_client, headers, tmp_path):
    client, _, calls = app_client
    out = tmp_path / "out"
    resp = client.post("/simulate", headers=headers,
                       json={"recipe_data": {}, "output_dir": str(out)})
    assert resp.status_code == 401
    assert client.get("/health", headers=headers).status_code == 401
    assert calls == [], "no job may run without the token"
    assert not out.exists()


def test_authenticated_requests_accepted(app_client, tmp_path):
    client, token, calls = app_client
    auth = {"Authorization": f"Bearer {token}"}
    resp = client.post("/simulate", headers=auth,
                       json={"recipe_data": {}, "output_dir": str(tmp_path)})
    assert resp.status_code == 200
    assert len(calls) == 1


def test_health_proves_token_knowledge(app_client):
    client, token, _ = app_client
    resp = client.get("/health", headers={"Authorization": f"Bearer {token}",
                                          gsas_server.NONCE_HEADER: "abc123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["proof"] == gsas_server.health_proof(token, "abc123")
    assert body["proof"] != gsas_server.health_proof("other-token", "abc123")


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_routes_disabled(app_client, path):
    client, token, _ = app_client
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": f"Bearer {token}"}).status_code == 404


@posix_only
def test_state_dir_uncreatable_raises_state_error(monkeypatch, tmp_path):
    # e.g. an unwritable tempdir: callers must get ServerStateError, not OSError.
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(gsas_server.tempfile, "gettempdir",
                        lambda: str(tmp_path / "missing" / "deeper"))
    with pytest.raises(gsas_server.ServerStateError, match="Cannot create"):
        gsas_server.state_dir()
