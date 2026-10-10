"""Unit tests for GSASClient.submit_simulation.

Primarily the server output-visibility guard: a GSAS-II server can be
reachable over localhost yet have a different filesystem view than the client
— e.g. running on another cluster node, or inside a sandbox/container with a
private /tmp. Such a server reports success while its output files never
appear for the client. submit_simulation must detect this (fit_profile.txt
not freshly written after a "successful" server run — a stale file from a
previous run into the same output_dir does not count), warn, and fall back
to in-process execution — or return a structured error when fallback is
disabled. Also covers the use_server bypass, the keyword-signature
regression guard, server identity (a server must prove it holds this user's
token) and the POWDERLINE_NO_SERVER switch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from powderline import gsas_server
from powderline.gsas_client import GSASClient

RECIPE = {"schema_name": "GSASII_Rietveld", "schema_version": "0.26.0", "payload": {}}


@pytest.fixture(autouse=True)
def _server_enabled(monkeypatch):
    """Tests assume server use is allowed unless they opt out explicitly."""
    monkeypatch.delenv("POWDERLINE_NO_SERVER", raising=False)


@pytest.fixture
def client(monkeypatch):
    c = GSASClient()
    monkeypatch.setattr(c, "is_server_available", lambda: True)
    return c


def test_server_success_without_visible_files_falls_back(client, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(client, "_submit_to_server",
                        lambda rd, od: {"success": True})
    calls = []
    monkeypatch.setattr(client, "_submit_via_subprocess",
                        lambda rd, od, verbose: calls.append(1) or {"success": True})

    result = client.submit_simulation(RECIPE, tmp_path)

    assert calls, "expected fallback to in-process execution"
    assert result["method"] == "subprocess"
    out = capsys.readouterr().out
    assert "did not write output files visible to this process" in out
    assert "gsas-server restart" in out


def test_server_success_with_visible_files_is_returned(client, monkeypatch, tmp_path):
    def fake_server_run(rd, od):
        (tmp_path / "fit_profile.txt").write_text("two_theta\ty_calc\n")
        return {"success": True}

    monkeypatch.setattr(client, "_submit_to_server", fake_server_run)
    monkeypatch.setattr(client, "_submit_via_subprocess",
                        lambda rd, od, verbose: pytest.fail("must not fall back"))

    result = client.submit_simulation(RECIPE, tmp_path)
    assert result["success"] is True
    assert result["method"] == "server"


def test_stale_fit_profile_does_not_mask_divergent_server(client, monkeypatch, tmp_path, capsys):
    """A fit_profile.txt left by a PREVIOUS run into the same output_dir must
    not satisfy the guard: if the server 'succeeds' without (re)writing the
    file, the run falls back in-process."""
    (tmp_path / "fit_profile.txt").write_text("two_theta\ty_calc\n1.0\t10.0\n")

    monkeypatch.setattr(client, "_submit_to_server",
                        lambda rd, od: {"success": True})  # touches nothing
    calls = []
    monkeypatch.setattr(client, "_submit_via_subprocess",
                        lambda rd, od, verbose: calls.append(1) or {"success": True})

    result = client.submit_simulation(RECIPE, tmp_path)

    assert calls, "stale fit_profile.txt must not pass the freshness check"
    assert result["method"] == "subprocess"
    assert "did not write output files visible" in capsys.readouterr().out


def test_rewritten_fit_profile_counts_as_fresh(client, monkeypatch, tmp_path):
    """Re-running into a used output_dir with a healthy server: the rewritten
    fit_profile.txt (changed stat) must pass the guard."""
    (tmp_path / "fit_profile.txt").write_text("two_theta\ty_calc\n1.0\t10.0\n")

    def fake_server_run(rd, od):
        # New content with a different size — deterministic stat change even
        # on filesystems with coarse mtime granularity
        (tmp_path / "fit_profile.txt").write_text(
            "two_theta\ty_calc\n1.0\t10.0\n2.0\t20.0\n")
        return {"success": True}

    monkeypatch.setattr(client, "_submit_to_server", fake_server_run)
    monkeypatch.setattr(client, "_submit_via_subprocess",
                        lambda rd, od, verbose: pytest.fail("must not fall back"))

    result = client.submit_simulation(RECIPE, tmp_path)
    assert result["success"] is True
    assert result["method"] == "server"


def test_server_error_result_not_subject_to_file_check(client, monkeypatch, tmp_path):
    """A server-reported failure is returned as-is (no file check, no fallback
    beyond the existing error path)."""
    monkeypatch.setattr(client, "_submit_to_server",
                        lambda rd, od: {"success": False, "error": "boom"})
    monkeypatch.setattr(client, "_submit_via_subprocess",
                        lambda rd, od, verbose: pytest.fail("must not fall back"))

    result = client.submit_simulation(RECIPE, tmp_path)
    assert result["success"] is False
    assert result["method"] == "server"


def test_use_server_false_skips_server_entirely(monkeypatch, tmp_path):
    c = GSASClient()
    monkeypatch.setattr(c, "is_server_available",
                        lambda: pytest.fail("server must not be consulted"))
    monkeypatch.setattr(c, "_submit_via_subprocess",
                        lambda rd, od, verbose: {"success": True})

    result = c.submit_simulation(RECIPE, tmp_path, use_server=False)
    assert result["success"] is True
    assert result["method"] == "subprocess"


def test_use_server_false_with_fallback_disabled_errors_honestly(monkeypatch, tmp_path):
    """Both execution paths disabled by the caller is a legal (if odd) state:
    it must return a structured error naming BOTH flags — not the misleading
    'Server not available' message (the server was never consulted)."""
    c = GSASClient(fallback_to_subprocess=False)
    monkeypatch.setattr(c, "is_server_available",
                        lambda: pytest.fail("server must not be consulted"))

    result = c.submit_simulation(RECIPE, tmp_path, use_server=False)
    assert result["success"] is False
    assert result["method"] == "none"
    assert "use_server=False" in result["error"]
    assert "fallback_to_subprocess=False" in result["error"]
    assert "Server not available" not in result["error"]


def test_invisible_output_with_fallback_disabled_returns_error(monkeypatch, tmp_path, capsys):
    """With fallback_to_subprocess=False (execution_mode='server'), invisible
    server output must yield a structured error naming the filesystem-view
    mismatch — not the misleading 'server not available' message, and no
    promise of a fallback that won't happen."""
    c = GSASClient(fallback_to_subprocess=False)
    monkeypatch.setattr(c, "is_server_available", lambda: True)
    monkeypatch.setattr(c, "_submit_to_server", lambda rd, od: {"success": True})
    monkeypatch.setattr(c, "_submit_via_subprocess",
                        lambda rd, od, verbose: pytest.fail("must not fall back"))

    result = c.submit_simulation(RECIPE, tmp_path)

    assert result["success"] is False
    assert result["method"] == "server"
    assert "did not write output files visible" in result["error"]
    out = capsys.readouterr().out
    assert "fallback is disabled" in out
    assert "Falling back" not in out


def test_invisible_output_guard_in_auto_start_branch(monkeypatch, tmp_path, capsys):
    """The guard must also cover the auto-start branch (server initially
    unavailable, then started and answering with invisible output)."""
    c = GSASClient()
    availability = iter([False, True])  # unavailable, then up after auto-start
    monkeypatch.setattr(c, "is_server_available", lambda: next(availability))
    monkeypatch.setattr(c, "_start_server_background", lambda verbose: True)
    monkeypatch.setattr(c, "_submit_to_server", lambda rd, od: {"success": True})
    calls = []
    monkeypatch.setattr(c, "_submit_via_subprocess",
                        lambda rd, od, verbose: calls.append(1) or {"success": True})

    result = c.submit_simulation(RECIPE, tmp_path, auto_start_server=True)

    assert calls, "expected fallback to in-process execution"
    assert result["method"] == "subprocess"
    assert ("did not write output files visible to this process"
            in capsys.readouterr().out)


def test_submit_simulation_wrong_keyword_raises():
    """Regression guard: 'recipe_path' keyword no longer exists on
    submit_simulation (mp_simulate.py once passed recipe_path=..., causing a
    runtime TypeError; the correct keyword is recipe=)."""
    client = GSASClient()
    with pytest.raises(TypeError, match="recipe_path"):
        client.submit_simulation(recipe_path=Path("dummy.json"), output_dir=Path("."))


# --- Server identity: only this user's server, proven by its token ---

class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def own_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(gsas_server, "state_dir", lambda: tmp_path)
    (tmp_path / "server.json").write_text(
        '{"pid": 4242, "port": 50123, "token": "my-token"}')
    return tmp_path


def _fake_health(monkeypatch, respond):
    """Patch httpx.get; ``respond(headers) -> _FakeResponse``. Returns call log."""
    import httpx
    calls = []

    def fake_get(url, headers=None, **kwargs):
        calls.append({"url": url, "headers": headers, **kwargs})
        return respond(headers)

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_server_unavailable_without_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(gsas_server, "state_dir", lambda: tmp_path)
    calls = _fake_health(monkeypatch, lambda h: pytest.fail("must not probe"))
    assert GSASClient().is_server_available() is False
    assert calls == []


def test_server_available_when_it_proves_the_token(monkeypatch, own_endpoint):
    calls = _fake_health(monkeypatch, lambda h: _FakeResponse(200, {
        "proof": gsas_server.health_proof("my-token", h[gsas_server.NONCE_HEADER])}))
    assert GSASClient().is_server_available() is True
    (call,) = calls
    assert call["url"] == "http://127.0.0.1:50123/health"
    assert call["headers"]["Authorization"] == "Bearer my-token"
    assert call["trust_env"] is False  # token never sent through a proxy


@pytest.mark.parametrize("response", [
    _FakeResponse(200, {"status": "ok"}),                  # old/unauthenticated server
    _FakeResponse(200, {"proof": "0" * 64}),               # impostor guessing
    _FakeResponse(200, {"proof": gsas_server.health_proof("other-token", "x")}),
    _FakeResponse(200, {"proof": 123}),
    _FakeResponse(200, ["not", "a", "dict"]),
    _FakeResponse(401, {"detail": "Unauthorized"}),         # another user's server
])
def test_server_unavailable_without_valid_proof(monkeypatch, own_endpoint, response):
    _fake_health(monkeypatch, lambda h: response)
    assert GSASClient().is_server_available() is False


def test_unsafe_state_dir_means_no_server(monkeypatch, capsys):
    def unsafe():
        raise gsas_server.ServerStateError("unsafe state dir")

    monkeypatch.setattr(gsas_server, "state_dir", unsafe)
    _fake_health(monkeypatch, lambda h: pytest.fail("must not probe"))
    c = GSASClient()
    assert c.is_server_available() is False
    assert c.is_server_available() is False
    assert c._start_server_background() is False
    assert capsys.readouterr().out.count("unsafe state dir") == 1


def test_submit_sends_bearer_token(monkeypatch, own_endpoint, tmp_path):
    import httpx
    seen = {}

    class _Ok:
        def raise_for_status(self):
            pass

        def json(self):
            return {"success": True}

    def fake_post(url, json=None, headers=None, **kwargs):
        seen.update(url=url, headers=headers, **kwargs)
        return _Ok()

    monkeypatch.setattr(httpx, "post", fake_post)
    GSASClient()._submit_to_server(RECIPE, tmp_path / "out")
    assert seen["url"] == "http://127.0.0.1:50123/simulate"
    assert seen["headers"]["Authorization"] == "Bearer my-token"
    assert seen["trust_env"] is False


# --- POWDERLINE_NO_SERVER ---

@pytest.fixture
def no_server(monkeypatch):
    monkeypatch.setenv("POWDERLINE_NO_SERVER", "1")


def _forbid_server(monkeypatch, c):
    for name in ("is_server_available", "_submit_to_server"):
        monkeypatch.setattr(c, name, lambda *a, **k: pytest.fail(f"{name} called"))
    monkeypatch.setattr(c, "_start_server_background",
                        lambda *a, **k: pytest.fail("server auto-started"))


def test_no_server_env_auto_runs_in_process(no_server, monkeypatch, tmp_path, capsys):
    c = GSASClient()
    _forbid_server(monkeypatch, c)
    monkeypatch.setattr(c, "_submit_via_subprocess",
                        lambda rd, od, verbose: {"success": True})
    result = c.submit_simulation(RECIPE, tmp_path, auto_start_server=True)
    assert result == {"success": True, "method": "subprocess"}
    assert "POWDERLINE_NO_SERVER" in capsys.readouterr().out


def test_no_server_env_server_mode_errors(no_server, monkeypatch, tmp_path):
    c = GSASClient(fallback_to_subprocess=False)
    _forbid_server(monkeypatch, c)
    monkeypatch.setattr(c, "_submit_via_subprocess",
                        lambda *a: pytest.fail("must not run in-process"))
    result = c.submit_simulation(RECIPE, tmp_path)
    assert result["success"] is False
    assert "POWDERLINE_NO_SERVER" in result["error"]


def test_no_server_env_blocks_background_start(no_server, monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(sp, "Popen", lambda *a, **k: pytest.fail("server spawned"))
    assert GSASClient()._start_server_background() is False
