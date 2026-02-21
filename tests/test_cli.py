from __future__ import annotations

import builtins

import httpx

from poecoder.cli import PoeCoderCLI


def test_cli_start_falls_back_to_direct_when_backend_unreachable(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    def fail_session(mode: str) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cli, "_ensure_session", fail_session)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "/quit")

    cli.start()
    assert cli.direct is True


def test_cli_command_backend_error_is_handled_without_traceback(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    def fake_session(mode: str) -> None:
        cli.state.session_id = "s1"

    responses = iter(["/listmodels", "/quit"])

    monkeypatch.setattr(cli, "_ensure_session", fake_session)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))
    monkeypatch.setattr(
        cli.http,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
    )

    cli.start()
    assert cli.direct is False
