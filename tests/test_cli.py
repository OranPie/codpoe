from __future__ import annotations

import asyncio
import builtins

import httpx

from poecoder.cli import PoeCoderCLI, StreamEventError


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
    monkeypatch.setattr(cli, "_prompt_api_key", lambda: "")
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
    monkeypatch.setattr(cli, "_prompt_api_key", lambda: "")
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))
    monkeypatch.setattr(
        cli.http,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
    )

    cli.start()
    assert cli.direct is False


def test_cli_prompts_for_api_key_when_missing(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=True,
        model="assistant",
        lang="en",
    )
    cli.settings.poe_api_key = None
    cli.direct_model_client.api_key = None

    monkeypatch.setattr(cli, "_prompt_api_key", lambda: "poe-demo-key")
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "/quit")

    cli.start()
    assert cli.settings.poe_api_key == "poe-demo-key"
    assert cli.direct_model_client.api_key == "poe-demo-key"


def test_cli_runtime_event_loop_closed_is_handled(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    def fake_session(mode: str) -> None:
        cli.state.session_id = "s1"

    async def fail_turn(prompt: str, images=None):
        raise RuntimeError("Event loop is closed")

    responses = iter(["show your model", "/quit"])
    monkeypatch.setattr(cli, "_ensure_session", fake_session)
    monkeypatch.setattr(cli, "_prompt_api_key", lambda: "")
    monkeypatch.setattr(cli, "_run_backend_turn", fail_turn)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))

    cli.start()


def test_cli_stream_error_retries_nonstream_once(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )
    cli.state.session_id = "s1"

    async def fail_stream(async_http, payload):
        raise httpx.RemoteProtocolError("incomplete chunked read")

    async def fallback(async_http, payload):
        return {
            "session_id": "s1",
            "model": "assistant",
            "output_text": "fallback-ok",
            "tool_events": [],
        }

    monkeypatch.setattr(cli, "_stream_backend_turn", fail_stream)
    monkeypatch.setattr(cli, "_run_backend_turn_nonstream", fallback)

    asyncio.run(cli._run_backend_turn("hello"))
    assert cli.state.active_model == "assistant"


def test_cli_stream_event_error_does_not_retry_nonstream(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )
    cli.state.session_id = "s1"

    async def fail_stream(async_http, payload):
        raise StreamEventError("poe_non_sse_error", retryable=False)

    async def fail_if_called(async_http, payload):
        raise AssertionError("non-stream retry should not run for non-retryable stream event errors")

    monkeypatch.setattr(cli, "_stream_backend_turn", fail_stream)
    monkeypatch.setattr(cli, "_run_backend_turn_nonstream", fail_if_called)

    try:
        asyncio.run(cli._run_backend_turn("hello"))
    except StreamEventError:
        return
    raise AssertionError("expected StreamEventError")


def test_cli_prints_per_message_cost(monkeypatch, capsys) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    def fake_session(mode: str) -> None:
        cli.state.session_id = "s1"

    async def ok_turn(prompt: str, images=None):
        return None

    points = iter([1000, 996])
    responses = iter(["hi", "/quit"])

    monkeypatch.setattr(cli, "_ensure_session", fake_session)
    monkeypatch.setattr(cli, "_prompt_api_key", lambda: "")
    monkeypatch.setattr(cli, "_run_backend_turn", ok_turn)
    monkeypatch.setattr(cli, "_read_balance_points", lambda: next(points, None))
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))

    cli.start()
    out = capsys.readouterr().out
    assert "message cost=4 points (balance 1000->996)" in out
