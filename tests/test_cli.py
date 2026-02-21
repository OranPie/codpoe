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


def test_cli_loginopenai_uses_openai_prompt(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=True,
        model="assistant",
        lang="en",
    )

    monkeypatch.setattr(cli, "_prompt_openai_api_key", lambda: "oa-demo-key")
    monkeypatch.setattr(cli, "_prompt_api_key", lambda: (_ for _ in ()).throw(AssertionError("wrong prompt")))

    handled = cli._handle_command("/loginopenai")
    assert handled is False
    assert cli.settings.openai_api_key == "oa-demo-key"
    assert cli.direct_model_client.openai_api_key == "oa-demo-key"


def test_cli_secretssave_calls_backend(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )
    captured: dict[str, object] = {}

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"ok": True, "path": "/tmp/provider_secrets.enc.json"}

    monkeypatch.setattr(cli, "_prompt_user_key", lambda: "user-pass")

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    monkeypatch.setattr(cli.http, "post", fake_post)
    handled = cli._handle_command("/secretssave")
    assert handled is False
    assert str(captured["url"]).endswith("/auth/secrets/save")
    assert captured["json"] == {"user_key": "user-pass"}


def test_cli_secretsload_updates_base_urls(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "ok": True,
                "poe_api_key_set": True,
                "openai_api_key_set": True,
                "poe_api_url": "https://api.poe.com/bot/",
                "openai_api_url": "https://openai.proxy/v1",
            }

    monkeypatch.setattr(cli, "_prompt_user_key", lambda: "user-pass")
    monkeypatch.setattr(cli.http, "post", lambda *_args, **_kwargs: Resp())

    handled = cli._handle_command("/secretsload")
    assert handled is False
    assert cli.settings.poe_api_url == "https://api.poe.com/bot/"
    assert cli.settings.openai_api_url == "https://openai.proxy/v1"
    assert cli.direct_model_client.api_url == "https://api.poe.com/bot/"
    assert cli.direct_model_client.openai_api_url == "https://openai.proxy/v1"


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


def test_cli_balance_prints_openai_details(monkeypatch, capsys) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "current_point_balance": 12,
                "openai": {
                    "api_key_configured": True,
                    "base_url": "https://api.openai.com/v1",
                    "model_count": 42,
                },
            }

    monkeypatch.setattr(cli.http, "get", lambda *_args, **_kwargs: Resp())
    handled = cli._handle_command("/balance")
    assert handled is False
    out = capsys.readouterr().out
    assert "Current balance: 12 points" in out
    assert "OpenAI: key_set=true" in out


def test_cli_apistatus_backend(monkeypatch, capsys) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "poe_api_key_set": True,
                "openai_api_key_set": False,
                "poe_api_url": "https://api.poe.com/bot/",
                "openai_api_url": "https://api.openai.com/v1",
            }

    monkeypatch.setattr(cli.http, "get", lambda *_args, **_kwargs: Resp())
    handled = cli._handle_command("/apistatus")
    assert handled is False
    out = capsys.readouterr().out
    assert "\"poe_api_key_set\": true" in out


def test_cli_resume_by_index_updates_session_state(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )
    cli.state.project_id = "demo"

    class Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    monkeypatch.setattr(
        cli.http,
        "get",
        lambda *args, **kwargs: Resp(
            [
                {
                    "id": "sess-1",
                    "title": "Resume target",
                    "mode": "coding",
                    "active_model": "gpt-5.2-codex",
                    "thinking_level": "balanced",
                    "thinking_budget": 12000,
                    "allow_model_command_create": True,
                    "encourage_model_command_create": True,
                    "project_id": "demo",
                }
            ]
        ),
    )

    handled = cli._handle_command("/resume 1")
    assert handled is False
    assert cli.state.session_id == "sess-1"
    assert cli.state.session_title == "Resume target"
    assert cli.state.active_model == "gpt-5.2-codex"


def test_cli_thinkdetails_updates_backend_state(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )
    cli.state.session_id = "s1"

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"id": "s1", "show_think_details": True}

    captured: dict[str, object] = {}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    monkeypatch.setattr(cli.http, "post", fake_post)
    handled = cli._handle_command("/thinkdetails on")
    assert handled is False
    assert cli.state.show_think_details is True
    assert str(captured["url"]).endswith("/sessions/s1/think-details")


def test_cli_setbaseuri_updates_direct_client(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=True,
        model="assistant",
        lang="en",
    )

    handled = cli._handle_command("/setbaseuri poe https://proxy.poe.local/api")
    assert handled is False
    assert cli.settings.poe_api_url == "https://proxy.poe.local/api"
    assert cli.direct_model_client.api_url == "https://proxy.poe.local/api/"

    handled2 = cli._handle_command("/setbaseuri openai https://openai.proxy/v1/")
    assert handled2 is False
    assert cli.settings.openai_api_url == "https://openai.proxy/v1/"
    assert cli.direct_model_client.openai_api_url == "https://openai.proxy/v1"


def test_cli_setbaseuri_updates_backend(monkeypatch) -> None:
    cli = PoeCoderCLI(
        backend_url="http://127.0.0.1:8765",
        direct=False,
        model="assistant",
        lang="en",
    )
    cli.state.session_id = "s1"
    captured: dict[str, object] = {}

    class Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"ok": True, "base_url": "https://api.poe.com/bot/"}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return Resp()

    monkeypatch.setattr(cli.http, "post", fake_post)
    handled = cli._handle_command("/setbaseuri poe https://api.poe.com")
    assert handled is False
    assert str(captured["url"]).endswith("/providers/poe/base-url")
    assert captured["json"] == {"base_url": "https://api.poe.com"}
