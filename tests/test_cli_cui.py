from __future__ import annotations

from typing import Any

import pytest

from poecoder.cli import CUIApp


class StubApi:
    def __init__(self) -> None:
        self.saved_key: str | None = None
        self.loaded_key: str | None = None
        self.shell_command: str | None = None
        self.fail_load = False

    def save_secrets(self, user_key: str) -> dict[str, Any]:
        self.saved_key = user_key
        return {"path": "/tmp/mock-secrets.enc"}

    def load_secrets(self, user_key: str) -> dict[str, Any]:
        self.loaded_key = user_key
        if self.fail_load:
            raise ValueError(f"bad key: {user_key}")
        return {"poe_api_key_set": True, "openai_api_key_set": False}

    def run_shell(self, command: str, cwd: str = ".", timeout_s: int = 60, danger_ack: bool = False) -> dict[str, Any]:
        self.shell_command = command
        _ = (cwd, timeout_s, danger_ack)
        return {"allowed": True, "exit_code": 0, "stdout": "ok-line", "stderr": "", "duration_ms": 12}

    def turn_stream(self, session_id: str, prompt: str, model: str | None = None):
        _ = (session_id, prompt, model)
        yield {
            "event": "note",
            "data": {"progress": "planning", "detail": "Direct note output", "next": "run shell"},
        }
        yield {
            "event": "final",
            "data": {
                "output": "done",
                "status": "completed",
                "agent_metrics": {},
            },
        }


@pytest.fixture
def app(monkeypatch) -> CUIApp:
    monkeypatch.setattr(CUIApp, "_render", lambda self: None)
    return CUIApp(api=StubApi())  # type: ignore[arg-type]


def test_input_history_recall_and_restore_draft(app: CUIApp) -> None:
    app.input_history = ["first prompt", "second prompt", "third prompt"]
    app.input_buffer = "draft text"
    app._history_prev()
    assert app.input_buffer == "third prompt"
    app._history_prev()
    assert app.input_buffer == "second prompt"
    app._history_next()
    assert app.input_buffer == "third prompt"
    app._history_next()
    assert app.input_buffer == "draft text"


def test_sensitive_history_is_redacted(app: CUIApp) -> None:
    app._record_input_history("/secretsload super-secret-key")
    app._record_input_history("/authopenai sk-secret-token")
    assert app.input_history[0] == "/secretsload [REDACTED]"
    assert app.input_history[1] == "/authopenai [REDACTED]"
    joined = "\n".join(app.input_history)
    assert "super-secret-key" not in joined
    assert "sk-secret-token" not in joined


def test_tab_completion_for_commands_and_subcommands(app: CUIApp) -> None:
    app.input_buffer = "/prog"
    app._apply_tab_completion()
    assert app.input_buffer == "/progress"
    app.input_buffer = "/progress v"
    app._apply_tab_completion()
    assert app.input_buffer == "/progress verbose"


def test_progress_command_updates_mode(app: CUIApp) -> None:
    app.input_buffer = "/progress verbose"
    app._submit_input()
    assert app.progress_mode == "verbose"
    app.input_buffer = "/progress ???"
    app._submit_input()
    assert "Usage: /progress <compact|verbose>" in app.messages[-1]["content"]


def test_llm_off_blocks_normal_prompt(app: CUIApp) -> None:
    app.llm_enabled = False
    app.input_buffer = "hello"
    app._submit_input()
    assert "LLM mode is off" in app.messages[-1]["content"]


def test_runshell_command_executes_without_llm(app: CUIApp) -> None:
    api = app.api
    assert isinstance(api, StubApi)
    app.input_buffer = "/runshell printf ok"
    app._submit_input()
    assert api.shell_command == "printf ok"
    assert any("Proceeding: printf ok" in item["content"] for item in app.messages)
    assert any("[runshell] exit=0" in item["content"] for item in app.messages)
    assert app.command_viz
    assert app.command_viz[-1]["status"] == "OK"


def test_stream_note_is_rendered_as_assistant_output(app: CUIApp) -> None:
    app.session_id = "demo-session"
    payload = app._run_turn_stream("hello")
    assert payload.get("output") == "done"
    assistant_lines = [item["content"] for item in app.messages if item["role"] == "assistant"]
    assert any("Direct note output" in line for line in assistant_lines)


def test_stream_command_visualization_from_note_and_runshell(app: CUIApp) -> None:
    def turn_stream_with_shell(session_id: str, prompt: str, model: str | None = None):
        _ = (session_id, prompt, model)
        yield {"event": "started", "data": {"agent_id": "demo"}}
        yield {
            "event": "note",
            "data": {"progress": "running shell command", "detail": "in progress: apt update", "next": "wait"},
        }
        yield {
            "event": "runshell",
            "data": {"command": "apt update", "exit_code": 0, "allowed": True, "duration_ms": 1500},
        }
        yield {"event": "final", "data": {"output": "done", "status": "completed", "agent_metrics": {}}}

    app.api.turn_stream = turn_stream_with_shell  # type: ignore[method-assign]
    app.session_id = "demo-session"
    app._run_turn_stream("hello")
    assert any(item["command"] == "apt update" for item in app.command_viz)
    assert app.command_viz[-1]["status"] == "OK"


def test_secretsload_without_arg_starts_masked_prompt(app: CUIApp) -> None:
    app.input_buffer = "/secretsload"
    app._submit_input()
    assert app.secret_prompt_action == "load"
    assert app.popup_visible is True


def test_progress_modes_compact_filters_noise_verbose_shows_all(app: CUIApp) -> None:
    app.progress_mode = "compact"
    app._emit_stream_feedback("action", {"step": 1, "action": "plan", "progress": "Inspect files"})
    first_count = len([m for m in app.messages if m["role"] == "assistant_progress"])
    app._emit_stream_feedback("action", {"step": 2, "action": "plan", "progress": "Inspect tests"})
    second_count = len([m for m in app.messages if m["role"] == "assistant_progress"])
    app._emit_stream_feedback("note", {"progress": "risk detected", "detail": "possible failure"})
    final_progress = [m["content"] for m in app.messages if m["role"] == "assistant_progress"]
    assert first_count == 1
    assert second_count == 1
    assert any("[WARN][PLAN]" in line for line in final_progress)

    verbose = CUIApp(api=StubApi())  # type: ignore[arg-type]
    verbose.progress_mode = "verbose"
    verbose._emit_stream_feedback("action", {"step": 1, "action": "plan", "progress": "Inspect files"})
    verbose._emit_stream_feedback("action", {"step": 2, "action": "plan", "progress": "Inspect tests"})
    verbose_lines = [m["content"] for m in verbose.messages if m["role"] == "assistant_progress"]
    assert len(verbose_lines) == 2


def test_secret_prompt_submit_uses_masked_buffer_and_keeps_chat_clean(app: CUIApp) -> None:
    api = app.api
    assert isinstance(api, StubApi)
    app._start_secret_prompt("load")
    app.secret_input_buffer = "my-user-key"
    app._submit_secret_prompt()
    assert api.loaded_key == "my-user-key"
    assert app.secret_prompt_action is None
    assert app.secret_input_buffer == ""
    assert all("my-user-key" not in item["content"] for item in app.messages)


def test_secret_load_error_is_redacted(app: CUIApp) -> None:
    api = app.api
    assert isinstance(api, StubApi)
    api.fail_load = True
    app._command_secrets_load("leaky-value")
    assert app.messages
    assert "[REDACTED]" in app.messages[-1]["content"]
    assert "leaky-value" not in app.messages[-1]["content"]
