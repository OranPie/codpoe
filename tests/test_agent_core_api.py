from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import poecoder.api as api_module
from poecoder.app_state import build_app_state
from poecoder.services.model_clients import ModelReply


class StubModelClient:
    def __init__(self, reply_fn: Callable[..., dict[str, Any]] | None = None) -> None:
        self.reply_fn = reply_fn or (lambda **_: {"action": "final", "output": "ok"})

    async def chat(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str] | None = None,
    ) -> ModelReply:
        payload = self.reply_fn(
            model=model,
            system_message=system_message,
            user_prompt=user_prompt,
            context=context,
            images=images or [],
        )
        return ModelReply(text=json.dumps(payload, ensure_ascii=True), raw={"stub": True})

    def update_poe(self, api_key: str | None = None, base_url: str | None = None) -> None:  # noqa: ARG002
        return

    def update_openai(self, api_key: str | None = None, base_url: str | None = None) -> None:  # noqa: ARG002
        return

    async def fetch_full_model_catalog(
        self,
        *,
        seeded_models: list[str] | None = None,
        include_openai_remote: bool = True,  # noqa: ARG002
        remote_limit: int = 5000,  # noqa: ARG002
    ) -> dict[str, Any]:
        models = list(seeded_models or [])
        if "openai/gpt-5" not in models:
            models.append("openai/gpt-5")
        return {
            "models": models,
            "sources": {
                "seeded_count": len(seeded_models or []),
                "openai_remote_count": 1,
                "openai_remote_error": "",
            },
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POECODER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POECODER_DB_PATH", str(tmp_path / "agentcore.db"))
    state = build_app_state(tmp_path)
    state.runtime.model_client = StubModelClient()
    state.model_client = state.runtime.model_client  # type: ignore[assignment]
    api_module.STATE = state
    with TestClient(api_module.app) as test_client:
        yield test_client, state
    api_module.STATE = None


def test_session_turn_flow(client) -> None:
    test_client, _state = client
    session = test_client.post("/agent-api/sessions", json={"title": "demo"}).json()
    sid = session["id"]

    turn = test_client.post(
        f"/agent-api/sessions/{sid}/turn",
        json={"prompt": "hello"},
    )
    assert turn.status_code == 200
    payload = turn.json()
    assert payload["session_id"] == sid
    assert payload["output"] == "ok"
    assert isinstance(payload.get("agent_metrics"), dict)

    messages = test_client.get(f"/agent-api/sessions/{sid}/messages").json()
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hello"
    assert messages[1]["content"] == "ok"


def test_agent_start_wait_get_and_events(client) -> None:
    test_client, state = client

    def two_step_reply(**kwargs: Any) -> dict[str, Any]:
        step = int(kwargs["context"].get("step", 1))
        if step == 1:
            return {
                "action": "runshell",
                "progress": "run one command",
                "no_spawn_reason": "single short command is enough",
                "command": "printf hello",
                "cwd": ".",
                "timeout_s": 10,
            }
        return {"action": "final", "output": "done"}

    state.runtime.model_client = StubModelClient(reply_fn=two_step_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]

    created = test_client.post(
        "/agent-api/agents/start",
        json={"name": "two-step", "goal": "two-step run", "max_steps": 3},
    )
    assert created.status_code == 200
    agent_id = created.json()["id"]

    waited = test_client.post(f"/agent-api/agents/{agent_id}/wait", json={"timeout_s": 30})
    assert waited.status_code == 200
    agent_payload = waited.json()["agent"]
    assert agent_payload["status"] == "completed"
    assert str(agent_payload["final_output"]).startswith("done")

    events_resp = test_client.get(f"/agent-api/agents/{agent_id}/events?limit=50")
    assert events_resp.status_code == 200
    event_types = [item["event_type"] for item in events_resp.json()]
    assert "model_action" in event_types
    assert "note" in event_types
    assert "runshell" in event_types


def test_agent_start_allows_large_max_steps(client) -> None:
    test_client, _state = client
    created = test_client.post(
        "/agent-api/agents/start",
        json={"name": "many-steps", "goal": "finish quickly", "max_steps": 120},
    )
    assert created.status_code == 200
    agent_id = created.json()["id"]
    waited = test_client.post(f"/agent-api/agents/{agent_id}/wait", json={"timeout_s": 30})
    assert waited.status_code == 200
    assert waited.json()["agent"]["status"] == "completed"


def test_agent_depth_cap(client) -> None:
    test_client, state = client
    state.runtime.max_depth = 0

    root = test_client.post(
        "/agent-api/agents/start",
        json={"name": "root", "goal": "root goal", "max_steps": 2},
    ).json()
    root_id = root["id"]
    root_wait = test_client.post(f"/agent-api/agents/{root_id}/wait", json={"timeout_s": 30}).json()
    assert root_wait["agent"]["status"] in {"completed", "failed"}

    child = test_client.post(
        "/agent-api/agents/start",
        json={
            "name": "child",
            "goal": "child goal",
            "parent_agent_id": root_id,
            "max_steps": 2,
        },
    )
    assert child.status_code == 200
    child_id = child.json()["id"]

    child_wait = test_client.post(f"/agent-api/agents/{child_id}/wait", json={"timeout_s": 30})
    assert child_wait.status_code == 200
    child_agent = child_wait.json()["agent"]
    assert child_agent["status"] == "failed"
    assert "max depth exceeded" in child_agent["error"]


def test_run_shell_safety_and_workspace(client) -> None:
    test_client, _state = client

    ok = test_client.post("/agent-api/run-shell", json={"command": "printf safe", "cwd": "."})
    assert ok.status_code == 200
    assert ok.json()["allowed"] is True
    assert ok.json()["exit_code"] == 0

    blocked = test_client.post("/agent-api/run-shell", json={"command": "rm -rf /tmp/demo", "cwd": "."})
    assert blocked.status_code == 200
    assert blocked.json()["allowed"] is False
    assert "danger_ack" in blocked.json()["blocked_reason"]

    bad_cwd = test_client.post("/agent-api/run-shell", json={"command": "pwd", "cwd": "/tmp"})
    assert bad_cwd.status_code == 200
    assert bad_cwd.json()["allowed"] is False
    assert "outside workspace" in bad_cwd.json()["blocked_reason"]


def test_memory_and_templates(client) -> None:
    test_client, _state = client
    session = test_client.post("/agent-api/sessions", json={"title": "memory-demo"}).json()
    sid = session["id"]

    user_write = test_client.post(
        "/agent-api/memory/user/write",
        json={"scope": "user", "key": "pref.lang", "value": "zh", "tags": ["pref"]},
    )
    assert user_write.status_code == 200
    user_rows = test_client.post("/agent-api/memory/user/read", json={"scope": "user", "key": "pref.lang"}).json()
    assert user_rows[0]["value"] == "zh"

    session_write = test_client.post(
        "/agent-api/memory/session/write",
        json={"scope": "session", "session_id": sid, "key": "topic", "value": "arxiv"},
    )
    assert session_write.status_code == 200
    session_rows = test_client.post(
        "/agent-api/memory/session/read",
        json={"scope": "session", "session_id": sid, "key": "topic"},
    ).json()
    assert session_rows[0]["value"] == "arxiv"

    templates = test_client.get("/agent-api/agents/templates").json()
    names = {item["name"] for item in templates}
    assert {"shell-reader", "python-runner", "wget-downloader", "web-searcher"} <= names


def test_research_routes(client) -> None:
    test_client, state = client

    async def fake_search_web(**kwargs: Any) -> dict[str, Any]:
        return {"query": kwargs["query"], "count": 1, "results": [{"title": "t", "url": "u", "snippet": "s"}]}

    async def fake_search_arxiv(**kwargs: Any) -> dict[str, Any]:
        return {"query": kwargs["query"], "count": 1, "results": [{"title": "paper", "pdf_url": "https://x"}]}

    async def fake_get_web(**kwargs: Any) -> dict[str, Any]:
        return {"url": kwargs["url"], "text": "compact text", "truncated": False}

    async def fake_download_urls(**kwargs: Any) -> dict[str, Any]:
        return {"count": len(kwargs["urls"]), "success": len(kwargs["urls"]), "results": []}

    state.web_tools.search_web = fake_search_web  # type: ignore[assignment]
    state.web_tools.search_arxiv = fake_search_arxiv  # type: ignore[assignment]
    state.web_tools.get_web = fake_get_web  # type: ignore[assignment]
    state.web_tools.download_urls = fake_download_urls  # type: ignore[assignment]

    sw = test_client.post("/agent-api/research/search-web", json={"query": "rl 2048"}).json()
    assert sw["count"] == 1

    sa = test_client.post("/agent-api/research/search-arxiv", json={"query": "rl 2048"}).json()
    assert sa["results"][0]["title"] == "paper"

    gw = test_client.post("/agent-api/research/get-web", json={"url": "https://example.com"}).json()
    assert gw["truncated"] is False

    dl = test_client.post(
        "/agent-api/research/download-urls",
        json={"urls": ["https://example.com/a.pdf", "https://example.com/b.pdf"]},
    ).json()
    assert dl["success"] == 2


def test_secret_load_compatibility(client) -> None:
    test_client, state = client
    payload = {
        "poe_api_key": "poe-secret",
        "openai_api_key": "openai-secret",
        "poe_api_url": "https://api.poe.com/bot/",
        "openai_api_url": "https://api.openai.com/v1",
    }
    state.provider_secrets.save("demo-user-key", payload)

    loaded = test_client.post("/agent-api/auth/secrets/load", json={"user_key": "demo-user-key"})
    assert loaded.status_code == 200
    data = loaded.json()
    assert data["ok"] is True
    assert data["poe_api_key_set"] is True
    assert data["openai_api_key_set"] is True
    assert data["poe_api_url"] == "https://api.poe.com/bot/"
    assert data["openai_api_url"] == "https://api.openai.com/v1"

    # legacy alias remains available for earlier clients
    legacy = test_client.post("/auth/secrets/load", json={"user_key": "demo-user-key"})
    assert legacy.status_code == 200


def test_models_query_endpoint(client) -> None:
    test_client, _state = client
    all_models = test_client.get("/agent-api/models")
    assert all_models.status_code == 200
    payload = all_models.json()
    assert isinstance(payload.get("models"), list)
    assert payload.get("count", 0) >= 1

    filtered = test_client.get("/agent-api/models", params={"query": "assistant", "limit": 10})
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert isinstance(filtered_payload.get("models"), list)
    for name in filtered_payload.get("models", []):
        assert "assistant" in str(name).lower()

    full = test_client.get("/agent-api/models", params={"full": "true", "query": "openai/"})
    assert full.status_code == 200
    full_payload = full.json()
    assert full_payload.get("source_meta", {}).get("mode") == "full"
    assert any("openai/" in str(name).lower() for name in full_payload.get("models", []))


def test_session_turn_stream_endpoint(client) -> None:
    test_client, _state = client
    session = test_client.post("/agent-api/sessions", json={"title": "stream-demo"}).json()
    sid = session["id"]

    resp = test_client.post(f"/agent-api/sessions/{sid}/turn/stream", json={"prompt": "hello stream"})
    assert resp.status_code == 200
    body = resp.text
    assert "event: started" in body
    assert "event: action" in body
    assert "event: final" in body


def test_session_turn_ask_response(client) -> None:
    test_client, state = client

    def ask_reply(**kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {
            "action": "ask",
            "progress": "need user choice",
            "question": "Which run mode do you want?",
            "input_mode": "single",
            "options": [
                {"id": "fast", "label": "Fast summary"},
                {"id": "deep", "label": "Deep analysis"},
            ],
        }

    state.runtime.model_client = StubModelClient(reply_fn=ask_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]
    session = test_client.post("/agent-api/sessions", json={"title": "ask-demo"}).json()
    sid = session["id"]

    turn = test_client.post(f"/agent-api/sessions/{sid}/turn", json={"prompt": "start"}).json()
    ask = turn.get("ask")
    assert isinstance(ask, dict)
    assert ask.get("question") == "Which run mode do you want?"
    assert ask.get("input_mode") == "single"
    assert "[ASK]" in str(turn.get("output", ""))


def test_session_turn_stream_ask_event(client) -> None:
    test_client, state = client

    def ask_reply(**kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {
            "action": "ask",
            "progress": "need clarification",
            "question": "Pick one source",
            "input_mode": "single",
            "options": [
                {"id": "arxiv", "label": "arXiv"},
                {"id": "web", "label": "General web"},
            ],
        }

    state.runtime.model_client = StubModelClient(reply_fn=ask_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]
    session = test_client.post("/agent-api/sessions", json={"title": "stream-ask-demo"}).json()
    sid = session["id"]

    resp = test_client.post(f"/agent-api/sessions/{sid}/turn/stream", json={"prompt": "start"}).text
    assert "event: ask" in resp
    assert '"question": "Pick one source"' in resp


def test_session_turn_note_middle_feedback(client) -> None:
    test_client, state = client

    def note_reply(**kwargs: Any) -> dict[str, Any]:
        step = int(kwargs["context"].get("step", 1))
        if step == 1:
            return {
                "action": "note",
                "progress": "planning execution",
                "detail": "Will inspect file list first, then summarize findings.",
                "next": "run focused file discovery",
            }
        return {"action": "final", "output": "done"}

    state.runtime.model_client = StubModelClient(reply_fn=note_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]
    session = test_client.post("/agent-api/sessions", json={"title": "note-demo"}).json()
    sid = session["id"]

    turn = test_client.post(f"/agent-api/sessions/{sid}/turn", json={"prompt": "start"}).json()
    assert turn.get("output") == "done"
    assert isinstance(turn.get("steps"), list)
    assert any("action=note" == str(item) for item in turn.get("steps", []))
    metrics = turn.get("agent_metrics", {})
    assert int(metrics.get("note_count", 0) or 0) >= 1

    stream_text = test_client.post(f"/agent-api/sessions/{sid}/turn/stream", json={"prompt": "start"}).text
    assert "event: note" in stream_text
    assert '"detail": "Will inspect file list first, then summarize findings."' in stream_text


def test_note_becomes_output_when_max_steps_hit(client) -> None:
    test_client, state = client

    def note_only_reply(**kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {
            "action": "note",
            "progress": "analyzing",
            "detail": "Need one more step to finish.",
            "next": "rerun with larger max_steps",
        }

    state.runtime.model_client = StubModelClient(reply_fn=note_only_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]

    created = test_client.post(
        "/agent-api/agents/start",
        json={"name": "note-only", "goal": "note only", "max_steps": 1},
    )
    assert created.status_code == 200
    agent_id = created.json()["id"]
    waited = test_client.post(f"/agent-api/agents/{agent_id}/wait", json={"timeout_s": 30})
    assert waited.status_code == 200
    agent_payload = waited.json()["agent"]
    assert agent_payload["status"] == "completed"
    assert "Need one more step to finish." in str(agent_payload["final_output"])


def test_cancel_agent_endpoint(client) -> None:
    test_client, state = client

    def long_reply(**kwargs: Any) -> dict[str, Any]:
        step = int(kwargs["context"].get("step", 1))
        if step <= 4:
            return {
                "action": "runshell",
                "progress": f"long step {step}",
                "no_spawn_reason": "single controlled long task",
                "command": "sleep 2",
                "cwd": ".",
                "timeout_s": 10,
            }
        return {"action": "final", "output": "done"}

    state.runtime.model_client = StubModelClient(reply_fn=long_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]

    created = test_client.post(
        "/agent-api/agents/start",
        json={"name": "cancel-me", "goal": "long run", "max_steps": 8},
    ).json()
    agent_id = created["id"]

    cancelled = test_client.post(f"/agent-api/agents/{agent_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    after = test_client.get(f"/agent-api/agents/{agent_id}").json()
    assert after["agent"]["status"] == "cancelled"


def test_agent_define_and_call_tool(client) -> None:
    test_client, state = client
    seen_contexts: list[dict[str, Any]] = []

    def tool_reply(**kwargs: Any) -> dict[str, Any]:
        context = kwargs.get("context", {})
        if isinstance(context, dict):
            seen_contexts.append(context)
        step = int(context.get("step", 1) if isinstance(context, dict) else 1)
        if step == 1:
            return {
                "action": "define_tool",
                "progress": "create reusable printf tool",
                "name": "echo_word",
                "language": "sh",
                "description": "print one word",
                "script": "printf {{word}}",
                "args_schema": {"word": "single token to print"},
            }
        if step == 2:
            return {
                "action": "call_tool",
                "progress": "run the reusable tool",
                "name": "echo_word",
                "args": {"word": "hello"},
                "cwd": ".",
                "timeout_s": 10,
            }
        return {"action": "final", "output": "done"}

    state.runtime.model_client = StubModelClient(reply_fn=tool_reply)
    state.model_client = state.runtime.model_client  # type: ignore[assignment]

    session = test_client.post("/agent-api/sessions", json={"title": "tool-demo"}).json()
    sid = session["id"]
    created = test_client.post(
        "/agent-api/agents/start",
        json={"name": "tool-agent", "goal": "define and call tool", "session_id": sid, "max_steps": 5},
    )
    agent_id = created.json()["id"]

    waited = test_client.post(f"/agent-api/agents/{agent_id}/wait", json={"timeout_s": 30}).json()
    agent_payload = waited["agent"]
    assert agent_payload["status"] == "completed"
    assert "done" in str(agent_payload.get("final_output", ""))
    assert "tool:echo_word" in str(agent_payload.get("final_output", ""))

    events = test_client.get(f"/agent-api/agents/{agent_id}/events?limit=80").json()
    event_types = [item.get("event_type") for item in events]
    assert "tool_define" in event_types
    assert "tool_call" in event_types

    assert len(seen_contexts) >= 2
    tools_step2 = seen_contexts[1].get("tools", [])
    assert isinstance(tools_step2, list)
    assert any(str(item.get("name", "")) == "echo_word" for item in tools_step2 if isinstance(item, dict))
