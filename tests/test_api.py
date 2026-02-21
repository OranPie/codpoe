from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient


def test_session_turn_and_memory(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)

    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    assert session_resp.status_code == 200
    session_id = session_resp.json()["id"]

    turn_resp = client.post(
        "/turns/execute",
        json={
            "session_id": session_id,
            "user_prompt": "hello",
            "system_message": "You are helpful",
        },
    )
    assert turn_resp.status_code == 200
    payload = turn_resp.json()
    assert payload["session_id"] == session_id
    assert "mock" in payload["output_text"]

    mem_write = client.post(
        "/memory/write",
        json={
            "scope": "session",
            "session_id": session_id,
            "project_id": "demo",
            "content": "remember this",
            "tags": ["note"],
            "priority": 1,
        },
    )
    assert mem_write.status_code == 200

    mem_read = client.post(
        "/memory/read",
        json={"scope": "session", "session_id": session_id, "limit": 5},
    )
    assert mem_read.status_code == 200
    rows = mem_read.json()
    assert len(rows) == 1
    assert rows[0]["content"] == "remember this"


def test_tool_read_raw_endpoint(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))
    target = tmp_path / "demo.txt"
    target.write_text("a\nb\nc\n", encoding="utf-8")

    from poecoder import api

    api.STATE = None
    api.STATE = api.build_app_state(tmp_path)
    client = TestClient(api.app)

    resp = client.post("/tools/read_raw", json={"file": "demo.txt", "line": 2})
    assert resp.status_code == 200
    assert "b" in resp.json()["content"]


def test_listmodels_and_changemodel(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))
    monkeypatch.setenv("POECODER_MODELS", "gpt-4o-mini,gpt-4.1,sonnet")

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)

    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    assert "sonnet" in models_resp.json()["models"]

    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    session_id = session_resp.json()["id"]

    change_resp = client.post(f"/sessions/{session_id}/change-model", json={"model": "sonnet"})
    assert change_resp.status_code == 200
    assert change_resp.json()["active_model"] == "sonnet"

    tool_resp = client.post(
        "/tools/invoke",
        json={"name": "ListModels", "args": {}},
    )
    assert tool_resp.status_code == 200
    assert "gpt-4.1" in tool_resp.json()["result"]["models"]


def test_usage_balance_and_tool(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = api.build_app_state(tmp_path)

    monkeypatch.setattr(
        type(api.STATE.usage),
        "get_current_balance",
        lambda self: {"current_point_balance": 12345, "fetched_at": "now", "source": "mock"},
    )

    client = TestClient(api.app)

    bal = client.get("/usage/current_balance")
    assert bal.status_code == 200
    assert bal.json()["current_point_balance"] == 12345

    tool = client.post("/tools/invoke", json={"name": "GetBalance", "args": {}})
    assert tool.status_code == 200
    assert tool.json()["result"]["current_point_balance"] == 12345


def test_turn_model_tool_communication(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.services.model_clients import ModelReply

    api.STATE = api.build_app_state(tmp_path)

    monkeypatch.setattr(
        type(api.STATE.usage),
        "get_current_balance",
        lambda self: {"current_point_balance": 321, "fetched_at": "now", "source": "mock"},
    )

    call_count = {"n": 0}

    async def fake_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ModelReply(text='@tool GetBalance {}', raw={"mock": True})
        assert "Tool results:" in user_prompt
        assert "current_point_balance" in user_prompt
        return ModelReply(text="balance checked", raw={"mock": True})

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fake_chat)

    client = TestClient(api.app)
    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    session_id = session_resp.json()["id"]

    turn_resp = client.post(
        "/turns/execute",
        json={
            "session_id": session_id,
            "user_prompt": "check balance",
        },
    )
    assert turn_resp.status_code == 200
    payload = turn_resp.json()
    assert payload["output_text"] == "balance checked"
    assert call_count["n"] == 2
    assert len(payload["tool_events"]) == 1
    assert payload["tool_events"][0]["name"] == "GetBalance"
    assert payload["tool_events"][0]["result"]["current_point_balance"] == 321


def test_turn_stream_emits_delta_and_final(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = api.build_app_state(tmp_path)

    async def fake_stream(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        yield "hello "
        yield "world"

    async def fail_chat(*args, **kwargs):
        raise AssertionError("chat should not be used for no-tool stream path")

    monkeypatch.setattr(api.STATE.turns.model_client, "chat_stream", fake_stream)
    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fail_chat)

    client = TestClient(api.app)
    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    session_id = session_resp.json()["id"]

    events = []
    with client.stream(
        "POST",
        "/turns/execute/stream",
        json={"session_id": session_id, "user_prompt": "say hi"},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            events.append(json.loads(line[6:]))

    deltas = [e["data"] for e in events if e.get("type") == "delta"]
    assert "".join(deltas) == "hello world"
    finals = [e for e in events if e.get("type") == "final"]
    assert len(finals) == 1
    assert finals[0]["data"]["output_text"] == "hello world"


def test_background_turn_task_lifecycle(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.models import TurnResult

    api.STATE = api.build_app_state(tmp_path)

    async def fake_execute(self, req):
        return TurnResult(
            session_id=req.session_id,
            model="assistant",
            output_text="background-ok",
            tool_events=[],
        )

    monkeypatch.setattr(type(api.STATE.turns), "execute", fake_execute)

    client = TestClient(api.app)
    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    session_id = session_resp.json()["id"]

    start = client.post(
        "/tasks/turns/start",
        json={"session_id": session_id, "user_prompt": "run bg"},
    )
    assert start.status_code == 200
    task_id = start.json()["id"]

    state = None
    payload = {}
    for _ in range(40):
        r = client.get(f"/tasks/{task_id}")
        assert r.status_code == 200
        payload = r.json()
        state = payload["state"]
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert state == "completed"
    assert payload["result"]["output_text"] == "background-ok"


def test_background_subagent_task_lifecycle(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = api.build_app_state(tmp_path)

    def fake_start(
        self,
        parent_session_id: str,
        model: str,
        perm: str,
        prompt: str,
        context_share: list[str],
        images=None,
        system_message_modifier=None,
    ):
        return {
            "id": "agent-1",
            "parent_session_id": parent_session_id,
            "model": model,
            "perm": perm,
            "prompt": prompt,
            "images": images or [],
            "state": "running",
            "result": None,
            "shared_context": context_share,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }

    async def fake_wait(self, agent_id: str, timeout_s: int = 60):
        return {
            "id": agent_id,
            "state": "completed",
            "result": "subagent-ok",
        }

    monkeypatch.setattr(type(api.STATE.subagents), "start", fake_start)
    monkeypatch.setattr(type(api.STATE.subagents), "wait", fake_wait)

    client = TestClient(api.app)
    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    session_id = session_resp.json()["id"]

    start = client.post(
        "/tasks/subagents/start",
        json={
            "parent_session_id": session_id,
            "model": "assistant",
            "perm": "readonly",
            "prompt": "do it",
            "context_share": [],
            "wait_timeout_s": 10,
        },
    )
    assert start.status_code == 200
    task_id = start.json()["id"]

    payload = {}
    state = None
    for _ in range(40):
        r = client.get(f"/tasks/{task_id}")
        assert r.status_code == 200
        payload = r.json()
        state = payload["state"]
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert state == "completed"
    assert payload["result"]["subagent_id"] == "agent-1"
    assert payload["result"]["subagent"]["result"] == "subagent-ok"


def test_task_output_endpoint(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.models import TurnResult

    api.STATE = api.build_app_state(tmp_path)

    async def fake_execute(self, req):
        return TurnResult(
            session_id=req.session_id,
            model="assistant",
            output_text="done",
            tool_events=[],
        )

    monkeypatch.setattr(type(api.STATE.turns), "execute", fake_execute)
    client = TestClient(api.app)

    session_resp = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    session_id = session_resp.json()["id"]
    started = client.post("/tasks/turns/start", json={"session_id": session_id, "user_prompt": "bg"})
    task_id = started.json()["id"]

    state = "queued"
    for _ in range(50):
        output = client.get(f"/tasks/{task_id}/output")
        assert output.status_code == 200
        state = output.json()["state"]
        if state == "completed":
            assert output.json()["result"]["output_text"] == "done"
            break
        time.sleep(0.02)
    assert state == "completed"


def test_planning_mode_uses_plan_system_message(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.prompts import PLAN_SYSTEM_MESSAGE
    from poecoder.services.model_clients import ModelReply

    api.STATE = api.build_app_state(tmp_path)
    captured: dict[str, str] = {}

    async def fake_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        captured["system_message"] = system_message
        return ModelReply(text="ok", raw={"mock": True})

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fake_chat)

    client = TestClient(api.app)
    session_resp = client.post("/sessions", json={"mode": "planning", "project_id": "demo"})
    session_id = session_resp.json()["id"]
    turn = client.post("/turns/execute", json={"session_id": session_id, "user_prompt": "plan this"})
    assert turn.status_code == 200
    assert captured["system_message"] == PLAN_SYSTEM_MESSAGE
