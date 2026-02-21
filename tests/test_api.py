from __future__ import annotations

import json
import time
import asyncio

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


def test_session_title_and_list_sessions(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.services.model_clients import ModelReply

    api.STATE = api.build_app_state(tmp_path)

    async def fake_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        return ModelReply(text="Implemented resume flow and session listing.", raw={"mock": True})

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fake_chat)

    client = TestClient(api.app)
    created = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    turn = client.post("/turns/execute", json={"session_id": session_id, "user_prompt": "do work"})
    assert turn.status_code == 200

    session = client.get(f"/sessions/{session_id}")
    assert session.status_code == 200
    assert session.json()["title"].startswith("Implemented resume flow")

    listed = client.get("/sessions", params={"project_id": "demo", "limit": "10"})
    assert listed.status_code == 200
    rows = listed.json()
    assert isinstance(rows, list)
    assert any(item["id"] == session_id for item in rows)


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


def test_tool_listfile_changeworkdir_and_exit(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b\n", encoding="utf-8")

    from poecoder import api

    api.STATE = api.build_app_state(tmp_path)
    client = TestClient(api.app)

    listed = client.post("/tools/invoke", json={"name": "ListFile", "args": {"pattern": "*.txt", "recursive": True}})
    assert listed.status_code == 200
    entries = listed.json()["result"]["entries"]
    assert "a.txt" in entries
    assert "sub/b.txt" in entries

    changed = client.post("/tools/invoke", json={"name": "ChangeWorkDir", "args": {"path": "sub"}})
    assert changed.status_code == 200
    assert changed.json()["result"]["cwd"] == "sub"

    listed_sub = client.post("/tools/invoke", json={"name": "ListFile", "args": {"pattern": "*.txt"}})
    assert listed_sub.status_code == 200
    assert listed_sub.json()["result"]["entries"] == ["sub/b.txt"]

    exited = client.post("/tools/invoke", json={"name": "Exit", "args": {"reason": "done"}})
    assert exited.status_code == 200
    assert exited.json()["result"]["exit"] is True
    assert exited.json()["result"]["reason"] == "done"


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


def test_api_login_updates_runtime_keys(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))
    monkeypatch.delenv("POECODER_POE_API_KEY", raising=False)

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)

    resp = client.post("/auth/poe/login", json={"api_key": "demo-key"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert api.STATE is not None
    assert api.STATE.settings.poe_api_key == "demo-key"
    assert api.STATE.turns.model_client.api_key == "demo-key"
    assert api.STATE.subagents.model_client.api_key == "demo-key"
    assert api.STATE.reviews.model_client.api_key == "demo-key"
    assert api.STATE.model_catalog.api_key == "demo-key"
    assert api.STATE.usage.api_key == "demo-key"


def test_openai_login_and_base_url_update(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))
    monkeypatch.delenv("POECODER_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("POECODER_OPENAI_API_URL", "https://openai.example/v1")

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)

    login = client.post("/auth/openai/login", json={"api_key": "oa-demo-key"})
    assert login.status_code == 200
    assert login.json()["ok"] is True
    assert api.STATE is not None
    assert api.STATE.settings.openai_api_key == "oa-demo-key"
    assert api.STATE.turns.model_client.openai_api_key == "oa-demo-key"
    assert api.STATE.subagents.model_client.openai_api_key == "oa-demo-key"
    assert api.STATE.reviews.model_client.openai_api_key == "oa-demo-key"

    base_url = client.post("/providers/openai/base-url", json={"base_url": "https://proxy.openai.local/v1/"})
    assert base_url.status_code == 200
    assert base_url.json()["base_url"] == "https://proxy.openai.local/v1"
    assert api.STATE.settings.openai_api_url == "https://proxy.openai.local/v1/"


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
            assert context["turn_protocol"]["phase"] == "tool_or_answer"
            return ModelReply(text='@tool GetBalance {}', raw={"mock": True})
        assert "Tool results:" in user_prompt
        assert "current_point_balance" in user_prompt
        assert "final-response turn" in user_prompt
        assert context["turn_protocol"]["phase"] == "final_response"
        assert context["turn_protocol"]["tool_events_count"] == 1
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


def test_turn_execute_maps_provider_errors(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.services.model_clients import ModelProviderError

    api.STATE = api.build_app_state(tmp_path)

    async def fail_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        raise ModelProviderError(
            model=model,
            code="poe_non_sse_error",
            detail="Poe returned non-SSE response",
            hint="check model and key",
            http_status=502,
        )

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fail_chat)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]
    turn_resp = client.post(
        "/turns/execute",
        json={"session_id": session_id, "user_prompt": "hello"},
    )
    assert turn_resp.status_code == 502
    detail = turn_resp.json()["detail"]
    assert detail["code"] == "poe_non_sse_error"
    assert "non-SSE" in detail["detail"]


def test_turn_repair_retry_for_incomplete_first_reply(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.services.model_clients import ModelReply

    api.STATE = api.build_app_state(tmp_path)

    monkeypatch.setattr(
        type(api.STATE.usage),
        "get_current_balance",
        lambda self: {"current_point_balance": 777, "fetched_at": "now", "source": "mock"},
    )

    call_count = {"n": 0}

    async def fake_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ModelReply(
                text="Generating...Generating...Generating...Generating... (2s elapsed)Generating... (3s elapsed)",
                raw={"mock": True},
            )
        if call_count["n"] == 2:
            assert "Previous response looked incomplete" in user_prompt
            assert context["turn_protocol"]["repair_attempt"] == 1
            return ModelReply(text='@tool GetBalance {}', raw={"mock": True})
        assert "Tool results:" in user_prompt
        return ModelReply(text="done-after-repair", raw={"mock": True})

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fake_chat)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]
    turn_resp = client.post("/turns/execute", json={"session_id": session_id, "user_prompt": "check balance"})
    assert turn_resp.status_code == 200
    payload = turn_resp.json()
    assert payload["output_text"] == "done-after-repair"
    assert call_count["n"] == 3
    assert payload["tool_events"][0]["name"] == "GetBalance"


def test_turn_stream_emits_error_event_on_provider_failure(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.services.model_clients import ModelProviderError

    api.STATE = api.build_app_state(tmp_path)

    async def fail_stream(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        raise ModelProviderError(
            model=model,
            code="poe_non_sse_error",
            detail="Poe returned non-SSE response",
            hint="check model and key",
            http_status=502,
        )
        yield ""  # pragma: no cover

    monkeypatch.setattr(api.STATE.turns.model_client, "chat_stream", fail_stream)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]

    events: list[dict[str, object]] = []
    with client.stream(
        "POST",
        "/turns/execute/stream",
        json={"session_id": session_id, "user_prompt": "hello"},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    errors = [event["data"] for event in events if event.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "poe_non_sse_error"


def test_turn_stream_emits_error_event_on_internal_failure(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = api.build_app_state(tmp_path)

    async def broken_execute_stream(self, req):
        raise RuntimeError("boom-stream")
        yield {"type": "status", "data": "never"}  # pragma: no cover

    monkeypatch.setattr(type(api.STATE.turns), "execute_stream", broken_execute_stream)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]
    events: list[dict[str, object]] = []
    with client.stream(
        "POST",
        "/turns/execute/stream",
        json={"session_id": session_id, "user_prompt": "hello"},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    errors = [event["data"] for event in events if event.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["code"] == "stream_internal_error"
    assert errors[0]["retryable"] is True


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
    assert captured["system_message"].startswith(PLAN_SYSTEM_MESSAGE)


def test_leader_mode_uses_leader_system_message(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.prompts import LEADER_SYSTEM_MESSAGE
    from poecoder.services.model_clients import ModelReply

    api.STATE = api.build_app_state(tmp_path)
    captured: dict[str, str] = {}

    async def fake_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        captured["system_message"] = system_message
        return ModelReply(text="ok", raw={"mock": True})

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fake_chat)

    client = TestClient(api.app)
    session_resp = client.post("/sessions", json={"mode": "leader", "project_id": "demo"})
    session_id = session_resp.json()["id"]
    turn = client.post("/turns/execute", json={"session_id": session_id, "user_prompt": "coordinate this"})
    assert turn.status_code == 200
    assert captured["system_message"].startswith(LEADER_SYSTEM_MESSAGE)


def test_leader_run_lifecycle_with_scoped_jobs(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.models import TurnResult

    api.STATE = api.build_app_state(tmp_path)
    captured: list[dict[str, object]] = []

    async def fake_execute(self, req):
        captured.append(
            {
                "system_message": req.system_message,
                "metadata": req.metadata,
                "user_prompt": req.user_prompt,
            }
        )
        return TurnResult(
            session_id=req.session_id,
            model="assistant",
            output_text=f"done-{req.metadata.get('leader_job_id', 'x')}",
            tool_events=[],
        )

    monkeypatch.setattr(type(api.STATE.turns), "execute", fake_execute)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]
    started = client.post(
        "/leader/start",
        json={
            "session_id": session_id,
            "goal": "implement two isolated updates",
            "jobs": [
                {
                    "name": "api",
                    "objective": "update api",
                    "scope": "api only",
                    "owned_paths": ["poecoder/api.py"],
                    "context_keys": [],
                },
                {
                    "name": "cli",
                    "objective": "update cli",
                    "scope": "cli only",
                    "owned_paths": ["poecoder/cli.py"],
                    "context_keys": [],
                },
            ],
            "max_parallel": 2,
        },
    )
    assert started.status_code == 200
    run_id = started.json()["id"]

    waited = client.post(f"/leader/{run_id}/wait", json={"timeout_s": 5})
    assert waited.status_code == 200
    assert waited.json()["state"] == "completed"

    jobs = client.get(f"/leader/{run_id}/jobs")
    assert jobs.status_code == 200
    rows = jobs.json()
    assert len(rows) == 2
    assert all(item["state"] == "completed" for item in rows)

    assert len(captured) == 2
    for entry in captured:
        metadata = entry["metadata"]
        assert metadata["isolation_rule"] == "do-not-touch-outside-owned-paths"
        assert "Owned scope:" in str(entry["system_message"])


def test_leader_run_cancel(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.models import TurnResult

    api.STATE = api.build_app_state(tmp_path)

    async def slow_execute(self, req):
        await asyncio.sleep(0.5)
        return TurnResult(
            session_id=req.session_id,
            model="assistant",
            output_text="slow",
            tool_events=[],
        )

    monkeypatch.setattr(type(api.STATE.turns), "execute", slow_execute)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]
    started = client.post(
        "/leader/start",
        json={
            "session_id": session_id,
            "goal": "slow run",
            "jobs": [
                {
                    "name": "one",
                    "objective": "slow",
                    "scope": "single",
                    "owned_paths": ["poecoder/services"],
                    "context_keys": [],
                }
            ],
        },
    )
    run_id = started.json()["id"]
    cancelled = client.post(f"/leader/{run_id}/cancel")
    assert cancelled.status_code == 200
    state = cancelled.json()["state"]
    for _ in range(40):
        if state == "cancelled":
            break
        snapshot = client.get(f"/leader/{run_id}")
        assert snapshot.status_code == 200
        state = snapshot.json()["state"]
        time.sleep(0.02)
    assert state == "cancelled"


def test_session_thinking_update(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)

    created = client.post("/sessions", json={"mode": "coding", "project_id": "demo"})
    assert created.status_code == 200
    session_id = created.json()["id"]

    updated = client.post(
        f"/sessions/{session_id}/thinking",
        json={"thinking_level": "deep", "thinking_budget": 24000},
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["thinking_level"] == "deep"
    assert payload["thinking_budget"] == 24000


def test_session_think_details_update(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]

    updated = client.post(
        f"/sessions/{session_id}/think-details",
        json={"show_think_details": True},
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["show_think_details"] is True


def test_session_command_policy_update(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]

    updated = client.post(
        f"/sessions/{session_id}/command-policy",
        json={"allow_model_command_create": False, "encourage_model_command_create": False},
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["allow_model_command_create"] is False
    assert payload["encourage_model_command_create"] is False


def test_model_install_command_blocked_by_policy(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)
    created = client.post(
        "/sessions",
        json={
            "mode": "coding",
            "project_id": "demo",
            "allow_model_command_create": False,
            "encourage_model_command_create": False,
        },
    )
    session_id = created.json()["id"]
    blocked = client.post(
        "/tools/invoke",
        params={"actor": "model"},
        json={
            "name": "InstallCommand",
            "args": {
                "session_id": session_id,
                "name": "DemoCmd",
                "definition": "echo hi",
                "runtime": "sh",
                "args_schema": {},
                "effect_schema": {},
                "capabilities": [],
                "source": "model",
                "signature": None,
            },
        },
    )
    assert blocked.status_code == 403


def test_review_endpoint_and_tool(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = api.build_app_state(tmp_path)

    async def fake_review(self, req):
        return {
            "model": "assistant",
            "thinking_level": "deep",
            "thinking_budget": 12000,
            "output_text": "review-ok",
            "raw": {"mock": True},
        }

    monkeypatch.setattr(type(api.STATE.reviews), "run", fake_review)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]

    endpoint = client.post("/review", json={"session_id": session_id, "prompt": "review this"})
    assert endpoint.status_code == 200
    assert endpoint.json()["output_text"] == "review-ok"

    tool = client.post(
        "/tools/invoke",
        json={"name": "Review", "args": {"session_id": session_id, "prompt": "review this"}},
    )
    assert tool.status_code == 200
    assert tool.json()["result"]["output_text"] == "review-ok"


def test_turn_context_is_ranked_and_compacted(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api
    from poecoder.services.model_clients import ModelReply

    api.STATE = api.build_app_state(tmp_path)
    captured: dict[str, object] = {}

    async def fake_chat(model: str, system_message: str, user_prompt: str, context: dict, images=None):
        captured["context"] = context
        return ModelReply(text="ok", raw={"mock": True})

    monkeypatch.setattr(api.STATE.turns.model_client, "chat", fake_chat)

    client = TestClient(api.app)
    session_id = client.post("/sessions", json={"mode": "coding", "project_id": "demo"}).json()["id"]
    for idx in range(35):
        client.put(
            f"/sessions/{session_id}/context",
            json={"key": f"k{idx}", "value": "x" * 2000, "scope": "pinned"},
        )
    run = client.post("/turns/execute", json={"session_id": session_id, "user_prompt": "use k1 and summary"})
    assert run.status_code == 200
    context = captured["context"]
    assert len(context["selected_context"]) <= 20
    diagnostics = context["context_diagnostics"]
    assert diagnostics["source"] == "auto_ranked"
    assert diagnostics["dropped_items"] >= 1


def test_model_table_endpoints(monkeypatch, tmp_path):
    db = tmp_path / "test.db"
    monkeypatch.setenv("POECODER_DB_PATH", str(db))

    from poecoder import api

    api.STATE = None
    client = TestClient(api.app)

    models = client.get("/models/table")
    assert models.status_code == 200
    assert isinstance(models.json(), list)
    assert len(models.json()) >= 1

    upsert = client.put(
        "/models/table/custom-model",
        json={
            "strategy": "custom",
            "best_for": "experiments",
            "speed_tier": 3,
            "quality_tier": 3,
            "cost_tier": 2,
            "max_context_hint": 32000,
        },
    )
    assert upsert.status_code == 200
    assert upsert.json()["model"] == "custom-model"

    catalog = client.get("/tools/catalog")
    assert catalog.status_code == 200
    names = {item["name"] for item in catalog.json()}
    assert "Review" in names
    assert "ReadTaskOutput" in names
    assert "StartLeaderRun" in names
    assert "ListFile" in names
    assert "ChangeWorkDir" in names
    assert "Exit" in names
