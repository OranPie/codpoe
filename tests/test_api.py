from __future__ import annotations

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
