from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from poecoder.app_state import AppState, build_app_state
from poecoder.backend.models import (
    ApiKeyRequest,
    AgentStartRequest,
    AgentTemplateUpsertRequest,
    AgentWaitRequest,
    DownloadUrlsRequest,
    GetWebRequest,
    MemoryReadRequest,
    MemoryWriteRequest,
    ProviderSecretsLoadRequest,
    ProviderSecretsSaveRequest,
    RunShellRequest,
    SearchArxivRequest,
    SearchWebRequest,
    SessionCreateRequest,
    SessionTurnRequest,
    WorkflowArxivRequest,
)
from poecoder.backend.prompting import VERSION_TAG

STATE: AppState | None = None


def get_state() -> AppState:
    global STATE
    if STATE is None:
        STATE = build_app_state(Path.cwd())
    return STATE


app = FastAPI(title="PoeCoder AgentCore API", version=VERSION_TAG)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agent-api/about")
def about() -> dict[str, Any]:
    return {
        "product": "PoeCoder AgentCore",
        "version_tag": VERSION_TAG,
        "architecture": "agent-driven",
        "execution_primitive": "RunShell",
        "nesting_depth": 2,
    }


@app.get("/agent-api/models")
async def list_models(query: str = "", limit: int = 100, full: bool = False) -> dict[str, Any]:
    state = get_state()
    cap = max(1, min(int(limit), 5000))
    query_raw = query.strip()
    all_models: list[str] = []
    for item in state.settings.supported_models:
        name = str(item).strip()
        if name and name not in all_models:
            all_models.append(name)

    source_meta: dict[str, Any] = {"mode": "configured-only"}
    if full:
        catalog = await state.model_client.fetch_full_model_catalog(
            seeded_models=all_models,
            include_openai_remote=True,
            remote_limit=5000,
        )
        merged = catalog.get("models", [])
        if isinstance(merged, list):
            all_models = [str(item) for item in merged if str(item).strip()]
        source_meta = catalog.get("sources", {}) if isinstance(catalog.get("sources", {}), dict) else {}
        source_meta["mode"] = "full"

    if query_raw:
        q = query_raw.lower()
        models = [item for item in all_models if q in item.lower()]
    else:
        models = all_models
    return {
        "query": query_raw,
        "count": len(models),
        "models": models[:cap],
        "default_small_model": state.settings.default_small_model,
        "default_large_model": state.settings.default_large_model,
        "source_meta": source_meta,
    }


@app.post("/agent-api/auth/poe/login")
def auth_poe(req: ApiKeyRequest) -> dict[str, Any]:
    state = get_state()
    state.settings.poe_api_key = req.api_key.strip()
    state.model_client.update_poe(api_key=state.settings.poe_api_key)
    return {"ok": True, "provider": "poe"}


@app.post("/agent-api/auth/openai/login")
def auth_openai(req: ApiKeyRequest) -> dict[str, Any]:
    state = get_state()
    state.settings.openai_api_key = req.api_key.strip()
    state.model_client.update_openai(api_key=state.settings.openai_api_key)
    return {"ok": True, "provider": "openai"}


def _save_provider_secrets(req: ProviderSecretsSaveRequest) -> dict[str, Any]:
    state = get_state()
    payload = {
        "poe_api_key": (req.poe_api_key if req.poe_api_key is not None else state.settings.poe_api_key) or "",
        "openai_api_key": (
            req.openai_api_key if req.openai_api_key is not None else state.settings.openai_api_key
        )
        or "",
        "poe_api_url": req.poe_api_url or state.settings.poe_api_url,
        "openai_api_url": req.openai_api_url or state.settings.openai_api_url,
    }
    try:
        state.provider_secrets.save(req.user_key, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to persist secrets: {exc}") from exc
    return {"ok": True, "path": str(state.provider_secrets.path)}


def _load_provider_secrets(req: ProviderSecretsLoadRequest) -> dict[str, Any]:
    state = get_state()
    try:
        payload = state.provider_secrets.load(req.user_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"secret file not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to read secrets: {exc}") from exc

    poe_api_key = str(payload.get("poe_api_key", "")).strip() or None
    openai_api_key = str(payload.get("openai_api_key", "")).strip() or None
    poe_api_url = str(payload.get("poe_api_url", state.settings.poe_api_url)).strip() or state.settings.poe_api_url
    openai_api_url = (
        str(payload.get("openai_api_url", state.settings.openai_api_url)).strip() or state.settings.openai_api_url
    )

    state.settings.poe_api_key = poe_api_key
    state.settings.openai_api_key = openai_api_key
    state.settings.poe_api_url = poe_api_url
    state.settings.openai_api_url = openai_api_url
    state.model_client.update_poe(api_key=poe_api_key, base_url=poe_api_url)
    state.model_client.update_openai(api_key=openai_api_key, base_url=openai_api_url)

    return {
        "ok": True,
        "poe_api_key_set": bool(poe_api_key),
        "openai_api_key_set": bool(openai_api_key),
        "poe_api_url": state.settings.poe_api_url,
        "openai_api_url": state.settings.openai_api_url,
    }


@app.post("/agent-api/auth/secrets/save")
def auth_secrets_save(req: ProviderSecretsSaveRequest) -> dict[str, Any]:
    return _save_provider_secrets(req)


@app.post("/agent-api/auth/secrets/load")
def auth_secrets_load(req: ProviderSecretsLoadRequest) -> dict[str, Any]:
    return _load_provider_secrets(req)


# Backward-compatible aliases for older clients.
@app.post("/auth/secrets/save")
def auth_secrets_save_legacy(req: ProviderSecretsSaveRequest) -> dict[str, Any]:
    return _save_provider_secrets(req)


@app.post("/auth/secrets/load")
def auth_secrets_load_legacy(req: ProviderSecretsLoadRequest) -> dict[str, Any]:
    return _load_provider_secrets(req)


@app.post("/agent-api/sessions")
def create_session(req: SessionCreateRequest) -> dict[str, Any]:
    return get_state().engine.create_session(req).model_dump(mode="json")


@app.get("/agent-api/sessions")
def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in get_state().engine.list_sessions(limit=limit)]


@app.get("/agent-api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    try:
        return get_state().engine.get_session(session_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/agent-api/sessions/{session_id}/messages")
def list_session_messages(session_id: str, limit: int = 30) -> list[dict[str, Any]]:
    try:
        return [item.model_dump(mode="json") for item in get_state().engine.list_session_messages(session_id, limit=limit)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agent-api/sessions/{session_id}/turn")
async def session_turn(session_id: str, req: SessionTurnRequest) -> dict[str, Any]:
    try:
        return (await get_state().engine.run_session_turn(session_id, req)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"


@app.post("/agent-api/sessions/{session_id}/turn/stream")
async def session_turn_stream(session_id: str, req: SessionTurnRequest) -> StreamingResponse:
    state = get_state()
    try:
        state.store.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_iter() -> AsyncIterator[str]:
        state.store.add_session_message(session_id, "user", req.prompt)
        run = state.runtime.start(
            AgentStartRequest(
                name="conversation-root",
                goal=req.prompt,
                session_id=session_id,
                model=req.model,
                scope=["."],
                max_steps=200,
            )
        )
        yield _sse("started", {"session_id": session_id, "agent_id": run.id})

        emitted = 0
        while True:
            await asyncio.sleep(0.25)
            payload = state.runtime.get(run.id, event_limit=1200)
            agent = payload.get("agent", {})
            events = payload.get("events", [])
            if not isinstance(events, list):
                events = []

            while emitted < len(events):
                item = events[emitted]
                emitted += 1
                if not isinstance(item, dict):
                    continue
                event_type = str(item.get("event_type", "event"))
                event_payload = item.get("payload", {})
                if not isinstance(event_payload, dict):
                    event_payload = {}

                if event_type == "model_action":
                    parsed = event_payload.get("parsed", {})
                    if not isinstance(parsed, dict):
                        parsed = {}
                    yield _sse(
                        "action",
                        {
                            "action": str(parsed.get("action", "")),
                            "step": event_payload.get("step"),
                            "progress": str(parsed.get("progress", "")),
                            "detail": str(parsed.get("detail", "")),
                            "next": str(parsed.get("next", "")),
                            "question": str(parsed.get("question", "")),
                            "input_mode": str(parsed.get("input_mode", "")),
                            "estimated_cost_usd": event_payload.get("estimated_cost_usd", 0.0),
                        },
                    )
                elif event_type == "runshell":
                    yield _sse(
                        "runshell",
                        {
                            "command": str(event_payload.get("command", "")),
                            "exit_code": event_payload.get("exit_code"),
                            "allowed": bool(event_payload.get("allowed", True)),
                            "duration_ms": event_payload.get("duration_ms", 0),
                            "progress": str(event_payload.get("progress", "")),
                            "stdout_preview": event_payload.get("stdout_preview", []),
                            "stderr_preview": event_payload.get("stderr_preview", []),
                        },
                    )
                elif event_type == "spawn":
                    yield _sse(
                        "spawn",
                        {
                            "child_agent_id": str(event_payload.get("child_agent_id", "")),
                            "child_status": str(event_payload.get("child_status", "")),
                            "progress": str(event_payload.get("progress", "")),
                            "child_command_summary": str(event_payload.get("child_command_summary", "")),
                            "child_output": str(event_payload.get("child_output", "")),
                        },
                    )
                elif event_type == "ask":
                    yield _sse("ask", event_payload)
                elif event_type == "note":
                    yield _sse("note", event_payload)
                elif event_type == "tool_define":
                    yield _sse("tool_define", event_payload)
                elif event_type == "tool_call":
                    yield _sse("tool_call", event_payload)

            status = str(agent.get("status", ""))
            if status in {"completed", "failed", "cancelled"}:
                output = str(agent.get("final_output", "")).strip() or str(agent.get("error", "")).strip()
                state.store.add_session_message(session_id, "assistant", output)
                metrics = state.engine.summarize_agent_events(events)
                ask_payload = None
                for item in reversed(events):
                    if isinstance(item, dict) and str(item.get("event_type", "")) == "ask":
                        payload_item = item.get("payload", {})
                        if isinstance(payload_item, dict):
                            ask_payload = payload_item
                        break
                yield _sse(
                    "final",
                    {
                        "session_id": session_id,
                        "agent_id": run.id,
                        "status": status,
                        "output": output,
                        "agent_metrics": metrics,
                        "ask": ask_payload,
                    },
                )
                return

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@app.post("/agent-api/agents/templates/register")
def register_template(req: AgentTemplateUpsertRequest) -> dict[str, Any]:
    return get_state().engine.upsert_template(req).model_dump(mode="json")


@app.get("/agent-api/agents/templates")
def list_templates() -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in get_state().engine.list_templates()]


@app.post("/agent-api/agents/start")
async def start_agent(req: AgentStartRequest) -> dict[str, Any]:
    try:
        if req.parent_agent_id and req.session_id is None:
            # inherit via parent run implicitly, but allow direct calls without session.
            pass
        return get_state().engine.start_agent(req).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/agent-api/agents/{agent_id}")
def get_agent(agent_id: str, event_limit: int = 200) -> dict[str, Any]:
    try:
        return get_state().engine.get_agent(agent_id, event_limit=event_limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agent-api/agents/{agent_id}/cancel")
def cancel_agent(agent_id: str) -> dict[str, Any]:
    try:
        return get_state().engine.cancel_agent(agent_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agent-api/agents/{agent_id}/wait")
async def wait_agent(agent_id: str, req: AgentWaitRequest) -> dict[str, Any]:
    try:
        return await get_state().engine.wait_agent(agent_id, timeout_s=req.timeout_s)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/agent-api/agents/{agent_id}/events")
def get_agent_events(agent_id: str, limit: int = 200) -> list[dict[str, Any]]:
    try:
        return get_state().engine.get_agent_events(agent_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agent-api/run-shell")
async def run_shell(req: RunShellRequest) -> dict[str, Any]:
    result = await get_state().shell.run(req)
    return result.model_dump(mode="json")


@app.post("/agent-api/memory/user/write")
def memory_user_write(req: MemoryWriteRequest) -> dict[str, Any]:
    if req.scope != "user":
        raise HTTPException(status_code=400, detail="scope must be user")
    return {"id": get_state().engine.write_memory(req)}


@app.post("/agent-api/memory/user/read")
def memory_user_read(req: MemoryReadRequest) -> list[dict[str, Any]]:
    if req.scope != "user":
        raise HTTPException(status_code=400, detail="scope must be user")
    return [item.model_dump(mode="json") for item in get_state().engine.read_memory(req)]


@app.post("/agent-api/memory/session/write")
def memory_session_write(req: MemoryWriteRequest) -> dict[str, Any]:
    if req.scope != "session":
        raise HTTPException(status_code=400, detail="scope must be session")
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for session memory")
    return {"id": get_state().engine.write_memory(req)}


@app.post("/agent-api/memory/session/read")
def memory_session_read(req: MemoryReadRequest) -> list[dict[str, Any]]:
    if req.scope != "session":
        raise HTTPException(status_code=400, detail="scope must be session")
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for session memory")
    return [item.model_dump(mode="json") for item in get_state().engine.read_memory(req)]


@app.post("/agent-api/workflows/arxiv")
async def workflow_arxiv(req: WorkflowArxivRequest) -> dict[str, Any]:
    try:
        return (await get_state().engine.run_arxiv_workflow(req)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/agent-api/research/search-web")
async def research_search_web(req: SearchWebRequest) -> dict[str, Any]:
    try:
        return await get_state().web_tools.search_web(
            query=req.query,
            limit=req.limit,
            timeout_s=req.timeout_s,
            max_snippet_chars=req.max_snippet_chars,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent-api/research/search-arxiv")
async def research_search_arxiv(req: SearchArxivRequest) -> dict[str, Any]:
    try:
        return await get_state().web_tools.search_arxiv(
            query=req.query,
            max_results=req.max_results,
            timeout_s=req.timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent-api/research/get-web")
async def research_get_web(req: GetWebRequest) -> dict[str, Any]:
    try:
        return await get_state().web_tools.get_web(
            url=req.url,
            focus=req.focus,
            timeout_s=req.timeout_s,
            max_chars=req.max_chars,
            selector=req.selector,
            regex=req.regex,
            max_matches=req.max_matches,
            download_if_large=req.download_if_large,
            download_folder=req.download_folder,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent-api/research/download-urls")
async def research_download_urls(req: DownloadUrlsRequest) -> dict[str, Any]:
    try:
        return await get_state().web_tools.download_urls(
            urls=req.urls,
            folder=req.folder,
            overwrite=req.overwrite,
            timeout_s=req.timeout_s,
            max_bytes=req.max_bytes,
            max_files=req.max_files,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run() -> None:
    parser = argparse.ArgumentParser(description="Run PoeCoder AgentCore API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    state = get_state()
    uvicorn.run(
        "poecoder.api:app",
        host=args.host or state.settings.host,
        port=args.port or state.settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
