from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from poecoder.app_state import AppState, build_app_state
from poecoder.models import (
    ApiLoginRequest,
    CommandInstallRequest,
    CommandPatchRequest,
    ChangeModelRequest,
    ContextPutRequest,
    LeaderRunRequest,
    LeaderWaitRequest,
    ModelProfileUpsertRequest,
    MemoryEditRequest,
    MemoryReadRequest,
    MemoryWriteRequest,
    ProviderBaseUrlRequest,
    GetWebRawRequest,
    GetWebRequest,
    GetWebFileRequest,
    ReplaceRequest,
    SearchRequest,
    SessionCreateRequest,
    ShellRunRequest,
    SubagentStartRequest,
    TmpWriteRequest,
    TurnRequest,
    WikiCompactRequest,
    WikiIngestRequest,
    WikiQueryRequest,
    WriteRawRequest,
    ReadRawRequest,
    ReadRecursiveRequest,
    ReadStructRequest,
    ReadTaskOutputRequest,
    ReviewRequest,
    ReviewSettingsRequest,
    SessionCommandPolicyRequest,
    SessionThinkingRequest,
    ToolCall,
    TaskStartSubagentRequest,
)
from poecoder.services.model_clients import ModelProviderError

STATE: AppState | None = None


def get_state() -> AppState:
    global STATE
    if STATE is None:
        STATE = build_app_state(Path.cwd())
    return STATE


app = FastAPI(title="PoeCoder API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _as_http_error(exc: ModelProviderError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=exc.to_payload())


@app.post("/auth/poe/login")
def auth_poe_login(req: ApiLoginRequest) -> dict[str, Any]:
    state = get_state()
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    state.settings.poe_api_key = api_key
    state.turns.model_client.update_poe(api_key=api_key)
    state.subagents.model_client.update_poe(api_key=api_key)
    state.reviews.model_client.update_poe(api_key=api_key)
    state.model_catalog.api_key = api_key
    state.usage.api_key = api_key
    return {"ok": True, "message": "poe api key updated"}


@app.post("/auth/openai/login")
def auth_openai_login(req: ApiLoginRequest) -> dict[str, Any]:
    state = get_state()
    api_key = req.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    state.settings.openai_api_key = api_key
    state.turns.model_client.update_openai(api_key=api_key)
    state.subagents.model_client.update_openai(api_key=api_key)
    state.reviews.model_client.update_openai(api_key=api_key)
    return {"ok": True, "message": "openai api key updated"}


@app.post("/providers/poe/base-url")
def provider_poe_base_url(req: ProviderBaseUrlRequest) -> dict[str, Any]:
    state = get_state()
    base_url = req.base_url.strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    state.settings.poe_api_url = base_url
    state.turns.model_client.update_poe(base_url=base_url)
    state.subagents.model_client.update_poe(base_url=base_url)
    state.reviews.model_client.update_poe(base_url=base_url)
    return {"ok": True, "base_url": state.turns.model_client.api_url}


@app.post("/providers/openai/base-url")
def provider_openai_base_url(req: ProviderBaseUrlRequest) -> dict[str, Any]:
    state = get_state()
    base_url = req.base_url.strip()
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url is required")
    state.settings.openai_api_url = base_url
    state.turns.model_client.update_openai(base_url=base_url)
    state.subagents.model_client.update_openai(base_url=base_url)
    state.reviews.model_client.update_openai(base_url=base_url)
    return {"ok": True, "base_url": state.turns.model_client.openai_api_url}


@app.post("/sessions")
def create_session(req: SessionCreateRequest) -> dict[str, Any]:
    state = get_state()
    return state.sessions.create(req).model_dump(mode="json")


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    state = get_state()
    try:
        return state.sessions.get(session_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/sessions/{session_id}/context")
def put_context(session_id: str, req: ContextPutRequest) -> dict[str, Any]:
    state = get_state()
    try:
        state.sessions.get(session_id)
        state.sessions.put_context(session_id, req.key, req.value, scope=req.scope, ttl_seconds=req.ttl_seconds)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc




@app.get("/models")
def list_models(refresh: bool = False) -> dict[str, Any]:
    state = get_state()
    return {"models": state.model_catalog.list_models(refresh=refresh)}


@app.get("/models/table")
def list_model_table() -> list[dict[str, Any]]:
    state = get_state()
    state.model_profiles.ensure_seeded(state.model_catalog.list_models(refresh=False))
    return state.model_profiles.list()


@app.put("/models/table/{model}")
def upsert_model_table(model: str, req: ModelProfileUpsertRequest) -> dict[str, Any]:
    state = get_state()
    payload = req.model_dump(mode="json")
    return state.model_profiles.upsert(
        model=model,
        strategy=payload["strategy"],
        best_for=payload["best_for"],
        speed_tier=payload["speed_tier"],
        quality_tier=payload["quality_tier"],
        cost_tier=payload["cost_tier"],
        max_context_hint=payload["max_context_hint"],
    )


@app.post("/sessions/{session_id}/change-model")
def change_model(session_id: str, req: ChangeModelRequest) -> dict[str, Any]:
    state = get_state()
    try:
        state.model_catalog.ensure_supported(req.model)
        updated = state.sessions.change_model(session_id, req.model)
        return updated.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/sessions/{session_id}/thinking")
def session_update_thinking(session_id: str, req: SessionThinkingRequest) -> dict[str, Any]:
    state = get_state()
    try:
        updated = state.sessions.update_thinking(session_id, req.thinking_level, req.thinking_budget)
        return updated.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/sessions/{session_id}/command-policy")
def session_update_command_policy(session_id: str, req: SessionCommandPolicyRequest) -> dict[str, Any]:
    state = get_state()
    try:
        updated = state.sessions.update_command_policy(
            session_id=session_id,
            allow_create=req.allow_model_command_create,
            encourage_create=req.encourage_model_command_create,
        )
        return updated.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.post("/turns/execute")
async def turn_execute(req: TurnRequest) -> dict[str, Any]:
    state = get_state()
    try:
        result = await state.turns.execute(req)
        return result.model_dump(mode="json")
    except ModelProviderError as exc:
        raise _as_http_error(exc) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/review")
async def review(req: ReviewRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return await state.reviews.run(req)
    except ModelProviderError as exc:
        raise _as_http_error(exc) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/review/settings")
def review_settings_get() -> dict[str, Any]:
    return get_state().reviews.get_settings()


@app.post("/review/settings")
def review_settings_update(req: ReviewSettingsRequest) -> dict[str, Any]:
    state = get_state()
    try:
        model = req.model
        if model:
            state.model_catalog.ensure_supported(model)
        return state.reviews.update_settings(
            model=model,
            thinking_level=req.thinking_level,
            thinking_budget=req.thinking_budget,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/turns/execute/stream")
async def turn_execute_stream(req: TurnRequest) -> StreamingResponse:
    state = get_state()

    async def event_gen():
        try:
            async for event in state.turns.execute_stream(req):
                payload = json.dumps(event, ensure_ascii=True)
                yield f"data: {payload}\n\n"
        except ModelProviderError as exc:
            payload = json.dumps({"type": "error", "data": exc.to_payload()}, ensure_ascii=True)
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")




@app.post("/tasks/turns/start")
async def task_start_turn(req: TurnRequest) -> dict[str, Any]:
    state = get_state()
    try:
        task = await state.tasks.start_turn(req)
        return task.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/tasks/subagents/start")
async def task_start_subagent(req: TaskStartSubagentRequest) -> dict[str, Any]:
    state = get_state()
    try:
        task = await state.tasks.start_subagent(req)
        return task.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/tasks")
def task_list(limit: int = 50, state: str | None = None, task_type: str | None = None) -> list[dict[str, Any]]:
    store = get_state().tasks
    tasks = store.list(limit=limit, state=state, task_type=task_type)
    return [item.model_dump(mode="json") for item in tasks]


@app.get("/tasks/{task_id}")
def task_get(task_id: str) -> dict[str, Any]:
    store = get_state().tasks
    try:
        return store.get(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/tasks/{task_id}/output")
def task_get_output(task_id: str) -> dict[str, Any]:
    store = get_state().tasks
    try:
        return store.read_output(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/tasks/{task_id}/cancel")
def task_cancel(task_id: str) -> dict[str, Any]:
    store = get_state().tasks
    try:
        return store.cancel(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/leader/start")
async def leader_start(req: LeaderRunRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return state.leader.start(req).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/leader/{run_id}")
def leader_read(run_id: str) -> dict[str, Any]:
    state = get_state()
    try:
        return state.leader.read(run_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/leader/{run_id}/jobs")
def leader_jobs(run_id: str) -> list[dict[str, Any]]:
    state = get_state()
    try:
        return [item.model_dump(mode="json") for item in state.leader.list_jobs(run_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/leader/{run_id}/wait")
async def leader_wait(run_id: str, req: LeaderWaitRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return (await state.leader.wait(run_id, timeout_s=req.timeout_s)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/leader/{run_id}/cancel")
def leader_cancel(run_id: str) -> dict[str, Any]:
    state = get_state()
    try:
        return state.leader.cancel(run_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/memory/write")
def memory_write(req: MemoryWriteRequest) -> dict[str, Any]:
    state = get_state()
    return {"id": state.memories.write(req)}


@app.post("/memory/read")
def memory_read(req: MemoryReadRequest) -> list[dict[str, Any]]:
    state = get_state()
    return [item.model_dump(mode="json") for item in state.memories.read(req)]


@app.post("/memory/edit")
def memory_edit(req: MemoryEditRequest) -> dict[str, Any]:
    state = get_state()
    return {"changed": state.memories.edit(req)}


@app.post("/wiki/ingest")
def wiki_ingest(req: WikiIngestRequest) -> dict[str, Any]:
    state = get_state()
    return {"id": state.wiki.ingest(req.project_id, req.topic, req.content, req.source)}


@app.post("/wiki/query")
def wiki_query(req: WikiQueryRequest) -> list[dict[str, Any]]:
    state = get_state()
    return state.wiki.query(req.project_id, req.query, req.limit)


@app.post("/wiki/compact")
def wiki_compact(req: WikiCompactRequest) -> dict[str, Any]:
    state = get_state()
    return state.wiki.compact(req.project_id)




@app.get("/usage/current_balance")
def usage_current_balance() -> dict[str, Any]:
    state = get_state()
    try:
        return state.usage.get_current_balance()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/commands/install")
def command_install(req: CommandInstallRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return state.commands.install(
            name=req.name,
            definition=req.definition,
            runtime=req.runtime,
            args_schema=req.args_schema,
            effect_schema=req.effect_schema,
            capabilities=req.capabilities,
            source=req.source,
            signature=req.signature,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/commands")
def command_list() -> list[dict[str, Any]]:
    state = get_state()
    return state.commands.list()


@app.patch("/commands/{name}")
def command_patch(name: str, req: CommandPatchRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return state.commands.patch(name, req.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.delete("/commands/{name}")
def command_delete(name: str) -> dict[str, bool]:
    state = get_state()
    return {"deleted": state.commands.delete(name)}


@app.post("/subagents/start")
def subagent_start(req: SubagentStartRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return state.subagents.start(
            parent_session_id=req.parent_session_id,
            model=req.model,
            perm=req.perm,
            prompt=req.prompt,
            context_share=req.context_share,
            images=req.images,
            system_message_modifier=req.system_message_modifier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/subagents/{agent_id}")
def subagent_read(agent_id: str) -> dict[str, Any]:
    state = get_state()
    try:
        return state.subagents.read(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/subagents/{agent_id}/wait")
async def subagent_wait(agent_id: str, timeout_s: int = 60) -> dict[str, Any]:
    state = get_state()
    try:
        return await state.subagents.wait(agent_id, timeout_s)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/subagents/{agent_id}/cancel")
def subagent_cancel(agent_id: str) -> dict[str, Any]:
    state = get_state()
    try:
        return state.subagents.cancel(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/shell/run")
async def shell_run(req: ShellRunRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return (await state.shell.run(req.session_id, req.command, req.danger_level, req.cwd, req.timeout_s)).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc




@app.post("/tools/invoke")
async def tool_invoke(req: ToolCall, actor: str = "user") -> dict[str, Any]:
    state = get_state()
    try:
        result = await state.tools.invoke(actor=actor, name=req.name, args=req.args)
        return {"ok": True, "result": result}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/tools/catalog")
def tool_catalog() -> list[dict[str, Any]]:
    state = get_state()
    return state.tools.command_catalog()

@app.post("/tools/read_raw")
def tool_read_raw(req: ReadRawRequest) -> dict[str, Any]:
    state = get_state()
    return state.tools.code_tools.read_raw(req.file, req.line, req.end_line)


@app.post("/tools/read_struct")
def tool_read_struct(req: ReadStructRequest) -> dict[str, Any]:
    state = get_state()
    return state.tools.code_tools.read_struct(req.target, req.language, req.dependency_depth)


@app.post("/tools/read_recursive")
def tool_read_recursive(req: ReadRecursiveRequest) -> dict[str, Any]:
    state = get_state()
    return state.tools.code_tools.read_recursive(req.seed_files, req.boundary)


@app.post("/tools/search")
def tool_search(req: SearchRequest) -> list[dict[str, Any]]:
    state = get_state()
    return state.tools.code_tools.search(req.pattern, req.file_pattern, req.boundary, req.root)


@app.post("/tools/write_raw")
def tool_write_raw(req: WriteRawRequest) -> dict[str, Any]:
    state = get_state()
    return state.tools.code_tools.write_raw(req.file, req.line, req.content, req.append)


@app.post("/tools/write_replace")
def tool_write_replace(req: ReplaceRequest) -> dict[str, Any]:
    state = get_state()
    return state.tools.code_tools.write_replace(req.pattern, req.replacement, req.location, req.max_changes)






@app.get("/tools/get_balance")
def tool_get_balance() -> dict[str, Any]:
    state = get_state()
    try:
        return state.usage.get_current_balance()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/tools/get_web_raw")
async def tool_get_web_raw(req: GetWebRawRequest) -> dict[str, Any]:
    state = get_state()
    return await state.tools.web_tools.get_web_raw(
        url=req.url,
        timeout_s=req.timeout_s,
        max_chars=req.max_chars,
        headers=req.headers or None,
    )


@app.post("/tools/get_web")
async def tool_get_web(req: GetWebRequest) -> dict[str, Any]:
    state = get_state()
    return await state.tools.web_tools.get_web(
        url=req.url,
        focus=req.focus,
        timeout_s=req.timeout_s,
        max_chars=req.max_chars,
    )


@app.post("/tools/get_web_file")
async def tool_get_web_file(req: GetWebFileRequest) -> dict[str, Any]:
    state = get_state()
    return await state.tools.web_tools.get_web_file(
        url=req.url,
        save_as=req.save_as,
        folder=req.folder,
        overwrite=req.overwrite,
        timeout_s=req.timeout_s,
        max_bytes=req.max_bytes,
    )


@app.post("/tools/read_task_output")
def tool_read_task_output(req: ReadTaskOutputRequest) -> dict[str, Any]:
    state = get_state()
    try:
        return state.tasks.read_output(req.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.post("/tools/tmp_write")
def tool_tmp_write(req: TmpWriteRequest) -> dict[str, Any]:
    state = get_state()
    return state.tools._tmp_write(req.name, req.content, req.ttl_seconds)


def run() -> None:
    parser = argparse.ArgumentParser(description="Run PoeCoder API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    state = get_state()
    host = args.host or state.settings.host
    port = args.port or state.settings.port
    uvicorn.run("poecoder.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
