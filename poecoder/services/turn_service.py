from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from poecoder.models import TurnRequest, TurnResult
from poecoder.router import ModelRouter
from poecoder.services.memory_service import MemoryReadRequest, MemoryService
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.session_service import SessionService
from poecoder.tools.runtime import ToolRuntime, parse_tool_calls
from poecoder.prompts import MAIN_SYSTEM_MESSAGE


@dataclass(slots=True)
class TurnService:
    sessions: SessionService
    memories: MemoryService
    model_client: PoeModelClient
    router: ModelRouter
    tools: ToolRuntime
    model_catalog: ModelCatalog

    async def execute(self, req: TurnRequest) -> TurnResult:
        session = self.sessions.get(req.session_id)
        self.sessions.reset_for_turn(session)

        selected_context = self.sessions.get_context(session.id, req.context_keys if req.context_keys else None)
        memory_context = self._load_memory(session.id, session.project_id)
        context = {
            "session": session.model_dump(mode="json"),
            "selected_context": selected_context,
            "memory": memory_context,
            "metadata": req.metadata,
        }

        decision = self.router.decide(req.user_prompt, context_size_hint=len(json.dumps(context)), tool_count_hint=0)
        model = decision.selected_model if session.active_model in {"", "auto"} else session.active_model
        try:
            self.model_catalog.ensure_supported(model)
        except ValueError:
            available = self.model_catalog.list_models(refresh=True)
            if available:
                model = available[0]

        system_message = req.system_message or MAIN_SYSTEM_MESSAGE
        first = await self.model_client.chat(
            model=model,
            system_message=system_message,
            user_prompt=req.user_prompt,
            context=context,
        )

        tool_calls = parse_tool_calls(first.text)
        tool_events: list[dict[str, Any]] = []
        output_text = first.text

        if tool_calls:
            for call in tool_calls:
                if call["name"] == "ChangeModel" and "session_id" not in call["args"]:
                    call["args"]["session_id"] = session.id
                result = await self.tools.invoke("model", call["name"], call["args"])
                tool_events.append({"name": call["name"], "args": call["args"], "result": result})
                self.sessions.put_context(session.id, f"tool:{call['name']}", result, scope="turn")

            second_prompt = (
                req.user_prompt
                + "\n\nTool results:\n"
                + json.dumps(tool_events, ensure_ascii=True)
                + "\nSynthesize final answer."
            )
            second = await self.model_client.chat(
                model=model,
                system_message=system_message,
                user_prompt=second_prompt,
                context=context,
            )
            output_text = second.text

        self.sessions.put_context(session.id, "router_decision", decision.model_dump(mode="json"), scope="turn")
        self.sessions.put_context(session.id, "last_user_prompt", req.user_prompt, scope="turn")
        self.sessions.put_context(session.id, "last_model_output", output_text, scope="turn")
        self.sessions.touch(session.id)

        return TurnResult(
            session_id=session.id,
            model=model,
            output_text=output_text,
            tool_events=tool_events,
        )

    async def execute_stream(self, req: TurnRequest) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "status", "data": "routing"}
        await asyncio.sleep(0)
        result = await self.execute(req)
        yield {"type": "model", "data": {"model": result.model}}

        if result.tool_events:
            yield {"type": "status", "data": "tools"}
        for event in result.tool_events:
            yield {"type": "tool", "data": event}

        yield {"type": "status", "data": "responding"}
        for chunk in self._chunk_text(result.output_text, size=64):
            yield {"type": "delta", "data": chunk}
        yield {"type": "final", "data": result.model_dump(mode="json")}

    @staticmethod
    def _chunk_text(text: str, size: int = 64) -> list[str]:
        if not text:
            return []
        return [text[i : i + size] for i in range(0, len(text), size)]

    def _load_memory(self, session_id: str, project_id: str) -> dict[str, list[dict[str, Any]]]:
        session_entries = self.memories.read(
            MemoryReadRequest(scope="session", session_id=session_id, limit=8)
        )
        project_entries = self.memories.read(
            MemoryReadRequest(scope="project", project_id=project_id, limit=8)
        )
        global_entries = self.memories.read(MemoryReadRequest(scope="global", limit=4))
        return {
            "session": [item.model_dump(mode="json") for item in session_entries],
            "project": [item.model_dump(mode="json") for item in project_entries],
            "global": [item.model_dump(mode="json") for item in global_entries],
        }
