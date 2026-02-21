from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

from poecoder.models import TurnRequest, TurnResult
from poecoder.prompts import default_system_message_for_mode
from poecoder.router import ModelRouter
from poecoder.services.memory_service import MemoryReadRequest, MemoryService
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.session_service import SessionService
from poecoder.tools.runtime import ToolRuntime, parse_tool_calls


@dataclass(slots=True)
class TurnService:
    sessions: SessionService
    memories: MemoryService
    model_client: PoeModelClient
    router: ModelRouter
    tools: ToolRuntime
    model_catalog: ModelCatalog

    async def execute(self, req: TurnRequest) -> TurnResult:
        session, context, model, system_message, decision = self._prepare_turn(req)

        first = await self.model_client.chat(
            model=model,
            system_message=system_message,
            user_prompt=req.user_prompt,
            context=context,
            images=req.images,
        )

        tool_calls = parse_tool_calls(first.text)
        tool_events: list[dict[str, Any]] = []
        output_text = first.text

        if tool_calls:
            for call in tool_calls:
                self._inject_default_tool_args(session.id, call)
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
                images=req.images,
            )
            output_text = second.text

        self._finalize_turn(session.id, decision, req.user_prompt, output_text)
        return TurnResult(
            session_id=session.id,
            model=model,
            output_text=output_text,
            tool_events=tool_events,
        )

    async def execute_stream(self, req: TurnRequest) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "status", "data": "routing"}
        await asyncio.sleep(0)

        session, context, model, system_message, decision = self._prepare_turn(req)
        yield {"type": "model", "data": {"model": model}}

        # First response stream (low latency path)
        first_chunks: list[str] = []
        yield {"type": "status", "data": "responding"}
        async for chunk in self.model_client.chat_stream(
            model=model,
            system_message=system_message,
            user_prompt=req.user_prompt,
            context=context,
            images=req.images,
        ):
            first_chunks.append(chunk)
            yield {"type": "delta", "data": chunk}

        first_text = "".join(first_chunks)
        tool_calls = parse_tool_calls(first_text)
        tool_events: list[dict[str, Any]] = []
        output_text = first_text

        if tool_calls:
            yield {"type": "status", "data": "tools"}
            for call in tool_calls:
                self._inject_default_tool_args(session.id, call)
                result = await self.tools.invoke("model", call["name"], call["args"])
                tool_event = {"name": call["name"], "args": call["args"], "result": result}
                tool_events.append(tool_event)
                self.sessions.put_context(session.id, f"tool:{call['name']}", result, scope="turn")
                yield {"type": "tool", "data": tool_event}

            second_prompt = (
                req.user_prompt
                + "\n\nTool results:\n"
                + json.dumps(tool_events, ensure_ascii=True)
                + "\nSynthesize final answer."
            )
            output_text = ""
            yield {"type": "status", "data": "responding"}
            async for chunk in self.model_client.chat_stream(
                model=model,
                system_message=system_message,
                user_prompt=second_prompt,
                context=context,
                images=req.images,
            ):
                output_text += chunk
                yield {"type": "delta", "data": chunk}

        self._finalize_turn(session.id, decision, req.user_prompt, output_text)

        final = TurnResult(
            session_id=session.id,
            model=model,
            output_text=output_text,
            tool_events=tool_events,
        )
        yield {"type": "final", "data": final.model_dump(mode="json")}

    def _prepare_turn(self, req: TurnRequest) -> tuple[Any, dict[str, Any], str, str, Any]:
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

        system_message = req.system_message or default_system_message_for_mode(session.mode)
        return session, context, model, system_message, decision

    @staticmethod
    def _inject_default_tool_args(session_id: str, call: dict[str, Any]) -> None:
        name = call.get("name")
        args = call.get("args", {})
        if not isinstance(args, dict):
            return
        if name == "ChangeModel" and "session_id" not in args:
            args["session_id"] = session_id
        if name == "StartBackgroundTurn" and "session_id" not in args:
            args["session_id"] = session_id
        if name == "StartBackgroundSubAgent" and "parent_session_id" not in args:
            args["parent_session_id"] = session_id

    def _finalize_turn(self, session_id: str, decision: Any, user_prompt: str, output_text: str) -> None:
        self.sessions.put_context(session_id, "router_decision", decision.model_dump(mode="json"), scope="turn")
        self.sessions.put_context(session_id, "last_user_prompt", user_prompt, scope="turn")
        self.sessions.put_context(session_id, "last_model_output", output_text, scope="turn")
        self.sessions.touch(session_id)

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
