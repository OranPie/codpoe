from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

from poecoder.models import TurnRequest, TurnResult
from poecoder.prompts import default_system_message_for_mode
from poecoder.router import ModelRouter
from poecoder.services.memory_service import MemoryReadRequest, MemoryService
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.model_profile_service import ModelProfileService
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
    model_profiles: ModelProfileService

    async def execute(self, req: TurnRequest) -> TurnResult:
        session, context, model, system_message, decision = self._prepare_turn(req)
        first_context = self._with_turn_phase(context, "tool_or_answer")

        first = await self.model_client.chat(
            model=model,
            system_message=system_message,
            user_prompt=req.user_prompt,
            context=first_context,
            images=req.images,
        )

        tool_calls = parse_tool_calls(first.text)
        tool_events: list[dict[str, Any]] = []
        output_text = first.text
        repair_attempt = 0
        while not tool_calls and self._needs_repair_output(output_text) and repair_attempt < 2:
            repair_attempt += 1
            repaired = await self.model_client.chat(
                model=model,
                system_message=system_message,
                user_prompt=self._repair_prompt(req.user_prompt, repair_attempt),
                context=self._with_turn_phase(context, "tool_or_answer", repair_attempt=repair_attempt),
                images=req.images,
            )
            output_text = repaired.text
            tool_calls = parse_tool_calls(output_text)

        if tool_calls:
            for call in tool_calls:
                self._inject_default_tool_args(session.id, call)
                try:
                    result = await self.tools.invoke("model", call["name"], call["args"])
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                tool_events.append({"name": call["name"], "args": call["args"], "result": result})
                self.sessions.put_context(session.id, f"tool:{call['name']}", result, scope="turn")

            second_prompt = (
                req.user_prompt
                + "\n\nTool results:\n"
                + json.dumps(tool_events, ensure_ascii=True)
                + "\n\nTurn phase: final_response."
                + "\nYou can answer in multiple turns. This is the final-response turn after tool execution."
                + "\nDo not emit placeholder markdown or fake tool calls."
                + "\nNow synthesize the final user-facing answer."
            )
            second_context = self._with_turn_phase(context, "final_response", tool_events=tool_events)
            second = await self.model_client.chat(
                model=model,
                system_message=system_message,
                user_prompt=second_prompt,
                context=second_context,
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
        first_context = self._with_turn_phase(context, "tool_or_answer")
        yield {"type": "model", "data": {"model": model}}

        # First response stream (low latency path)
        first_chunks: list[str] = []
        yield {"type": "status", "data": "responding"}
        async for chunk in self.model_client.chat_stream(
            model=model,
            system_message=system_message,
            user_prompt=req.user_prompt,
            context=first_context,
            images=req.images,
        ):
            first_chunks.append(chunk)
            yield {"type": "delta", "data": chunk}

        first_text = "".join(first_chunks)
        tool_calls = parse_tool_calls(first_text)
        repair_attempt = 0
        while not tool_calls and self._needs_repair_output(first_text) and repair_attempt < 2:
            repair_attempt += 1
            repaired_text = ""
            yield {"type": "status", "data": "responding"}
            async for chunk in self.model_client.chat_stream(
                model=model,
                system_message=system_message,
                user_prompt=self._repair_prompt(req.user_prompt, repair_attempt),
                context=self._with_turn_phase(context, "tool_or_answer", repair_attempt=repair_attempt),
                images=req.images,
            ):
                repaired_text += chunk
                yield {"type": "delta", "data": chunk}
            first_text = repaired_text
            tool_calls = parse_tool_calls(first_text)
        tool_events: list[dict[str, Any]] = []
        output_text = first_text

        if tool_calls:
            yield {"type": "status", "data": "tools"}
            for call in tool_calls:
                self._inject_default_tool_args(session.id, call)
                try:
                    result = await self.tools.invoke("model", call["name"], call["args"])
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                tool_event = {"name": call["name"], "args": call["args"], "result": result}
                tool_events.append(tool_event)
                self.sessions.put_context(session.id, f"tool:{call['name']}", result, scope="turn")
                yield {"type": "tool", "data": tool_event}

            second_prompt = (
                req.user_prompt
                + "\n\nTool results:\n"
                + json.dumps(tool_events, ensure_ascii=True)
                + "\n\nTurn phase: final_response."
                + "\nYou can answer in multiple turns. This is the final-response turn after tool execution."
                + "\nDo not emit placeholder markdown or fake tool calls."
                + "\nNow synthesize the final user-facing answer."
            )
            second_context = self._with_turn_phase(context, "final_response", tool_events=tool_events)
            output_text = ""
            yield {"type": "status", "data": "responding"}
            async for chunk in self.model_client.chat_stream(
                model=model,
                system_message=system_message,
                user_prompt=second_prompt,
                context=second_context,
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

        selected_context, context_diagnostics = self.sessions.select_context_for_prompt(
            session_id=session.id,
            prompt=req.user_prompt,
            keys=req.context_keys if req.context_keys else None,
            max_items=20,
            max_value_chars=10000,
        )
        memory_context = self._load_memory(session.id, session.project_id, req.user_prompt)
        available_models = self.model_catalog.list_models(refresh=False)
        self.model_profiles.ensure_seeded(available_models)
        model_table = [self._json_safe_profile(item) for item in self.model_profiles.list()]
        context = {
            "session": session.model_dump(mode="json"),
            "selected_context": selected_context,
            "memory": memory_context,
            "context_diagnostics": context_diagnostics,
            "command_catalog": self.tools.command_catalog(),
            "model_table": model_table,
            "command_policy": {
                "allow_model_command_create": session.allow_model_command_create,
                "encourage_model_command_create": session.encourage_model_command_create,
            },
            "metadata": req.metadata,
        }

        decision = self.router.decide(req.user_prompt, context_size_hint=len(json.dumps(context)), tool_count_hint=0)
        thinking_level = req.thinking_level or session.thinking_level
        thinking_budget = req.thinking_budget or session.thinking_budget
        show_think_details = (
            bool(req.metadata.get("show_think_details"))
            if isinstance(req.metadata.get("show_think_details"), bool)
            else session.show_think_details
        )
        context["model_settings"] = {
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "show_think_details": show_think_details,
        }
        model = decision.selected_model if session.active_model in {"", "auto"} else session.active_model
        try:
            self.model_catalog.ensure_supported(model)
        except ValueError:
            available = self.model_catalog.list_models(refresh=True)
            if available:
                model = available[0]
        if session.active_model in {"", "auto"}:
            model = self.model_profiles.choose_model(
                available_models=available_models,
                fallback_model=model,
                complexity=decision.complexity,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
            )

        system_message = req.system_message or default_system_message_for_mode(session.mode)
        if not show_think_details:
            system_message += (
                "\n\nOutput style:\n"
                "- Do not emit progress filler like 'Thinking...' or 'Generating...'.\n"
                "- Return either strict @tool lines or the final answer."
            )
        if session.encourage_model_command_create:
            system_message += (
                "\n\nCommand autonomy hint:\n"
                "- If a reusable workflow appears 2+ times, prefer creating/updating a command via InstallCommand/EditCommand.\n"
                "- Keep command args/effects compact and explicit."
            )
        return session, context, model, system_message, decision

    @staticmethod
    def _inject_default_tool_args(session_id: str, call: dict[str, Any]) -> None:
        name = call.get("name")
        args = call.get("args", {})
        if not isinstance(args, dict):
            return
        if name == "ChangeModel" and "session_id" not in args:
            args["session_id"] = session_id
        if name == "Review" and "session_id" not in args:
            args["session_id"] = session_id
        if name in {"InstallCommand", "EditCommand", "DelCommand"} and "session_id" not in args:
            args["session_id"] = session_id
        if name == "StartLeaderRun" and "session_id" not in args:
            args["session_id"] = session_id
        if name == "StartBackgroundTurn" and "session_id" not in args:
            args["session_id"] = session_id
        if name == "StartBackgroundSubAgent" and "parent_session_id" not in args:
            args["parent_session_id"] = session_id

    @staticmethod
    def _json_safe_profile(item: dict[str, Any]) -> dict[str, Any]:
        out = dict(item)
        for key in ("created_at", "updated_at"):
            value = out.get(key)
            if hasattr(value, "isoformat"):
                out[key] = value.isoformat()
        return out

    def _finalize_turn(self, session_id: str, decision: Any, user_prompt: str, output_text: str) -> None:
        self.sessions.put_context(session_id, "router_decision", decision.model_dump(mode="json"), scope="turn")
        self.sessions.put_context(session_id, "last_user_prompt", user_prompt, scope="turn")
        self.sessions.put_context(session_id, "last_model_output", output_text, scope="turn")
        self.sessions.maybe_update_title_from_turn(session_id, user_prompt, output_text)
        self.sessions.touch(session_id)

    def _load_memory(self, session_id: str, project_id: str, query: str) -> dict[str, list[dict[str, Any]]]:
        session_entries = self.memories.read(
            MemoryReadRequest(scope="session", session_id=session_id, limit=8)
        )
        project_entries = self.memories.read(
            MemoryReadRequest(scope="project", project_id=project_id, limit=8)
        )
        global_entries = self.memories.read(MemoryReadRequest(scope="global", limit=4))
        query_entries = self.memories.read(
            MemoryReadRequest(query=query, project_id=project_id, limit=6)
        )
        return {
            "session": [item.model_dump(mode="json") for item in session_entries],
            "project": [item.model_dump(mode="json") for item in project_entries],
            "global": [item.model_dump(mode="json") for item in global_entries],
            "query_hits": [item.model_dump(mode="json") for item in query_entries],
        }

    @staticmethod
    def _with_turn_phase(
        context: dict[str, Any],
        phase: str,
        tool_events: list[dict[str, Any]] | None = None,
        repair_attempt: int = 0,
    ) -> dict[str, Any]:
        out = dict(context)
        out["turn_protocol"] = {
            "phase": phase,
            "multi_stage": True,
            "tool_events_count": len(tool_events or []),
            "repair_attempt": repair_attempt,
        }
        return out

    @staticmethod
    def _repair_prompt(user_prompt: str, repair_attempt: int) -> str:
        return (
            user_prompt
            + f"\n\nPrevious response looked incomplete (status/filler text). Repair attempt #{repair_attempt}."
            + "\nReply correctly now:"
            + "\n- If tools are needed, output only strict @tool lines."
            + "\n- Otherwise output a direct final answer."
            + "\n- No placeholder markdown and no filler text."
        )

    @staticmethod
    def _needs_repair_output(text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        lowered = stripped.lower()
        if "@tool " in lowered:
            return False
        filler_matches = re.findall(
            r"(thinking\.{3}(?: \(\d+s elapsed\))?|generating\.{3}(?: \(\d+s elapsed\))?)",
            lowered,
        )
        if filler_matches:
            filler_chars = sum(len(item) for item in filler_matches)
            if filler_chars >= int(len(lowered) * 0.6):
                return True
        if lowered.startswith("thinking...") and "@tool " not in stripped and len(stripped) < 120:
            return True
        if lowered.startswith("generating...") and "@tool " not in stripped and len(stripped) < 120:
            return True
        if "mistake tool call placeholder" in lowered:
            return True
        return False
