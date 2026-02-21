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
        tool_events: list[dict[str, Any]] = []
        usage = self._init_usage()
        output_text = ""

        phase = "tool_or_answer"
        current_prompt = req.user_prompt
        repair_attempt = 0
        repeated_tool_rounds: dict[str, int] = {}
        while True:
            stage_context = self._with_turn_phase(context, phase, tool_events=tool_events, repair_attempt=repair_attempt)
            reply = await self.model_client.chat(
                model=model,
                system_message=system_message,
                user_prompt=current_prompt,
                context=stage_context,
                images=req.images,
            )
            self._record_usage_stage(
                usage=usage,
                phase=phase,
                system_message=system_message,
                user_prompt=current_prompt,
                context=stage_context,
                images=req.images,
                output_text=reply.text,
                raw=reply.raw,
            )

            tool_calls = parse_tool_calls(reply.text)
            if not tool_calls and self._needs_repair_output(reply.text) and repair_attempt < 1:
                repair_attempt += 1
                current_prompt = self._repair_prompt(current_prompt, repair_attempt)
                continue

            if not tool_calls:
                output_text = reply.text
                break
            signature = self._tool_calls_signature(tool_calls)
            if signature:
                seen = repeated_tool_rounds.get(signature, 0) + 1
                repeated_tool_rounds[signature] = seen
                if seen >= 3:
                    output_text = self._loop_guard_response()
                    break

            for call in tool_calls:
                self._inject_default_tool_args(session.id, call)
                try:
                    result = await self.tools.invoke("model", call["name"], call["args"])
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                tool_events.append({"name": call["name"], "args": call["args"], "result": result})
                self.sessions.put_context(session.id, f"tool:{call['name']}", result, scope="turn")

            forwarded_tool_events, forwarding_meta = self._prepare_tool_events_for_prompt(
                tool_events=tool_events,
                metadata=req.metadata,
            )
            usage["tool_forwarding"] = forwarding_meta
            phase = "final_response"
            repair_attempt = 0
            current_prompt = self._tool_followup_prompt(req.user_prompt, forwarded_tool_events, forwarding_meta)

        self._finalize_turn(session.id, decision, req.user_prompt, output_text)
        return TurnResult(
            session_id=session.id,
            model=model,
            output_text=output_text,
            tool_events=tool_events,
            usage=usage,
        )

    async def execute_stream(self, req: TurnRequest) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "status", "data": "routing"}
        await asyncio.sleep(0)

        session, context, model, system_message, decision = self._prepare_turn(req)
        yield {"type": "model", "data": {"model": model}}
        tool_events: list[dict[str, Any]] = []
        usage = self._init_usage()
        output_text = ""
        phase = "tool_or_answer"
        current_prompt = req.user_prompt
        repair_attempt = 0
        repeated_tool_rounds: dict[str, int] = {}

        while True:
            stage_context = self._with_turn_phase(context, phase, tool_events=tool_events, repair_attempt=repair_attempt)
            stage_text = ""
            yield {"type": "status", "data": "responding"}
            async for chunk in self.model_client.chat_stream(
                model=model,
                system_message=system_message,
                user_prompt=current_prompt,
                context=stage_context,
                images=req.images,
            ):
                stage_text += chunk

            self._record_usage_stage(
                usage=usage,
                phase=phase,
                system_message=system_message,
                user_prompt=current_prompt,
                context=stage_context,
                images=req.images,
                output_text=stage_text,
                raw={},
            )

            tool_calls = parse_tool_calls(stage_text)
            if not tool_calls and self._needs_repair_output(stage_text) and repair_attempt < 1:
                repair_attempt += 1
                current_prompt = self._repair_prompt(current_prompt, repair_attempt)
                continue

            if not tool_calls:
                output_text = stage_text
                if output_text:
                    yield {"type": "delta", "data": output_text}
                break
            signature = self._tool_calls_signature(tool_calls)
            if signature:
                seen = repeated_tool_rounds.get(signature, 0) + 1
                repeated_tool_rounds[signature] = seen
                if seen >= 3:
                    output_text = self._loop_guard_response()
                    yield {"type": "delta", "data": output_text}
                    break

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

            forwarded_tool_events, forwarding_meta = self._prepare_tool_events_for_prompt(
                tool_events=tool_events,
                metadata=req.metadata,
            )
            usage["tool_forwarding"] = forwarding_meta
            phase = "final_response"
            repair_attempt = 0
            current_prompt = self._tool_followup_prompt(req.user_prompt, forwarded_tool_events, forwarding_meta)

        self._finalize_turn(session.id, decision, req.user_prompt, output_text)

        final = TurnResult(
            session_id=session.id,
            model=model,
            output_text=output_text,
            tool_events=tool_events,
            usage=usage,
        )
        yield {"type": "final", "data": final.model_dump(mode="json")}

    def _prepare_turn(self, req: TurnRequest) -> tuple[Any, dict[str, Any], str, str, Any]:
        session = self.sessions.get(req.session_id)
        self.sessions.reset_for_turn(session)
        previous_user_message = self.sessions.get_context(session.id, ["last_user_prompt"]).get("last_user_prompt", "")
        if not isinstance(previous_user_message, str):
            previous_user_message = str(previous_user_message)
        if len(previous_user_message) > 4000:
            previous_user_message = previous_user_message[:3997] + "..."

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
        context = {
            "session": session.model_dump(mode="json"),
            "selected_context": selected_context,
            "conversation": {
                "previous_user_message": previous_user_message,
            },
            "memory": memory_context,
            "context_diagnostics": context_diagnostics,
            "command_catalog": self.tools.command_catalog(),
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
        if name == "RunShell" and "session_id" not in args:
            args["session_id"] = session_id

    def _finalize_turn(self, session_id: str, decision: Any, user_prompt: str, output_text: str) -> None:
        self.sessions.put_context(session_id, "router_decision", decision.model_dump(mode="json"), scope="turn")
        self.sessions.put_context(session_id, "last_user_prompt", user_prompt, scope="pinned")
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
    def _tool_followup_prompt(
        user_prompt: str,
        tool_events: list[dict[str, Any]],
        forwarding_meta: dict[str, Any] | None = None,
    ) -> str:
        prompt = (
            user_prompt
            + "\n\nTool results:\n"
            + json.dumps(tool_events, ensure_ascii=True)
            + "\n\nTurn phase: final_response."
            + "\nYou are now in a final-response turn."
            + "\nThis is not a single-response protocol. Tool results are new input for this next turn."
            + "\nThe user may only see a partial tool preview; provide a clear final conclusion with key findings."
            + "\nYou can run more tools if still required; otherwise provide the final natural-language answer."
            + "\nDo not emit placeholder markdown or fake tool calls."
        )
        if isinstance(forwarding_meta, dict) and int(forwarding_meta.get("compacted_events", 0) or 0) > 0:
            prompt += (
                "\nLarge tool payloads were compacted before this turn to reduce token cost."
                "\nIf you need more detail, call a narrower follow-up tool query."
            )
        return prompt

    def _prepare_tool_events_for_prompt(
        self,
        tool_events: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        mode = str(metadata.get("tool_result_mode", "auto") or "auto").strip().lower()
        if mode not in {"auto", "compact", "full"}:
            mode = "auto"
        threshold_chars = self._tool_forward_threshold_chars(metadata)

        forwarded: list[dict[str, Any]] = []
        compacted_events = 0
        original_chars_total = 0
        forwarded_chars_total = 0
        original_tokens_total = 0
        forwarded_tokens_total = 0
        alerts: list[str] = []

        for idx, event in enumerate(tool_events, start=1):
            if not isinstance(event, dict):
                continue
            name = str(event.get("name", "tool"))
            args = event.get("args", {})
            result = event.get("result")
            result_json = self._safe_json(result)
            original_chars = len(result_json)
            original_tokens = self._estimate_tokens(result_json)
            original_chars_total += original_chars
            original_tokens_total += original_tokens

            should_compact = mode == "compact" or (mode == "auto" and original_chars > threshold_chars)
            forwarded_result = result
            if should_compact and mode != "full":
                compacted_events += 1
                forwarded_result = self._compact_tool_payload(result)
            forwarded_json = self._safe_json(forwarded_result)
            forwarded_chars = len(forwarded_json)
            forwarded_tokens = self._estimate_tokens(forwarded_json)
            forwarded_chars_total += forwarded_chars
            forwarded_tokens_total += forwarded_tokens

            if should_compact and mode != "full":
                alerts.append(
                    f"tool#{idx} {name}: compacted {original_tokens}->{forwarded_tokens} tokens "
                    f"({original_chars}->{forwarded_chars} chars)"
                )

            forwarded.append(
                {
                    "name": name,
                    "args": args,
                    "result": forwarded_result,
                }
            )

        return forwarded, {
            "mode": mode,
            "threshold_chars": threshold_chars,
            "event_count": len(forwarded),
            "compacted_events": compacted_events,
            "original_chars_total": original_chars_total,
            "forwarded_chars_total": forwarded_chars_total,
            "original_tokens_total": original_tokens_total,
            "forwarded_tokens_total": forwarded_tokens_total,
            "alerts": alerts,
        }

    @staticmethod
    def _tool_forward_threshold_chars(metadata: dict[str, Any]) -> int:
        raw = metadata.get("tool_result_max_chars", 6000)
        try:
            value = int(raw)
        except Exception:  # noqa: BLE001
            value = 6000
        return max(500, min(200000, value))

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=True)
        except Exception:  # noqa: BLE001
            return repr(value)

    def _compact_tool_payload(
        self,
        value: Any,
        *,
        depth: int = 0,
        max_depth: int = 4,
        max_items: int = 60,
        max_str_chars: int = 1200,
    ) -> Any:
        if depth >= max_depth:
            return "<truncated-depth>"
        if isinstance(value, str):
            if len(value) <= max_str_chars:
                return value
            return value[: max_str_chars - 3] + "..."
        if isinstance(value, list):
            out = [
                self._compact_tool_payload(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_str_chars=max_str_chars,
                )
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                out.append({"__truncated_items__": len(value) - max_items})
            return out
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[:max_items]:
                out[str(key)] = self._compact_tool_payload(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_str_chars=max_str_chars,
                )
            if len(items) > max_items:
                out["__truncated_keys__"] = len(items) - max_items
            return out
        return value

    @staticmethod
    def _tool_calls_signature(tool_calls: list[dict[str, Any]]) -> str:
        stable = [
            {
                "name": str(call.get("name", "")),
                "args": call.get("args", {}),
            }
            for call in tool_calls
        ]
        return json.dumps(stable, sort_keys=True, ensure_ascii=True)

    @staticmethod
    def _loop_guard_response() -> str:
        return (
            "I detected a repeated tool-call loop and stopped automatic retries.\n"
            "Please refine the request or ask me to continue with a specific next action."
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        clean = (text or "").strip()
        if not clean:
            return 0
        return max(1, int(len(clean) / 4))

    @staticmethod
    def _init_usage() -> dict[str, Any]:
        return {
            "estimated_tokens": {
                "input_total": 0,
                "output_total": 0,
                "total": 0,
                "stages": [],
            },
            "provider_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "stages": [],
            },
            "tool_forwarding": {
                "mode": "auto",
                "threshold_chars": 6000,
                "event_count": 0,
                "compacted_events": 0,
                "original_chars_total": 0,
                "forwarded_chars_total": 0,
                "original_tokens_total": 0,
                "forwarded_tokens_total": 0,
                "alerts": [],
            },
        }

    def _record_usage_stage(
        self,
        usage: dict[str, Any],
        phase: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str],
        output_text: str,
        raw: dict[str, Any],
    ) -> None:
        est = usage.setdefault("estimated_tokens", {})
        stage_no = len(est.setdefault("stages", [])) + 1
        input_system = self._estimate_tokens(system_message)
        input_user = self._estimate_tokens(user_prompt)
        input_context = self._estimate_tokens(json.dumps(context, ensure_ascii=True))
        input_images = 256 * len(images)
        input_total = input_system + input_user + input_context + input_images
        output_total = self._estimate_tokens(output_text)
        est_stage = {
            "stage": stage_no,
            "phase": phase,
            "input": {
                "system": input_system,
                "user_prompt": input_user,
                "context": input_context,
                "images": input_images,
                "total": input_total,
            },
            "output": {
                "text": output_total,
                "total": output_total,
            },
        }
        est["stages"].append(est_stage)
        est["input_total"] = int(est.get("input_total", 0)) + input_total
        est["output_total"] = int(est.get("output_total", 0)) + output_total
        est["total"] = int(est.get("total", 0)) + input_total + output_total

        provider = usage.setdefault("provider_usage", {})
        stages = provider.setdefault("stages", [])
        extracted = self._extract_provider_tokens(raw)
        if extracted is not None:
            provider_name, prompt_tokens, completion_tokens, total_tokens = extracted
            source = "provider"
        else:
            provider_name = self._infer_provider_name(raw)
            prompt_tokens = input_total
            completion_tokens = output_total
            total_tokens = input_total + output_total
            source = "estimated"
        stages.append(
            {
                "stage": stage_no,
                "phase": phase,
                "provider": provider_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "source": source,
            }
        )
        provider["prompt_tokens"] = int(provider.get("prompt_tokens", 0)) + prompt_tokens
        provider["completion_tokens"] = int(provider.get("completion_tokens", 0)) + completion_tokens
        provider["total_tokens"] = int(provider.get("total_tokens", 0)) + total_tokens

    @staticmethod
    def _infer_provider_name(raw: dict[str, Any]) -> str:
        if isinstance(raw, dict):
            provider = raw.get("provider")
            if isinstance(provider, str) and provider.strip():
                return provider.strip()
        return "poe"

    def _extract_provider_tokens(self, raw: dict[str, Any]) -> tuple[str, int, int, int] | None:
        provider_name = self._infer_provider_name(raw)
        if not isinstance(raw, dict):
            return None

        usage = raw.get("usage")
        direct = self._normalize_provider_usage(usage)
        if direct is not None:
            prompt_tokens, completion_tokens, total_tokens = direct
            return provider_name, prompt_tokens, completion_tokens, total_tokens

        # Poe responses can carry usage in raw event payloads; scan nested event objects.
        events = raw.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                for value in event.values():
                    nested = self._find_usage_in_tree(value)
                    if nested is not None:
                        prompt_tokens, completion_tokens, total_tokens = nested
                        return provider_name, prompt_tokens, completion_tokens, total_tokens
        return None

    def _find_usage_in_tree(self, value: Any) -> tuple[int, int, int] | None:
        if isinstance(value, dict):
            if "prompt_tokens" in value or "completion_tokens" in value or "total_tokens" in value:
                usage = self._normalize_provider_usage(value)
                if usage is not None:
                    return usage
            for nested in value.values():
                found = self._find_usage_in_tree(nested)
                if found is not None:
                    return found
            return None
        if isinstance(value, list):
            for nested in value:
                found = self._find_usage_in_tree(nested)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _normalize_provider_usage(value: Any) -> tuple[int, int, int] | None:
        if not isinstance(value, dict):
            return None
        prompt_tokens = int(value.get("prompt_tokens", 0) or 0)
        completion_tokens = int(value.get("completion_tokens", 0) or 0)
        total_tokens = int(value.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
        if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
            return None
        return prompt_tokens, completion_tokens, total_tokens

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
        remainder = re.sub(
            r"(thinking\.{3}(?: \(\d+s elapsed\))?|generating\.{3}(?: \(\d+s elapsed\))?)",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
        if len(remainder.strip()) >= 10:
            # Keep normal answers even when providers prepend progress markers.
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
