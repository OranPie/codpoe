from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from typing import Any

from poecoder.backend.models import (
    AgentStartRequest,
    AgentView,
    AgentWaitRequest,
    RunShellRequest,
)
from poecoder.backend.prompting import build_agent_prompt
from poecoder.backend.shell_runtime import ShellRuntime
from poecoder.backend.store import AgentStore
from poecoder.services.model_clients import PoeModelClient


@dataclass(slots=True)
class AgentRuntime:
    store: AgentStore
    model_client: PoeModelClient
    shell: ShellRuntime
    default_model: str
    max_depth: int = 2
    _tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    _context_notes: dict[str, str] = field(default_factory=dict)
    _price_usd_per_1m_tokens: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "openai/gpt-5-mini": (0.25, 2.0),
            "openai/gpt-5": (1.25, 10.0),
            "openai/gpt-5.1": (1.25, 10.0),
            "openai/gpt-5.2": (1.25, 10.0),
            "openai/gpt-5.3": (1.25, 10.0),
            "openai/gpt-4.1-mini": (0.3, 1.2),
            "openai/gpt-4.1": (2.0, 8.0),
            "openai/o4-mini": (1.1, 4.4),
        }
    )

    def start(self, req: AgentStartRequest) -> AgentView:
        depth = self.store.max_agent_depth(req.parent_agent_id)
        if req.parent_agent_id:
            depth += 1
        model = self._resolve_model(req.model)
        view = self.store.create_agent_run(req, model=model, depth=depth)
        task = asyncio.create_task(self._run(view.id))
        self._tasks[view.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(view.id, None))
        return view

    def cancel(self, agent_id: str) -> AgentView:
        view = self.store.get_agent(agent_id)
        if view.status in {"completed", "failed", "cancelled"}:
            return view
        task = self._tasks.get(agent_id)
        if task is not None and not task.done():
            task.cancel()
        return self.store.update_agent_state(
            agent_id,
            "cancelled",
            error="cancelled_by_user",
        )

    async def wait(self, agent_id: str, req: AgentWaitRequest | None = None) -> AgentView:
        timeout_s = req.timeout_s if req is not None else 120
        task = self._tasks.get(agent_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
            except asyncio.TimeoutError:
                pass
        return self.store.get_agent(agent_id)

    def get(self, agent_id: str, event_limit: int = 200) -> dict[str, Any]:
        view = self.store.get_agent(agent_id)
        return {
            "agent": view.model_dump(mode="json"),
            "events": self.store.list_agent_events(agent_id, limit=max(1, event_limit)),
            "context_compaction_note": self._context_notes.get(agent_id, ""),
        }

    async def _run(self, agent_id: str) -> None:
        view = self.store.update_agent_state(agent_id, "running")
        try:
            if view.depth > self.max_depth:
                self.store.update_agent_state(
                    agent_id,
                    "failed",
                    error=f"max depth exceeded ({self.max_depth})",
                )
                return

            template_prompt = None
            if view.template_name:
                try:
                    template_prompt = self.store.get_template(str(view.template_name)).system_prompt
                except KeyError:
                    template_prompt = None

            session_messages: list[dict[str, str]] = []
            session_mem: list[dict[str, Any]] = []
            if view.session_id:
                session_messages = [
                    {"role": item.role, "content": item.content}
                    for item in self.store.list_session_messages(view.session_id, limit=30)
                ]
                session_mem = [item.model_dump(mode="json") for item in self.store.read_memory(
                    self._memory_read_req(scope="session", session_id=view.session_id)
                )]
            user_mem = [item.model_dump(mode="json") for item in self.store.read_memory(self._memory_read_req(scope="user"))]

            prompt_pack = build_agent_prompt(
                goal=view.goal,
                scope=view.scope,
                expected_output_schema=view.expected_output_schema,
                template_prompt=template_prompt,
                conversation_messages=session_messages,
                user_memory=user_mem,
                session_memory=session_mem,
            )
            self._context_notes[agent_id] = prompt_pack.context_compaction_note

            observations: list[dict[str, Any]] = []
            final_output = ""
            for step in range(1, view.max_steps + 1):
                context = {
                    "agent_id": view.id,
                    "step": step,
                    "max_steps": view.max_steps,
                    "observations": observations[-10:],
                    "expected_output_schema": view.expected_output_schema,
                    "tools": self._tool_context(view.session_id, limit=30),
                }
                user_prompt = prompt_pack.user_prompt + f"\n\nStep {step}/{view.max_steps}. Return next action JSON."
                reply = await self.model_client.chat(
                    model=view.model,
                    system_message=prompt_pack.system_prompt,
                    user_prompt=user_prompt,
                    context=context,
                    images=[],
                )
                action = self._parse_action(reply.text)
                usage = reply.raw.get("usage", {}) if isinstance(reply.raw, dict) else {}
                est_cost = self._estimate_cost_usd(view.model, usage)
                self.store.append_agent_event(
                    view.id,
                    "model_action",
                    {
                        "step": step,
                        "raw": reply.text[:4000],
                        "parsed": action,
                        "usage": usage if isinstance(usage, dict) else {},
                        "estimated_cost_usd": est_cost,
                    },
                )
                kind = str(action.get("action", "")).strip().lower()
                if kind == "final":
                    final_output = str(action.get("output", "")).strip()
                    command_log = self._command_log_from_observations(observations)
                    if command_log:
                        if final_output:
                            final_output = f"{final_output}\n\nExecuted commands:\n{command_log}"
                        else:
                            final_output = f"Executed commands:\n{command_log}"
                    self._persist_session_memory(view, action)
                    self.store.update_agent_state(view.id, "completed", final_output=final_output, error="")
                    return
                if kind == "note":
                    progress = str(action.get("progress", "")).strip()
                    detail = str(action.get("detail", "")).strip()
                    next_hint = str(action.get("next", "")).strip()
                    if not progress and not detail:
                        progress = "working"
                    note_payload = {
                        "progress": progress,
                        "detail": detail,
                        "next": next_hint,
                    }
                    self.store.append_agent_event(view.id, "note", note_payload)
                    observations.append(
                        {
                            "type": "note",
                            "progress": progress,
                            "detail": detail,
                            "next": next_hint,
                        }
                    )
                    continue
                if kind == "runshell":
                    progress = str(action.get("progress", "")).strip()
                    no_spawn_reason = str(action.get("no_spawn_reason", "")).strip()
                    # Soft policy guard: root agent should not jump into shell on first step
                    # unless it explains why spawning is unnecessary.
                    if view.depth == 0 and step == 1 and len(no_spawn_reason) < 8:
                        warning = {
                            "type": "runshell",
                            "error": "spawn_first_policy_violation",
                            "hint": "spawn a child agent first, or provide no_spawn_reason for single-command tasks",
                            "progress": progress,
                        }
                        observations.append(warning)
                        self.store.append_agent_event(view.id, "policy_warning", warning)
                        continue
                    shell_req = RunShellRequest(
                        command=str(action.get("command", "")),
                        cwd=str(action.get("cwd", ".")),
                        timeout_s=int(action.get("timeout_s", 30) or 30),
                        danger_ack=bool(action.get("danger_ack", False)),
                    )
                    if not shell_req.command.strip():
                        observations.append(
                            {
                                "type": "runshell",
                                "error": "empty command",
                                "progress": progress,
                            }
                        )
                        continue
                    result = await self.shell.run(shell_req)
                    stdout_preview = self._preview_text_lines(result.stdout, max_lines=3)
                    stderr_preview = self._preview_text_lines(result.stderr, max_lines=3)
                    shell_payload = result.model_dump(mode="json")
                    shell_payload["command"] = shell_req.command
                    shell_payload["cwd"] = shell_req.cwd
                    shell_payload["progress"] = progress
                    shell_payload["no_spawn_reason"] = no_spawn_reason
                    shell_payload["stdout_preview"] = stdout_preview
                    shell_payload["stderr_preview"] = stderr_preview
                    self.store.append_agent_event(view.id, "runshell", shell_payload)
                    observations.append(
                        {
                            "type": "runshell",
                            "progress": progress,
                            "command": shell_req.command,
                            "exit_code": result.exit_code,
                            "allowed": result.allowed,
                            "stdout": result.stdout[:3000],
                            "stderr": result.stderr[:1000],
                            "stdout_preview": stdout_preview,
                            "stderr_preview": stderr_preview,
                        }
                    )
                    continue
                if kind == "spawn":
                    progress = str(action.get("progress", "")).strip()
                    if view.depth >= self.max_depth:
                        observations.append(
                            {
                                "type": "spawn",
                                "progress": progress,
                                "error": f"depth limit {self.max_depth} reached",
                            }
                        )
                        continue
                    child_req = AgentStartRequest(
                        name=str(action.get("name", "child-agent"))[:80],
                        goal=str(action.get("goal", "")).strip(),
                        session_id=view.session_id,
                        parent_agent_id=view.id,
                        model=action.get("model"),
                        template_name=action.get("template_name"),
                        scope=action.get("scope") if isinstance(action.get("scope"), list) else view.scope,
                        expected_output_schema={},
                        max_steps=int(action.get("max_steps", 4) or 4),
                    )
                    child = self.start(child_req)
                    self.store.update_agent_state(view.id, "waiting_child")
                    child_done = await self.wait(child.id, AgentWaitRequest(timeout_s=300))
                    extra_wait_s = 0
                    while child_done.status in {"queued", "running", "waiting_child"} and extra_wait_s < 900:
                        self.store.append_agent_event(
                            view.id,
                            "spawn_wait_extend",
                            {
                                "child_agent_id": child.id,
                                "waited_extra_seconds": extra_wait_s,
                                "reason": "child still running",
                            },
                        )
                        child_done = await self.wait(child.id, AgentWaitRequest(timeout_s=120))
                        extra_wait_s += 120
                    self.store.update_agent_state(view.id, "running")
                    child_payload = self.get(child_done.id, event_limit=600)
                    child_events = child_payload.get("events", [])
                    if not isinstance(child_events, list):
                        child_events = []
                    child_commands = self._extract_command_entries(child_events)
                    child_command_summary = self._command_entries_to_text(child_commands)
                    child_output_text = child_done.final_output[:2000]
                    if child_command_summary:
                        if child_output_text:
                            child_output_text = f"{child_output_text}\n\n[child commands]\n{child_command_summary}"
                        else:
                            child_output_text = f"[child commands]\n{child_command_summary}"
                    observations.append(
                        {
                            "type": "spawn",
                            "progress": progress,
                            "child_agent_id": child_done.id,
                            "child_status": child_done.status,
                            "child_output": child_output_text,
                            "child_commands": child_commands[:10],
                            "child_command_summary": child_command_summary,
                        }
                    )
                    self.store.append_agent_event(view.id, "spawn", observations[-1])
                    continue
                if kind == "ask":
                    progress = str(action.get("progress", "")).strip()
                    ask_payload = self._normalize_ask_payload(action, agent_id=view.id, step=step)
                    if not ask_payload.get("question"):
                        observations.append(
                            {
                                "type": "ask",
                                "progress": progress,
                                "error": "missing question",
                            }
                        )
                        self.store.append_agent_event(view.id, "ask_invalid", observations[-1])
                        continue
                    ask_payload["progress"] = progress
                    observations.append(
                        {
                            "type": "ask",
                            "progress": progress,
                            "question": ask_payload.get("question", ""),
                            "input_mode": ask_payload.get("input_mode", "text"),
                            "options": ask_payload.get("options", []),
                        }
                    )
                    self.store.append_agent_event(view.id, "ask", ask_payload)
                    final_output = self._format_ask_output(ask_payload)
                    self._persist_session_memory(view, action)
                    self.store.update_agent_state(view.id, "completed", final_output=final_output, error="")
                    return
                if kind == "define_tool":
                    progress = str(action.get("progress", "")).strip()
                    name = str(action.get("name", "")).strip()[:80]
                    language = str(action.get("language", "sh")).strip().lower()
                    description = str(action.get("description", "")).strip()[:300]
                    script = str(action.get("script", "")).strip()
                    args_schema = action.get("args_schema")
                    if not isinstance(args_schema, dict):
                        args_schema = {}
                    if not name or language not in {"sh", "python"} or not script:
                        payload = {
                            "type": "define_tool",
                            "progress": progress,
                            "error": "invalid tool definition",
                            "name": name,
                            "language": language,
                        }
                        observations.append(payload)
                        self.store.append_agent_event(view.id, "tool_define_invalid", payload)
                        continue
                    tool = self.store.upsert_tool(
                        session_id=view.session_id,
                        name=name,
                        language=language,
                        description=description or f"{language} tool",
                        script=script,
                        args_schema=args_schema,
                        created_by_agent_id=view.id,
                    )
                    payload = {
                        "tool_name": tool.get("name"),
                        "language": tool.get("language"),
                        "description": tool.get("description"),
                        "args_schema": tool.get("args_schema", {}),
                        "progress": progress,
                    }
                    self.store.append_agent_event(view.id, "tool_define", payload)
                    observations.append(
                        {
                            "type": "tool_define",
                            "progress": progress,
                            "tool_name": tool.get("name"),
                            "language": tool.get("language"),
                            "description": tool.get("description"),
                        }
                    )
                    continue
                if kind == "call_tool":
                    progress = str(action.get("progress", "")).strip()
                    tool_name = str(action.get("name", "")).strip()[:80]
                    args = action.get("args")
                    if not isinstance(args, dict):
                        args = {}
                    tool = self.store.get_tool(session_id=view.session_id, name=tool_name)
                    if tool is None:
                        payload = {
                            "type": "tool_call",
                            "progress": progress,
                            "error": "tool not found",
                            "tool_name": tool_name,
                        }
                        observations.append(payload)
                        self.store.append_agent_event(view.id, "tool_call_missing", payload)
                        continue
                    command, render_error = self._render_tool_command(tool, args)
                    if render_error:
                        payload = {
                            "type": "tool_call",
                            "progress": progress,
                            "error": render_error,
                            "tool_name": tool_name,
                        }
                        observations.append(payload)
                        self.store.append_agent_event(view.id, "tool_call_invalid", payload)
                        continue
                    shell_req = RunShellRequest(
                        command=command,
                        cwd=str(action.get("cwd", ".")),
                        timeout_s=int(action.get("timeout_s", 60) or 60),
                        danger_ack=bool(action.get("danger_ack", False)),
                    )
                    result = await self.shell.run(shell_req)
                    stdout_preview = self._preview_text_lines(result.stdout, max_lines=3)
                    stderr_preview = self._preview_text_lines(result.stderr, max_lines=3)
                    payload = result.model_dump(mode="json")
                    payload["tool_name"] = tool_name
                    payload["tool_language"] = str(tool.get("language", ""))
                    payload["tool_description"] = str(tool.get("description", ""))
                    payload["tool_args"] = args
                    payload["command"] = command
                    payload["cwd"] = shell_req.cwd
                    payload["progress"] = progress
                    payload["stdout_preview"] = stdout_preview
                    payload["stderr_preview"] = stderr_preview
                    self.store.append_agent_event(view.id, "tool_call", payload)
                    observations.append(
                        {
                            "type": "tool_call",
                            "progress": progress,
                            "tool_name": tool_name,
                            "command": command,
                            "exit_code": result.exit_code,
                            "allowed": result.allowed,
                            "stdout": result.stdout[:3000],
                            "stderr": result.stderr[:1000],
                            "stdout_preview": stdout_preview,
                            "stderr_preview": stderr_preview,
                        }
                    )
                    continue

                # fallback: treat text as final to avoid loops
                final_output = str(action.get("output") or reply.text).strip()
                self.store.update_agent_state(view.id, "completed", final_output=final_output, error="")
                return

            if not final_output:
                final_output = (
                    "Agent reached max steps without final action. "
                    "Please refine goal or split into smaller child agents."
                )
            self.store.update_agent_state(view.id, "completed", final_output=final_output, error="max_steps_exceeded")
        except asyncio.CancelledError:
            self.store.update_agent_state(view.id, "cancelled", error="cancelled_by_user")
            raise
        except Exception as exc:  # noqa: BLE001
            self.store.update_agent_state(view.id, "failed", error=f"runtime_error: {exc}")

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"action": "final", "output": ""}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:  # noqa: BLE001
                pass
        return {"action": "final", "output": raw}

    @staticmethod
    def _memory_read_req(*, scope: str, session_id: str | None = None) -> Any:
        from poecoder.backend.models import MemoryReadRequest

        if scope == "session":
            return MemoryReadRequest(scope="session", session_id=session_id, limit=40)
        return MemoryReadRequest(scope="user", user_key="default", limit=40)

    def _persist_session_memory(self, view: AgentView, action: dict[str, Any]) -> None:
        if not view.session_id:
            return
        items = action.get("session_memory")
        if not isinstance(items, list):
            return
        from poecoder.backend.models import MemoryWriteRequest

        for item in items[:20]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            self.store.write_memory(
                MemoryWriteRequest(
                    scope="session",
                    session_id=view.session_id,
                    key=key[:120],
                    value=item.get("value"),
                    tags=item.get("tags") if isinstance(item.get("tags"), list) else [],
                )
            )

    def _resolve_model(self, requested: str | None) -> str:
        model = (requested or "").strip()
        if model in {"", "auto"}:
            return self.default_model
        return model

    @staticmethod
    def _extract_command_entries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type", ""))
            if event_type not in {"runshell", "tool_call"}:
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            command = str(payload.get("command", "")).strip()
            if not command:
                continue
            tool_name = str(payload.get("tool_name", "")).strip()
            if tool_name:
                command = f"[tool:{tool_name}] {command}"
            out.append(
                {
                    "command": command,
                    "exit_code": payload.get("exit_code"),
                    "allowed": bool(payload.get("allowed", True)),
                }
            )
        return out

    @staticmethod
    def _command_entries_to_text(entries: list[dict[str, Any]], max_items: int = 8) -> str:
        lines: list[str] = []
        for item in entries[: max(1, max_items)]:
            if not isinstance(item, dict):
                continue
            cmd = str(item.get("command", "")).strip()
            if not cmd:
                continue
            lines.append(
                f"- exit={item.get('exit_code')} allowed={item.get('allowed')} cmd={cmd}"
            )
        return "\n".join(lines)

    def _command_log_from_observations(self, observations: list[dict[str, Any]], max_items: int = 12) -> str:
        lines: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", "")).strip().lower()
            if kind == "runshell":
                cmd = str(item.get("command", "")).strip()
                if not cmd:
                    continue
                lines.append(
                    f"- exit={item.get('exit_code')} allowed={item.get('allowed')} cmd={cmd}"
                )
                continue
            if kind == "tool_call":
                cmd = str(item.get("command", "")).strip()
                tool_name = str(item.get("tool_name", "")).strip()
                if not cmd:
                    continue
                prefix = f"[tool:{tool_name}] " if tool_name else ""
                lines.append(
                    f"- exit={item.get('exit_code')} allowed={item.get('allowed')} cmd={prefix}{cmd}"
                )
                continue
            if kind == "spawn":
                child_id = str(item.get("child_agent_id", "")).strip()
                child_summary = str(item.get("child_command_summary", "")).strip()
                if not child_summary:
                    continue
                prefix = f"[child {child_id[:12]}] " if child_id else "[child] "
                for line in child_summary.splitlines():
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                    lines.append(prefix + line_clean)
        return "\n".join(lines[: max(1, max_items)])

    @staticmethod
    def _normalize_ask_payload(action: dict[str, Any], *, agent_id: str, step: int) -> dict[str, Any]:
        question = str(action.get("question", "")).strip()
        input_mode_raw = str(action.get("input_mode", "text")).strip().lower()
        input_mode = input_mode_raw if input_mode_raw in {"text", "single", "multiple"} else "text"
        allow_free_text = bool(action.get("allow_free_text", False))
        max_select = int(action.get("max_select", 1) or 1)
        if input_mode == "single":
            max_select = 1
        if input_mode == "text":
            max_select = 1

        options_raw = action.get("options", [])
        options: list[dict[str, str]] = []
        if isinstance(options_raw, list):
            for idx, item in enumerate(options_raw[:8], start=1):
                if isinstance(item, str):
                    label = item.strip()
                    if not label:
                        continue
                    options.append({"id": str(idx), "label": label})
                    continue
                if not isinstance(item, dict):
                    continue
                option_id = str(item.get("id", "")).strip() or str(idx)
                label = str(item.get("label", "")).strip()
                hint = str(item.get("hint", "")).strip()
                if not label:
                    continue
                payload = {"id": option_id[:40], "label": label[:120]}
                if hint:
                    payload["hint"] = hint[:160]
                options.append(payload)
        if input_mode in {"single", "multiple"} and not options:
            input_mode = "text"

        ask_id = str(action.get("ask_id", "")).strip()
        if not ask_id:
            ask_id = f"{agent_id[:8]}-s{step}"

        return {
            "ask_id": ask_id,
            "question": question[:500],
            "input_mode": input_mode,
            "options": options,
            "allow_free_text": allow_free_text,
            "max_select": max(1, min(max_select, 8)),
            "placeholder": str(action.get("placeholder", "")).strip()[:200],
            "why": str(action.get("why", "")).strip()[:500],
        }

    @staticmethod
    def _format_ask_output(ask_payload: dict[str, Any]) -> str:
        question = str(ask_payload.get("question", "")).strip()
        mode = str(ask_payload.get("input_mode", "text")).strip() or "text"
        ask_id = str(ask_payload.get("ask_id", "")).strip()
        lines = [f"[ASK][{ask_id}] {question}", f"mode={mode}"]
        options = ask_payload.get("options", [])
        if isinstance(options, list) and options:
            lines.append("options:")
            for item in options[:8]:
                if not isinstance(item, dict):
                    continue
                option_id = str(item.get("id", "")).strip()
                label = str(item.get("label", "")).strip()
                hint = str(item.get("hint", "")).strip()
                if not option_id and not label:
                    continue
                line = f"- {option_id}: {label}".strip()
                if hint:
                    line += f" ({hint})"
                lines.append(line)
        why = str(ask_payload.get("why", "")).strip()
        if why:
            lines.append(f"why: {why}")
        lines.append("Please answer in next message.")
        return "\n".join(lines)

    def _estimate_cost_usd(self, model: str, usage: Any) -> float:
        if not isinstance(usage, dict):
            return 0.0
        prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        if prompt <= 0 and completion <= 0:
            return 0.0
        model_name = model.strip().lower()
        matched: tuple[float, float] | None = None
        for key, price in self._price_usd_per_1m_tokens.items():
            key_clean = key.lower()
            key_suffix = key_clean.split("/", 1)[1] if "/" in key_clean else key_clean
            if model_name.startswith(key_clean) or model_name.startswith(key_suffix):
                matched = price
                break
        if matched is None:
            matched = (1.0, 4.0)
        prompt_rate, completion_rate = matched
        cost = (prompt / 1_000_000.0) * prompt_rate + (completion / 1_000_000.0) * completion_rate
        return round(cost, 8)

    @staticmethod
    def _render_tool_command(tool: dict[str, Any], args: dict[str, Any]) -> tuple[str, str]:
        language = str(tool.get("language", "")).strip().lower()
        script = str(tool.get("script", "")).strip()
        if language not in {"sh", "python"}:
            return "", f"unsupported tool language: {language}"
        if not script:
            return "", "tool script is empty"
        safe_args = args if isinstance(args, dict) else {}
        if language == "sh":
            rendered = script
            for key, value in safe_args.items():
                token = "{{" + str(key).strip() + "}}"
                rendered = rendered.replace(token, shlex.quote(str(value)))
            return rendered, ""
        payload_json = json.dumps(safe_args, ensure_ascii=True)
        py_bootstrap = (
            "import json, os\n"
            "args = json.loads(os.environ.get('AGENT_TOOL_ARGS', '{}'))\n"
            + script
        )
        command = (
            f"AGENT_TOOL_ARGS={shlex.quote(payload_json)} "
            f"python3 -c {shlex.quote(py_bootstrap)}"
        )
        return command, ""

    def _tool_context(self, session_id: str | None, limit: int = 30) -> list[dict[str, Any]]:
        rows = self.store.list_tools(session_id=session_id, limit=max(1, limit))
        out: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            script = str(item.get("script", "")).strip()
            out.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "language": str(item.get("language", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "args_schema": item.get("args_schema", {}),
                    "script_preview": script[:600],
                }
            )
        return out

    @staticmethod
    def _preview_text_lines(text: str, max_lines: int = 3, max_cols: int = 220) -> list[str]:
        if not text:
            return []
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        out: list[str] = []
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            out.append(clean[:max(20, max_cols)])
            if len(out) >= max(1, max_lines):
                break
        return out
