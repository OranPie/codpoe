from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poecoder.backend.agent_runtime import AgentRuntime
from poecoder.backend.models import (
    AgentStartRequest,
    AgentTemplateUpsertRequest,
    AgentView,
    MemoryReadRequest,
    MemoryWriteRequest,
    SessionCreateRequest,
    SessionTurnRequest,
    SessionTurnResponse,
    WorkflowArxivRequest,
    WorkflowArxivResponse,
)
from poecoder.backend.prompting import DEFAULT_TEMPLATES
from poecoder.backend.store import AgentStore


@dataclass(slots=True)
class BackendEngine:
    store: AgentStore
    runtime: AgentRuntime

    def bootstrap_defaults(self) -> None:
        for item in DEFAULT_TEMPLATES:
            self.store.upsert_template(AgentTemplateUpsertRequest(**item))

    # session conversation
    def create_session(self, req: SessionCreateRequest):
        return self.store.create_session(req)

    def list_sessions(self, limit: int = 20):
        return self.store.list_sessions(limit=limit)

    def get_session(self, session_id: str):
        return self.store.get_session(session_id)

    def list_session_messages(self, session_id: str, limit: int = 30):
        self.store.get_session(session_id)
        return self.store.list_session_messages(session_id, limit=limit)

    async def run_session_turn(self, session_id: str, req: SessionTurnRequest) -> SessionTurnResponse:
        self.store.get_session(session_id)
        self.store.add_session_message(session_id, "user", req.prompt)
        run = self.runtime.start(
            AgentStartRequest(
                name="conversation-root",
                goal=req.prompt,
                session_id=session_id,
                model=req.model,
                scope=["."],
                max_steps=200,
            )
        )
        done = await self.runtime.wait(run.id)
        output = done.final_output if done.final_output.strip() else done.error
        self.store.add_session_message(session_id, "assistant", output)
        payload = self.runtime.get(done.id, event_limit=500)
        events = payload.get("events", [])
        steps = self._extract_steps(done.id, events=events if isinstance(events, list) else None)
        metrics = self.summarize_agent_events(events if isinstance(events, list) else [])
        ask_payload = self._extract_last_ask(events if isinstance(events, list) else [])
        return SessionTurnResponse(
            session_id=session_id,
            agent_id=done.id,
            model=done.model,
            output=output,
            steps=steps,
            context_compaction_note=str(payload.get("context_compaction_note", "")),
            agent_metrics=metrics,
            ask=ask_payload,
        )

    # agents
    def start_agent(self, req: AgentStartRequest) -> AgentView:
        return self.runtime.start(req)

    async def wait_agent(self, agent_id: str, timeout_s: int = 120) -> dict[str, Any]:
        from poecoder.backend.models import AgentWaitRequest

        await self.runtime.wait(agent_id, AgentWaitRequest(timeout_s=timeout_s))
        return self.runtime.get(agent_id)

    def get_agent(self, agent_id: str, event_limit: int = 200) -> dict[str, Any]:
        return self.runtime.get(agent_id, event_limit=event_limit)

    def cancel_agent(self, agent_id: str) -> AgentView:
        return self.runtime.cancel(agent_id)

    def get_agent_events(self, agent_id: str, limit: int = 200) -> list[dict[str, Any]]:
        payload = self.runtime.get(agent_id, event_limit=limit)
        events = payload.get("events", [])
        return events if isinstance(events, list) else []

    # templates
    def upsert_template(self, req: AgentTemplateUpsertRequest):
        return self.store.upsert_template(req)

    def list_templates(self):
        return self.store.list_templates()

    # memory
    def write_memory(self, req: MemoryWriteRequest) -> int:
        return self.store.write_memory(req)

    def read_memory(self, req: MemoryReadRequest):
        return self.store.read_memory(req)

    # workflow
    async def run_arxiv_workflow(self, req: WorkflowArxivRequest) -> WorkflowArxivResponse:
        self.store.get_session(req.session_id)
        self.store.add_session_message(req.session_id, "user", f"[workflow/arxiv] {req.query}")
        self.store.write_memory(
            MemoryWriteRequest(
                scope="session",
                session_id=req.session_id,
                key="workflow.arxiv.query",
                value={"query": req.query, "max_results": req.max_results},
                tags=["workflow", "arxiv"],
            )
        )
        init = self.runtime.start(
            AgentStartRequest(
                name="init-agent",
                goal=f"Initialize task for query: {req.query}. Return concise kickoff.",
                session_id=req.session_id,
                template_name="shell-reader",
                expected_output_schema={
                    "kickoff": "string",
                    "plan": ["short next steps"],
                },
                max_steps=2,
            )
        )
        init_done = await self.runtime.wait(init.id)
        self._store_workflow_stage_output(req.session_id, "init", init_done)

        search = self.runtime.start(
            AgentStartRequest(
                name="arxiv-finder-agent",
                goal=(
                    "Use shell and web endpoints (curl) to find arXiv papers for query: "
                    f"{req.query}. Return candidate pdf links and summaries."
                ),
                session_id=req.session_id,
                template_name="web-searcher",
                expected_output_schema={
                    "papers": [
                        {
                            "title": "string",
                            "pdf_url": "string",
                            "why_relevant": "string",
                        }
                    ]
                },
                max_steps=4,
            )
        )
        search_done = await self.runtime.wait(search.id)
        self._store_workflow_stage_output(req.session_id, "search", search_done)

        download = self.runtime.start(
            AgentStartRequest(
                name="arxiv-download-agent",
                goal=(
                    "Use wget/curl to download top relevant PDFs for query "
                    f"{req.query}. Keep under max_results={req.max_results}."
                ),
                session_id=req.session_id,
                template_name="wget-downloader",
                expected_output_schema={
                    "downloads": [
                        {
                            "pdf_url": "string",
                            "saved_path": "string",
                            "ok": True,
                            "error": "string-if-failed",
                        }
                    ]
                },
                max_steps=4,
            )
        )
        download_done = await self.runtime.wait(download.id)
        self._store_workflow_stage_output(req.session_id, "download", download_done)

        final = self.runtime.start(
            AgentStartRequest(
                name="final-report-agent",
                goal=(
                    "Build final report using session memory keys workflow.arxiv.query, "
                    "workflow.arxiv.init, workflow.arxiv.search, workflow.arxiv.download. "
                    "Set clear status (done/partial/failed), include downloaded files, failures, and next actions."
                ),
                session_id=req.session_id,
                expected_output_schema={
                    "status": "done|partial|failed",
                    "summary": "string",
                    "downloaded_files": ["paths"],
                    "failures": ["string"],
                    "next_actions": ["string"],
                },
                max_steps=2,
            )
        )
        final_done = await self.runtime.wait(final.id)
        self._store_workflow_stage_output(req.session_id, "final", final_done)
        self.store.add_session_message(req.session_id, "assistant", final_done.final_output or final_done.error)
        return WorkflowArxivResponse(
            session_id=req.session_id,
            init_agent_id=init.id,
            search_agent_id=search_done.id,
            download_agent_id=download_done.id,
            final_agent_id=final_done.id,
            final_output=final_done.final_output or final_done.error,
        )

    def _store_workflow_stage_output(self, session_id: str, stage: str, agent: AgentView) -> None:
        payload = {
            "agent_id": agent.id,
            "status": agent.status,
            "output": (agent.final_output or "")[:12000],
            "error": agent.error,
        }
        self.store.write_memory(
            MemoryWriteRequest(
                scope="session",
                session_id=session_id,
                key=f"workflow.arxiv.{stage}",
                value=payload,
                tags=["workflow", "arxiv", stage],
            )
        )

    def summarize_agent_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        model_actions = 0
        runshell = 0
        spawn = 0
        ask = 0
        note = 0
        tool_define = 0
        tool_call = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        estimated_cost_usd = 0.0
        last_runshell: list[dict[str, Any]] = []
        last_tool_calls: list[dict[str, Any]] = []

        for event in events:
            if not isinstance(event, dict):
                continue
            et = event.get("event_type")
            payload = event.get("payload", {})
            if et == "model_action":
                model_actions += 1
                if isinstance(payload, dict):
                    usage = payload.get("usage", {})
                    if isinstance(usage, dict):
                        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                        total_tokens += int(usage.get("total_tokens", 0) or 0)
                    estimated_cost_usd += float(payload.get("estimated_cost_usd", 0.0) or 0.0)
            elif et == "runshell":
                runshell += 1
                if isinstance(payload, dict):
                    last_runshell.append(
                        {
                            "command": str(payload.get("command", "")),
                            "exit_code": payload.get("exit_code"),
                            "allowed": bool(payload.get("allowed", True)),
                            "progress": str(payload.get("progress", "")),
                        }
                    )
            elif et == "spawn":
                spawn += 1
            elif et == "ask":
                ask += 1
            elif et == "note":
                note += 1
            elif et == "tool_define":
                tool_define += 1
            elif et == "tool_call":
                tool_call += 1
                if isinstance(payload, dict):
                    last_tool_calls.append(
                        {
                            "tool_name": str(payload.get("tool_name", "")),
                            "command": str(payload.get("command", "")),
                            "exit_code": payload.get("exit_code"),
                            "allowed": bool(payload.get("allowed", True)),
                            "progress": str(payload.get("progress", "")),
                        }
                    )

        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "model_actions": model_actions,
            "runshell_count": runshell,
            "spawn_count": spawn,
            "ask_count": ask,
            "note_count": note,
            "tool_define_count": tool_define,
            "tool_call_count": tool_call,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 8),
            "last_runshell": last_runshell[-5:],
            "last_tool_calls": last_tool_calls[-5:],
        }

    def _extract_steps(self, agent_id: str, events: list[dict[str, Any]] | None = None) -> list[str]:
        if events is None:
            payload = self.runtime.get(agent_id)
            events = payload.get("events", [])
        out: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            et = event.get("event_type")
            if et == "runshell":
                shell_payload = event.get("payload", {})
                if isinstance(shell_payload, dict):
                    out.append(f"runshell exit={shell_payload.get('exit_code')}")
            elif et == "spawn":
                out.append("spawn child agent")
            elif et == "model_action":
                parsed = event.get("payload", {}).get("parsed", {})
                if isinstance(parsed, dict) and "action" in parsed:
                    out.append(f"action={parsed['action']}")
            elif et == "ask":
                out.append("ask user clarification")
            elif et == "note":
                note_payload = event.get("payload", {})
                if isinstance(note_payload, dict):
                    detail = str(note_payload.get("detail", "")).strip()
                    progress = str(note_payload.get("progress", "")).strip()
                    text = detail or progress
                    if text:
                        out.append(f"note: {text[:120]}")
                    else:
                        out.append("note")
                else:
                    out.append("note")
            elif et == "tool_define":
                out.append("define reusable tool")
            elif et == "tool_call":
                out.append("call reusable tool")
        return out[:12]

    def _extract_last_ask(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            if str(event.get("event_type", "")) != "ask":
                continue
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                return payload
        return None
