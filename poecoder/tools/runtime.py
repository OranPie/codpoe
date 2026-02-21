from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from poecoder.models import (
    MemoryEditRequest,
    MemoryReadRequest,
    MemoryWriteRequest,
    ReviewRequest,
    TaskStartSubagentRequest,
    TurnRequest,
)
from poecoder.services.audit_service import AuditService
from poecoder.services.command_service import CommandService
from poecoder.services.memory_service import MemoryService
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.review_service import ReviewService
from poecoder.services.session_service import SessionService
from poecoder.services.shell_service import ShellService
from poecoder.services.subagent_service import SubagentService
from poecoder.services.usage_service import UsageService
from poecoder.services.wiki_service import WikiService
from poecoder.tools.code_tools import CodeTools
from poecoder.tools.web_tools import WebTools

if TYPE_CHECKING:
    from poecoder.services.task_service import TaskService


@dataclass(slots=True)
class ToolRuntime:
    code_tools: CodeTools
    memory_service: MemoryService
    wiki_service: WikiService
    command_service: CommandService
    subagent_service: SubagentService
    shell_service: ShellService
    audit_service: AuditService
    sessions: SessionService
    model_catalog: ModelCatalog
    web_tools: WebTools
    usage_service: UsageService
    review_service: ReviewService | None = None
    task_service: "TaskService | None" = None

    def command_catalog(self) -> list[dict[str, Any]]:
        return [
            {"name": "ReadRaw", "args": "file,line,end_line?", "effect": "Read source text by lines"},
            {"name": "ReadStruct", "args": "target,language,dependency_depth", "effect": "Read symbol/structure summary"},
            {"name": "ReadRecursive", "args": "seed_files,boundary", "effect": "Expand related files recursively"},
            {"name": "Search", "args": "pattern,file_pattern,boundary,root", "effect": "Regex search with snippets"},
            {"name": "WriteRaw", "args": "file,line,content,append?", "effect": "Insert/append raw text in file"},
            {"name": "WriteReplace", "args": "pattern,replacement,location,max_changes", "effect": "Regex replace in files"},
            {"name": "GetWebRaw", "args": "url,timeout_s,max_chars,headers?", "effect": "Fetch raw web content"},
            {"name": "GetWeb", "args": "url,focus?,timeout_s,max_chars", "effect": "Fetch and summarize web page"},
            {"name": "GetWebFile", "args": "url,save_as?,folder,overwrite,timeout_s,max_bytes", "effect": "Download file from web"},
            {"name": "WriteMemory", "args": "scope,content,tags?,priority,session_id?,project_id?", "effect": "Store memory"},
            {"name": "ReadMemory", "args": "scope?,query?,session_id?,project_id?,limit", "effect": "Read memory entries"},
            {"name": "EditMemory", "args": "entry_id?/query,operation,payload,scope?", "effect": "Edit memory entries"},
            {"name": "DelMemory", "args": "entry_id?/query,scope?", "effect": "Delete memory entries"},
            {"name": "TmpWrite", "args": "name,content,ttl_seconds?", "effect": "Write temporary named content"},
            {"name": "InstallCommand", "args": "name,definition,runtime,args_schema,effect_schema,capabilities,source,signature?,session_id?", "effect": "Install custom command"},
            {"name": "EditCommand", "args": "name,definition?/args_schema?/effect_schema?/capabilities?/signature?,session_id?", "effect": "Patch installed command"},
            {"name": "DelCommand", "args": "name,session_id?", "effect": "Delete installed command"},
            {"name": "ListModels", "args": "refresh?", "effect": "List supported models"},
            {"name": "ChangeModel", "args": "session_id,model", "effect": "Change active session model"},
            {"name": "Review", "args": "session_id,prompt,context_keys?,model?,thinking_level?,thinking_budget?", "effect": "Run reviewer role analysis"},
            {"name": "StartSubAgent", "args": "parent_session_id,model,perm,prompt,context_share,images?,system_message_modifier?", "effect": "Start subagent"},
            {"name": "ReadSubAgent", "args": "agent_id", "effect": "Read subagent state"},
            {"name": "WaitSubAgent", "args": "agent_id,timeout_s?", "effect": "Wait for subagent"},
            {"name": "CancelSubAgent", "args": "agent_id", "effect": "Cancel subagent"},
            {"name": "StartBackgroundTurn", "args": "session_id,user_prompt,system_message?,images?,context_keys?,metadata?", "effect": "Start async turn task"},
            {"name": "StartBackgroundSubAgent", "args": "parent_session_id,model,perm,prompt,images?,context_share?,system_message_modifier?,wait_timeout_s?", "effect": "Start async subagent task"},
            {"name": "ListTasks", "args": "limit?,state?,task_type?", "effect": "List background tasks"},
            {"name": "ReadTaskOutput", "args": "task_id", "effect": "Read task result/error"},
            {"name": "CancelTask", "args": "task_id", "effect": "Cancel background task"},
            {"name": "RunShell", "args": "session_id,command,danger_level,cwd?,timeout_s?", "effect": "Run shell command with policy"},
            {"name": "WikiQuery", "args": "project_id,query,limit?", "effect": "Query project wiki"},
            {"name": "WikiCompact", "args": "project_id", "effect": "Compact wiki docs"},
            {"name": "GetBalance", "args": "", "effect": "Read Poe balance"},
        ]

    async def invoke(self, actor: str, name: str, args: dict[str, Any]) -> Any:
        started = time.perf_counter()
        policy = "allowed"
        status = "ok"
        try:
            self._enforce_model_command_policy(actor=actor, name=name, args=args)
            result = await self._dispatch(name, args)
            return result
        except Exception as exc:  # noqa: BLE001
            policy = "runtime_error"
            status = "error"
            raise exc
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.audit_service.log(
                actor=actor,
                tool_name=name,
                args=args,
                policy_decision=policy,
                result_status=status,
                duration_ms=elapsed_ms,
            )

    async def _dispatch(self, name: str, args: dict[str, Any]) -> Any:
        if name == "ReadRaw":
            return self.code_tools.read_raw(**args)
        if name == "ReadStruct":
            return self.code_tools.read_struct(**args)
        if name in {"ReadRecursive", "ReadRecurisive"}:
            return self.code_tools.read_recursive(**args)
        if name == "WriteRaw":
            return self.code_tools.write_raw(**args)
        if name == "WriteReplace":
            return self.code_tools.write_replace(**args)
        if name == "Search":
            return self.code_tools.search(**args)

        if name == "TmpWrite":
            return self._tmp_write(**args)
        if name == "GetWebRaw":
            return await self.web_tools.get_web_raw(**args)
        if name == "GetWeb":
            return await self.web_tools.get_web(**args)
        if name == "GetWebFile":
            return await self.web_tools.get_web_file(**args)

        if name == "GetBalance":
            return self.usage_service.get_current_balance()

        if name == "ListModels":
            refresh = bool(args.get("refresh", False))
            return {"models": self.model_catalog.list_models(refresh=refresh)}
        if name == "ChangeModel":
            session_id = args.get("session_id")
            model = args.get("model")
            if not isinstance(session_id, str) or not isinstance(model, str):
                raise ValueError("ChangeModel requires session_id and model")
            self.model_catalog.ensure_supported(model)
            updated = self.sessions.change_model(session_id, model)
            return updated.model_dump(mode="json")
        if name == "Review":
            if self.review_service is None:
                raise ValueError("review service unavailable")
            return await self.review_service.run(ReviewRequest(**args))

        if name == "ReadMemory":
            return [
                entry.model_dump(mode="json")
                for entry in self.memory_service.read(MemoryReadRequest(**args))
            ]
        if name == "EditMemory":
            return {"changed": self.memory_service.edit(MemoryEditRequest(**args))}
        if name == "DelMemory":
            delete_args = dict(args)
            delete_args["operation"] = "delete"
            return {"changed": self.memory_service.edit(MemoryEditRequest(**delete_args))}
        if name == "WriteMemory":
            return {"id": self.memory_service.write(MemoryWriteRequest(**args))}

        if name == "InstallCommand":
            install_args = dict(args)
            install_args.pop("session_id", None)
            return self.command_service.install(**install_args)
        if name == "EditCommand":
            patch_args = dict(args)
            patch_args.pop("session_id", None)
            target = patch_args.pop("name")
            return self.command_service.patch(target, patch_args)
        if name == "DelCommand":
            return {"deleted": self.command_service.delete(args["name"])}

        if name == "StartSubAgent":
            return self.subagent_service.start(**args)
        if name == "ReadSubAgent":
            return self.subagent_service.read(args["agent_id"])
        if name == "WaitSubAgent":
            return await self.subagent_service.wait(args["agent_id"], timeout_s=args.get("timeout_s", 60))
        if name == "CancelSubAgent":
            return self.subagent_service.cancel(args["agent_id"])

        if name == "StartBackgroundTurn":
            if self.task_service is None:
                raise ValueError("task service unavailable")
            task = await self.task_service.start_turn(
                TurnRequest(
                    session_id=args["session_id"],
                    user_prompt=args["user_prompt"],
                    system_message=args.get("system_message"),
                    images=args.get("images", []) or [],
                    context_keys=args.get("context_keys", []) or [],
                    metadata=args.get("metadata", {}) or {},
                )
            )
            return task.model_dump(mode="json")
        if name == "StartBackgroundSubAgent":
            if self.task_service is None:
                raise ValueError("task service unavailable")
            task = await self.task_service.start_subagent(
                TaskStartSubagentRequest(
                    parent_session_id=args["parent_session_id"],
                    model=args["model"],
                    perm=args.get("perm", "readonly"),
                    prompt=args["prompt"],
                    images=args.get("images", []) or [],
                    context_share=args.get("context_share", []) or [],
                    system_message_modifier=args.get("system_message_modifier"),
                    wait_timeout_s=args.get("wait_timeout_s", 600),
                )
            )
            return task.model_dump(mode="json")
        if name == "ListTasks":
            if self.task_service is None:
                raise ValueError("task service unavailable")
            return [
                item.model_dump(mode="json")
                for item in self.task_service.list(
                    limit=int(args.get("limit", 50)),
                    state=args.get("state"),
                    task_type=args.get("task_type"),
                )
            ]
        if name in {"ReadTaskOutput", "ReadTask"}:
            if self.task_service is None:
                raise ValueError("task service unavailable")
            return self.task_service.read_output(args["task_id"])
        if name == "CancelTask":
            if self.task_service is None:
                raise ValueError("task service unavailable")
            return self.task_service.cancel(args["task_id"]).model_dump(mode="json")

        if name == "RunShell":
            return (await self.shell_service.run(**args)).model_dump(mode="json")

        if name == "WikiQuery":
            return self.wiki_service.query(**args)
        if name == "WikiCompact":
            return self.wiki_service.compact(**args)

        raise KeyError(f"unknown tool: {name}")

    def _tmp_write(self, name: str, content: str, ttl_seconds: int = 3600) -> dict[str, Any]:
        from datetime import datetime, timedelta, timezone

        expires_at = datetime.now(tz=timezone.utc) + timedelta(seconds=ttl_seconds)
        self.audit_service.db.execute(
            """
            INSERT INTO tmp_writes(name, content, expires_at, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET content = excluded.content, expires_at = excluded.expires_at
            """,
            (name, content, expires_at.isoformat(), datetime.now(tz=timezone.utc).isoformat()),
        )
        return {"name": name, "expires_at": expires_at.isoformat()}

    def _enforce_model_command_policy(self, actor: str, name: str, args: dict[str, Any]) -> None:
        if actor != "model":
            return
        if name not in {"InstallCommand", "EditCommand", "DelCommand"}:
            return
        session_id = args.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{name} requires session_id for model actor")
        session = self.sessions.get(session_id)
        if not session.allow_model_command_create:
            raise PermissionError("model command creation is disabled for this session")



def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("@tool "):
            continue
        try:
            _, rest = line.split("@tool ", 1)
            name, raw_args = rest.split(" ", 1)
            args = json.loads(raw_args)
            calls.append({"name": name, "args": args})
        except Exception:
            continue
    return calls
