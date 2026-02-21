from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from poecoder.models import (
    LeaderRunRequest,
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
    from poecoder.services.leader_service import LeaderService
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
    settings: Any
    model_client: Any
    review_service: ReviewService | None = None
    task_service: "TaskService | None" = None
    leader_service: "LeaderService | None" = None
    cwd: str = "."

    def command_catalog(self) -> list[dict[str, Any]]:
        return [
            {"name": "ReadRaw", "args": "file,line,end_line?", "effect": "Read source text by lines"},
            {"name": "ReadStruct", "args": "target,language,dependency_depth", "effect": "Read symbol/structure summary"},
            {"name": "ReadRecursive", "args": "seed_files,boundary", "effect": "Expand related files recursively"},
            {"name": "Search", "args": "pattern,file_pattern,boundary,root", "effect": "Regex search with snippets"},
            {"name": "ListFile", "args": "path?,pattern?,recursive?,include_dirs?,limit?", "effect": "List files under current/target directory"},
            {"name": "ChangeWorkDir", "args": "path", "effect": "Change tool working directory"},
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
            {"name": "StartLeaderRun", "args": "session_id,goal,jobs?,planner_model?,worker_model?,max_parallel?,per_job_timeout_s?,context_keys?,verify_command?,verify_cwd?,verify_timeout_s?,verify_danger_level?", "effect": "Start leader orchestration run"},
            {"name": "ReadLeaderRun", "args": "run_id", "effect": "Read leader run status/result"},
            {"name": "ListLeaderJobs", "args": "run_id", "effect": "List jobs under a leader run"},
            {"name": "WaitLeaderRun", "args": "run_id,timeout_s?", "effect": "Wait for leader run completion"},
            {"name": "CancelLeaderRun", "args": "run_id", "effect": "Cancel active leader run"},
            {"name": "RunShell", "args": "session_id,command,danger_level,cwd?,timeout_s?", "effect": "Run shell command with policy"},
            {"name": "Exit", "args": "reason?", "effect": "Signal exit request from model/user"},
            {"name": "WikiQuery", "args": "project_id,query,limit?", "effect": "Query project wiki"},
            {"name": "WikiCompact", "args": "project_id", "effect": "Compact wiki docs"},
            {"name": "GetBalance", "args": "", "effect": "Read Poe balance"},
            {"name": "SetBaseUri", "args": "provider,base_uri", "effect": "Set provider base URI (poe|openai)"},
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
        if name == "ListFile":
            return self._list_file(**args)
        if name == "ChangeWorkDir":
            path = args.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("ChangeWorkDir requires path")
            return self._change_workdir(path)
        if name == "ReadRaw":
            read_args = dict(args)
            read_args["file"] = self._resolve_tool_path(str(read_args["file"]))
            return self.code_tools.read_raw(**read_args)
        if name == "ReadStruct":
            read_args = dict(args)
            read_args["target"] = self._resolve_tool_path(str(read_args["target"]))
            return self.code_tools.read_struct(**read_args)
        if name in {"ReadRecursive", "ReadRecurisive"}:
            read_args = dict(args)
            read_args["seed_files"] = [self._resolve_tool_path(str(item)) for item in args.get("seed_files", [])]
            return self.code_tools.read_recursive(**read_args)
        if name == "WriteRaw":
            write_args = dict(args)
            write_args["file"] = self._resolve_tool_path(str(write_args["file"]))
            return self.code_tools.write_raw(**write_args)
        if name == "WriteReplace":
            write_args = dict(args)
            location = str(write_args.get("location", "."))
            write_args["location"] = self._resolve_tool_path(location)
            return self.code_tools.write_replace(**write_args)
        if name == "Search":
            search_args = dict(args)
            root = str(search_args.get("root", "."))
            search_args["root"] = self._resolve_tool_path(root)
            return self.code_tools.search(**search_args)

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
        if name == "SetBaseUri":
            provider = str(args.get("provider", "")).strip().lower()
            base_uri = str(args.get("base_uri") or args.get("base_url") or "").strip()
            if not base_uri:
                raise ValueError("SetBaseUri requires base_uri")
            if provider == "poe":
                self.settings.poe_api_url = base_uri
                self.model_client.update_poe(base_url=base_uri)
                return {"provider": "poe", "base_uri": self.model_client.api_url}
            if provider in {"openai", "oa"}:
                self.settings.openai_api_url = base_uri
                self.model_client.update_openai(base_url=base_uri)
                return {"provider": "openai", "base_uri": self.model_client.openai_api_url}
            raise ValueError("SetBaseUri requires provider=poe|openai")

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

        if name == "StartLeaderRun":
            if self.leader_service is None:
                raise ValueError("leader service unavailable")
            run = self.leader_service.start(LeaderRunRequest(**args))
            return run.model_dump(mode="json")
        if name == "ReadLeaderRun":
            if self.leader_service is None:
                raise ValueError("leader service unavailable")
            return self.leader_service.read(args["run_id"]).model_dump(mode="json")
        if name == "ListLeaderJobs":
            if self.leader_service is None:
                raise ValueError("leader service unavailable")
            return [item.model_dump(mode="json") for item in self.leader_service.list_jobs(args["run_id"])]
        if name == "WaitLeaderRun":
            if self.leader_service is None:
                raise ValueError("leader service unavailable")
            timeout_s = int(args.get("timeout_s", 120))
            return (await self.leader_service.wait(args["run_id"], timeout_s=timeout_s)).model_dump(mode="json")
        if name == "CancelLeaderRun":
            if self.leader_service is None:
                raise ValueError("leader service unavailable")
            return self.leader_service.cancel(args["run_id"]).model_dump(mode="json")

        if name == "RunShell":
            run_args = dict(args)
            cwd = run_args.get("cwd")
            if cwd is None:
                run_args["cwd"] = str(self._absolute_cwd())
            else:
                run_args["cwd"] = str(self._absolute_cwd(str(cwd)))
            return (await self.shell_service.run(**run_args)).model_dump(mode="json")

        if name == "Exit":
            return {"exit": True, "reason": str(args.get("reason", "requested"))}

        if name == "WikiQuery":
            return self.wiki_service.query(**args)
        if name == "WikiCompact":
            return self.wiki_service.compact(**args)

        raise KeyError(f"unknown tool: {name}")

    def _list_file(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
        include_dirs: bool = False,
        limit: int = 200,
    ) -> dict[str, Any]:
        return self.code_tools.list_files(
            path=self._resolve_tool_path(path),
            pattern=pattern,
            recursive=recursive,
            include_dirs=include_dirs,
            limit=int(limit),
        )

    def _change_workdir(self, path: str) -> dict[str, Any]:
        resolved = self.code_tools._resolve(self._resolve_tool_path(path))
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"not a directory: {resolved}")
        self.cwd = str(resolved.relative_to(self.code_tools.root))
        if self.cwd == ".":
            self.cwd = "."
        return {"cwd": self.cwd, "absolute": str(resolved)}

    def _resolve_tool_path(self, path: str) -> str:
        value = (path or ".").strip()
        if not value:
            value = "."
        p = Path(value)
        if p.is_absolute():
            return value
        if self.cwd in {"", "."}:
            return value
        return str(Path(self.cwd) / value)

    def _absolute_cwd(self, path: str | None = None) -> Path:
        target = self.cwd if path is None else self._resolve_tool_path(path)
        return self.code_tools._resolve(target)

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
    decoder = json.JSONDecoder()

    def _append_call(name: Any, args: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        if not isinstance(args, dict):
            return
        calls.append({"name": name.strip(), "args": args})

    marker = "@tool "
    pos = 0
    while True:
        idx = text.find(marker, pos)
        if idx < 0:
            break
        cursor = idx + len(marker)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        end = cursor
        while end < len(text) and (text[end].isalnum() or text[end] in {"_", "-"}):
            end += 1
        name = text[cursor:end]
        if not name:
            pos = idx + len(marker)
            continue
        brace = text.find("{", end)
        if brace < 0:
            pos = end
            continue
        try:
            args, consumed = decoder.raw_decode(text[brace:])
        except Exception:
            pos = brace + 1
            continue
        _append_call(name, args)
        pos = brace + consumed

    # Fallback parser for non-compliant JSON-style tool calls sometimes emitted by models.
    # Example: {"tool_name":"ListModels","args":{"refresh":false}}
    seen = {(item["name"], json.dumps(item["args"], sort_keys=True, ensure_ascii=True)) for item in calls}
    pos = 0
    while True:
        brace = text.find("{", pos)
        if brace < 0:
            break
        try:
            obj, consumed = decoder.raw_decode(text[brace:])
        except Exception:
            pos = brace + 1
            continue
        pos = brace + consumed
        if not isinstance(obj, dict):
            continue
        tool_name = obj.get("tool_name")
        tool_args = obj.get("args")
        if not isinstance(tool_name, str) or not isinstance(tool_args, dict):
            continue
        key = (tool_name.strip(), json.dumps(tool_args, sort_keys=True, ensure_ascii=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append({"name": tool_name.strip(), "args": tool_args})
    return calls
