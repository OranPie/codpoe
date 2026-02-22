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
            {"name": "Help", "args": "tool_name?/query?", "effect": "Get detailed help for tool usage"},
            {"name": "ReadRaw", "args": "file,line,end_line?", "effect": "Read source text by lines"},
            {"name": "ReadStruct", "args": "target,language,dependency_depth", "effect": "Read symbol/structure summary"},
            {"name": "ReadRecursive", "args": "seed_files,boundary", "effect": "Expand related files recursively"},
            {"name": "Search", "args": "pattern,file_pattern,boundary,root", "effect": "Regex search with snippets"},
            {"name": "ListFile", "args": "path?,pattern?,recursive?,include_dirs?,limit?", "effect": "List files under current/target directory"},
            {"name": "ChangeWorkDir", "args": "path", "effect": "Change tool working directory"},
            {"name": "WriteRaw", "args": "file,line,content,append?", "effect": "Insert/append raw text in file"},
            {"name": "WriteReplace", "args": "pattern,replacement,location,max_changes", "effect": "Regex replace in files"},
            {"name": "GetWebRaw", "args": "url,timeout_s,max_chars,headers?,selector?,regex?,max_matches?", "effect": "Fetch web content with optional selector/regex filtering"},
            {"name": "GetWeb", "args": "url,focus?,timeout_s,max_chars,selector?,regex?,max_matches?,download_if_large?,download_folder?", "effect": "Fetch summarized web content with optional filtering/local-download fallback"},
            {"name": "GetWebFile", "args": "url,save_as?,folder,overwrite,timeout_s,max_bytes", "effect": "Download file from web"},
            {"name": "WriteMemory", "args": "scope,content,tags?,priority,session_id?,project_id?", "effect": "Store memory"},
            {"name": "ReadMemory", "args": "scope?,query?,session_id?,project_id?,tags_any?,min_priority?,include_content?,max_content_chars?,limit", "effect": "Read memory entries with filters"},
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
            {"name": "WikiQuery", "args": "project_id,query,topic?,include_content?,include_meta?,max_content_chars?,limit?", "effect": "Query project wiki with filters"},
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
        if name == "Help":
            return self._tool_help(
                tool_name=args.get("tool_name") or args.get("name"),
                query=args.get("query"),
            )
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

    def _tool_help(self, tool_name: Any = None, query: Any = None) -> dict[str, Any]:
        catalog = self.command_catalog()
        indexed = {str(item.get("name", "")).lower(): item for item in catalog}
        if "readrecursive" in indexed:
            indexed["readrecurisive"] = indexed["readrecursive"]

        requested = ""
        if isinstance(tool_name, str) and tool_name.strip():
            requested = tool_name.strip().lower()
        elif isinstance(query, str) and query.strip():
            requested = query.strip().lower()

        if not requested:
            names = [str(item.get("name", "")) for item in catalog if str(item.get("name", ""))]
            return {
                "usage": "Call Help with tool_name, for example: Help(tool_name='StartLeaderRun').",
                "tool_count": len(names),
                "tools": names,
                "complex_tools": [
                    "WriteReplace",
                    "ReadRecursive",
                    "RunShell",
                    "StartSubAgent",
                    "StartLeaderRun",
                    "StartBackgroundTurn",
                    "StartBackgroundSubAgent",
                    "Review",
                    "ReadMemory",
                    "WikiQuery",
                ],
            }

        entry = indexed.get(requested)
        if entry is None:
            matches = [
                str(item.get("name", ""))
                for item in catalog
                if requested in str(item.get("name", "")).lower()
            ]
            return {
                "found": False,
                "requested": requested,
                "matches": matches[:20],
                "hint": "Use exact tool name from command_catalog, then call Help(tool_name=...).",
            }

        name = str(entry.get("name", ""))
        payload: dict[str, Any] = {
            "found": True,
            "tool": name,
            "args": str(entry.get("args", "")),
            "effect": str(entry.get("effect", "")),
        }
        override = self._tool_help_overrides().get(name, {})
        guidance = override.get("guidance")
        if isinstance(guidance, list) and guidance:
            payload["guidance"] = [str(item) for item in guidance]
        else:
            payload["guidance"] = [
                "Use exact arg names from args string.",
                "Keep scope narrow to reduce latency and token cost.",
            ]
        example = override.get("example")
        if isinstance(example, str) and example.strip():
            payload["example"] = example.strip()
        related = override.get("related")
        if isinstance(related, list) and related:
            payload["related"] = [str(item) for item in related]
        return payload

    @staticmethod
    def _tool_help_overrides() -> dict[str, dict[str, Any]]:
        return {
            "Help": {
                "guidance": [
                    "Call Help(tool_name='ToolName') to get detailed usage.",
                    "Call Help() to list available tool names and complex-tool suggestions.",
                ],
                "example": "@tool Help {\"tool_name\":\"WriteReplace\"}",
            },
            "WriteReplace": {
                "guidance": [
                    "Set location narrowly (file or small folder) before replacing.",
                    "Use max_changes to cap risk on broad regex patterns.",
                    "Verify with Search or ReadRaw after mutation.",
                ],
                "example": "@tool WriteReplace {\"pattern\":\"foo\",\"replacement\":\"bar\",\"location\":\"poecoder/cli.py\",\"max_changes\":2}",
                "related": ["Search", "ReadRaw", "WriteRaw"],
            },
            "ReadRecursive": {
                "guidance": [
                    "Seed with the fewest files possible for focused expansion.",
                    "Use low boundary first and increase only if dependencies are missing.",
                    "Prefer ReadStruct for fast symbol-level overview before recursive expansion.",
                ],
                "example": "@tool ReadRecursive {\"seed_files\":[\"poecoder/cli.py\"],\"boundary\":2}",
                "related": ["ReadStruct", "ReadRaw", "Search"],
            },
            "RunShell": {
                "guidance": [
                    "Set danger_level=0 for read-only commands.",
                    "Set cwd when command must run in a specific directory.",
                    "Keep commands deterministic and bounded by timeout_s.",
                ],
                "example": "@tool RunShell {\"session_id\":\"<session>\",\"command\":\"pwd\",\"danger_level\":0,\"timeout_s\":10}",
                "related": ["ListFile", "ChangeWorkDir"],
            },
            "StartSubAgent": {
                "guidance": [
                    "Use readonly perm unless mutation is required.",
                    "Share only minimal context keys needed for the subtask.",
                    "Use system_message_modifier to tighten subtask behavior.",
                ],
                "example": "@tool StartSubAgent {\"parent_session_id\":\"<session>\",\"model\":\"assistant\",\"perm\":\"readonly\",\"prompt\":\"Summarize failing tests\",\"context_share\":[\"last_user_prompt\"]}",
                "related": ["ReadSubAgent", "WaitSubAgent", "CancelSubAgent"],
            },
            "StartLeaderRun": {
                "guidance": [
                    "Define non-overlapping job scopes to avoid edit conflicts.",
                    "Use verify_command to run final integration checks.",
                    "Tune max_parallel and per_job_timeout_s for stability.",
                ],
                "example": "@tool StartLeaderRun {\"session_id\":\"<session>\",\"goal\":\"Implement feature X with tests\"}",
                "related": ["ReadLeaderRun", "ListLeaderJobs", "WaitLeaderRun", "CancelLeaderRun"],
            },
            "StartBackgroundTurn": {
                "guidance": [
                    "Use for long-running prompts that should not block foreground flow.",
                    "Pass context_keys to keep background context minimal.",
                    "Read output with ReadTaskOutput.",
                ],
                "example": "@tool StartBackgroundTurn {\"session_id\":\"<session>\",\"user_prompt\":\"run deep analysis\"}",
                "related": ["ListTasks", "ReadTaskOutput", "CancelTask"],
            },
            "StartBackgroundSubAgent": {
                "guidance": [
                    "Use for long-running delegated subtasks.",
                    "Prefer readonly perm for analysis-only work.",
                    "Use wait_timeout_s to bound blocking behavior.",
                ],
                "example": "@tool StartBackgroundSubAgent {\"parent_session_id\":\"<session>\",\"model\":\"assistant\",\"perm\":\"readonly\",\"prompt\":\"analyze logs\"}",
                "related": ["ListTasks", "ReadTaskOutput", "CancelTask"],
            },
            "Review": {
                "guidance": [
                    "Use targeted prompt and context_keys for high-signal review.",
                    "Set model/thinking only when defaults are insufficient.",
                    "Expect findings-first output with severity ordering.",
                ],
                "example": "@tool Review {\"session_id\":\"<session>\",\"prompt\":\"review this patch for regressions\"}",
            },
            "ReadMemory": {
                "guidance": [
                    "Use tags_any and min_priority to cut noise.",
                    "Set include_content=false when only ids/meta are needed.",
                    "Use max_content_chars to avoid large payloads.",
                ],
                "example": "@tool ReadMemory {\"scope\":\"project\",\"project_id\":\"default\",\"tags_any\":[\"design\"],\"limit\":8}",
                "related": ["WriteMemory", "EditMemory", "DelMemory"],
            },
            "WikiQuery": {
                "guidance": [
                    "Use topic to constrain retrieval before large sessions.",
                    "Disable include_content/include_meta when you only need ids.",
                    "Set max_content_chars to keep token usage predictable.",
                ],
                "example": "@tool WikiQuery {\"project_id\":\"default\",\"query\":\"router\",\"topic\":\"architecture\",\"limit\":5}",
                "related": ["WikiCompact", "WriteMemory"],
            },
            "SetBaseUri": {
                "guidance": [
                    "Use provider=poe or provider=openai.",
                    "Set full provider base URI before refreshing model list.",
                    "Validate with ListModels or /apistatus afterwards.",
                ],
                "example": "@tool SetBaseUri {\"provider\":\"openai\",\"base_uri\":\"https://api.openai.com/v1\"}",
                "related": ["ListModels"],
            },
            "GetWebRaw": {
                "guidance": [
                    "Prefer selector or regex to reduce payload size and token cost.",
                    "Use max_matches to cap large extraction results.",
                    "If page is still huge, switch to GetWebFile and analyze locally.",
                ],
                "example": "@tool GetWebRaw {\"url\":\"https://example.com\",\"selector\":\"article\",\"max_chars\":8000}",
                "related": ["GetWeb", "GetWebFile", "ReadRaw", "Search"],
            },
            "GetWeb": {
                "guidance": [
                    "Use selector/regex/focus together for precise extraction.",
                    "Set download_if_large=true to save oversized pages for local analysis.",
                    "Use ReadRaw/Search on downloaded files for iterative extraction.",
                ],
                "example": "@tool GetWeb {\"url\":\"https://example.com\",\"focus\":\"release notes\",\"selector\":\"main\",\"max_chars\":6000}",
                "related": ["GetWebRaw", "GetWebFile", "ReadRaw", "Search"],
            },
            "GetWebFile": {
                "guidance": [
                    "Use for very large pages/binaries to avoid sending huge payloads to the model.",
                    "Download first, then inspect with ReadRaw/Search in narrow windows.",
                    "Keep overwrite=false unless replacing an existing download intentionally.",
                ],
                "example": "@tool GetWebFile {\"url\":\"https://example.com/huge.html\",\"folder\":\"downloads\",\"overwrite\":false}",
                "related": ["GetWeb", "ReadRaw", "Search"],
            },
        }

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
