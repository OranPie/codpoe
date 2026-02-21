from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["coding", "chat", "planning"]
MemoryScope = Literal["session", "project", "global"]
DangerLevel = Literal[0, 1, 2]
SubagentState = Literal["running", "completed", "cancelled", "failed"]


class SessionCreateRequest(BaseModel):
    mode: Mode = "coding"
    active_model: str | None = "auto"
    policy_profile: str = "default"
    project_id: str = "default"


class SessionResponse(BaseModel):
    id: str
    mode: Mode
    active_model: str
    policy_profile: str
    project_id: str
    created_at: datetime
    updated_at: datetime


class ContextPutRequest(BaseModel):
    key: str
    value: Any
    scope: str = "turn"
    ttl_seconds: int | None = None


class ChangeModelRequest(BaseModel):
    model: str

class TurnRequest(BaseModel):
    session_id: str
    user_prompt: str
    system_message: str | None = None
    direct_model: bool = False
    images: list[str] = Field(default_factory=list)
    context_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnResult(BaseModel):
    session_id: str
    model: str
    output_text: str
    tool_events: list[dict[str, Any]] = Field(default_factory=list)


class MemoryWriteRequest(BaseModel):
    scope: MemoryScope
    content: str
    tags: list[str] = Field(default_factory=list)
    priority: int = 0
    session_id: str | None = None
    project_id: str | None = None


class MemoryEditRequest(BaseModel):
    entry_id: int | None = None
    query: str | None = None
    operation: Literal["replace", "append", "delete"]
    payload: str = ""
    scope: MemoryScope | None = None


class MemoryReadRequest(BaseModel):
    query: str | None = None
    scope: MemoryScope | None = None
    session_id: str | None = None
    project_id: str | None = None
    limit: int = 20


class MemoryEntryView(BaseModel):
    id: int
    scope: MemoryScope
    session_id: str | None
    project_id: str | None
    tags: list[str]
    priority: int
    content: str
    created_at: datetime
    updated_at: datetime


class WikiIngestRequest(BaseModel):
    project_id: str
    topic: str
    content: str
    source: str = "user"


class WikiQueryRequest(BaseModel):
    project_id: str
    query: str
    limit: int = 10


class WikiCompactRequest(BaseModel):
    project_id: str


class CommandInstallRequest(BaseModel):
    name: str
    definition: str
    runtime: Literal["py", "sh"]
    args_schema: dict[str, Any] = Field(default_factory=dict)
    effect_schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    source: str = "user"
    signature: str | None = None


class CommandPatchRequest(BaseModel):
    definition: str | None = None
    args_schema: dict[str, Any] | None = None
    effect_schema: dict[str, Any] | None = None
    capabilities: list[str] | None = None
    signature: str | None = None


class ShellRunRequest(BaseModel):
    session_id: str
    command: str
    danger_level: DangerLevel = 0
    cwd: str | None = None
    timeout_s: int = 60


class ShellRunResponse(BaseModel):
    allowed: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    policy_reason: str = ""


class SubagentStartRequest(BaseModel):
    parent_session_id: str
    model: str
    perm: Literal["readonly", "standard", "privileged"] = "readonly"
    prompt: str
    images: list[str] = Field(default_factory=list)
    context_share: list[str] = Field(default_factory=list)
    system_message_modifier: str | None = None


class SubagentResponse(BaseModel):
    id: str
    parent_session_id: str
    model: str
    perm: str
    state: SubagentState
    prompt: str
    images: list[str] = Field(default_factory=list)
    result: str | None
    created_at: datetime
    updated_at: datetime


class SubagentReadRequest(BaseModel):
    query: str | None = None


class TmpWriteRequest(BaseModel):
    name: str
    content: str
    ttl_seconds: int = 3600


class SearchRequest(BaseModel):
    pattern: str
    file_pattern: str = "*"
    boundary: int = 2
    root: str = "."


class ReadRawRequest(BaseModel):
    file: str
    line: int = 1
    end_line: int | None = None


class WriteRawRequest(BaseModel):
    file: str
    line: int
    content: str
    append: bool = False


class ReplaceRequest(BaseModel):
    pattern: str
    replacement: str
    location: str = "."
    max_changes: int = 0


class ReadStructRequest(BaseModel):
    target: str
    language: Literal["python", "javascript", "typescript"]
    dependency_depth: int = 1


class ReadRecursiveRequest(BaseModel):
    seed_files: list[str]
    boundary: int = 2




class GetWebRawRequest(BaseModel):
    url: str
    timeout_s: int = 20
    max_chars: int = 200000
    headers: dict[str, str] = Field(default_factory=dict)


class GetWebRequest(BaseModel):
    url: str
    focus: str | None = None
    timeout_s: int = 20
    max_chars: int = 16000


class GetWebFileRequest(BaseModel):
    url: str
    save_as: str | None = None
    folder: str = "downloads"
    overwrite: bool = False
    timeout_s: int = 60
    max_bytes: int = 20000000



class TaskStartSubagentRequest(SubagentStartRequest):
    wait_timeout_s: int = 600


class TaskView(BaseModel):
    id: str
    task_type: str
    state: str
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ReadTaskOutputRequest(BaseModel):
    task_id: str


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class RouterDecision(BaseModel):
    classifier_model: str
    selected_model: str
    complexity: Literal["small", "medium", "large"]
    reason: str
