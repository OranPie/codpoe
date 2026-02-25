from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentState = Literal["queued", "running", "waiting_child", "completed", "failed", "cancelled"]
MemoryScope = Literal["user", "session"]
AskInputMode = Literal["text", "single", "multiple"]


class SessionCreateRequest(BaseModel):
    title: str = ""


class SessionView(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionMessageView(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime


class SessionTurnRequest(BaseModel):
    prompt: str
    model: str | None = None


class SessionTurnResponse(BaseModel):
    session_id: str
    agent_id: str
    model: str
    output: str
    steps: list[str] = Field(default_factory=list)
    context_compaction_note: str = ""
    agent_metrics: dict[str, Any] = Field(default_factory=dict)
    ask: dict[str, Any] | None = None


class AgentTemplateUpsertRequest(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    default_scope: list[str] = Field(default_factory=lambda: ["."])
    default_model: str = "auto"


class AgentTemplateView(BaseModel):
    name: str
    description: str
    system_prompt: str
    default_scope: list[str] = Field(default_factory=list)
    default_model: str
    created_at: datetime
    updated_at: datetime


class AgentStartRequest(BaseModel):
    name: str = "runtime-agent"
    goal: str
    session_id: str | None = None
    parent_agent_id: str | None = None
    model: str | None = None
    template_name: str | None = None
    scope: list[str] = Field(default_factory=lambda: ["."])
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=50, ge=1)


class AgentView(BaseModel):
    id: str
    session_id: str | None = None
    parent_agent_id: str | None = None
    depth: int
    name: str
    goal: str
    status: AgentState
    model: str
    template_name: str | None = None
    scope: list[str] = Field(default_factory=list)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    max_steps: int
    final_output: str = ""
    error: str = ""
    created_at: datetime
    updated_at: datetime


class AgentWaitRequest(BaseModel):
    timeout_s: int = Field(default=120, ge=1, le=3600)


class MemoryWriteRequest(BaseModel):
    scope: MemoryScope
    key: str
    value: Any
    tags: list[str] = Field(default_factory=list)
    session_id: str | None = None
    user_key: str = "default"


class MemoryReadRequest(BaseModel):
    scope: MemoryScope
    key: str | None = None
    session_id: str | None = None
    user_key: str = "default"
    limit: int = Field(default=40, ge=1, le=200)


class MemoryEntryView(BaseModel):
    id: int
    scope: MemoryScope
    key: str
    value: Any
    tags: list[str]
    session_id: str | None = None
    user_key: str | None = None
    updated_at: datetime


class RunShellRequest(BaseModel):
    command: str
    cwd: str = "."
    timeout_s: int = Field(default=30, ge=1, le=600)
    danger_ack: bool = False


class RunShellResult(BaseModel):
    allowed: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    blocked_reason: str = ""


class WorkflowArxivRequest(BaseModel):
    session_id: str
    query: str
    max_results: int = Field(default=5, ge=1, le=20)


class WorkflowArxivResponse(BaseModel):
    session_id: str
    init_agent_id: str
    search_agent_id: str
    download_agent_id: str
    final_agent_id: str
    final_output: str


class ApiKeyRequest(BaseModel):
    api_key: str


class ProviderSecretsSaveRequest(BaseModel):
    user_key: str
    poe_api_key: str | None = None
    openai_api_key: str | None = None
    poe_api_url: str | None = None
    openai_api_url: str | None = None


class ProviderSecretsLoadRequest(BaseModel):
    user_key: str


class SearchWebRequest(BaseModel):
    query: str
    limit: int = Field(default=8, ge=1, le=30)
    timeout_s: int = Field(default=20, ge=1, le=120)
    max_snippet_chars: int = Field(default=280, ge=80, le=1200)


class SearchArxivRequest(BaseModel):
    query: str
    max_results: int = Field(default=8, ge=1, le=50)
    timeout_s: int = Field(default=20, ge=1, le=120)


class GetWebRequest(BaseModel):
    url: str
    focus: str | None = None
    timeout_s: int = Field(default=20, ge=1, le=120)
    max_chars: int = Field(default=16000, ge=500, le=120000)
    selector: str | None = None
    regex: str | None = None
    max_matches: int = Field(default=60, ge=1, le=500)
    download_if_large: bool = False
    download_folder: str = "downloads"


class DownloadUrlsRequest(BaseModel):
    urls: list[str]
    folder: str = "downloads"
    overwrite: bool = False
    timeout_s: int = Field(default=60, ge=5, le=600)
    max_bytes: int = Field(default=20_000_000, ge=1_000_000, le=200_000_000)
    max_files: int = Field(default=8, ge=1, le=40)
