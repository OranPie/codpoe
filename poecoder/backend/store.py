from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poecoder.backend.models import (
    AgentStartRequest,
    AgentTemplateUpsertRequest,
    AgentTemplateView,
    AgentView,
    MemoryEntryView,
    MemoryReadRequest,
    MemoryWriteRequest,
    SessionCreateRequest,
    SessionMessageView,
    SessionView,
)


def utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ac_memory_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope TEXT NOT NULL,
  user_key TEXT,
  session_id TEXT,
  mem_key TEXT NOT NULL,
  value_json TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_templates (
  name TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  default_scope_json TEXT NOT NULL,
  default_model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_runs (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  parent_agent_id TEXT,
  depth INTEGER NOT NULL,
  name TEXT NOT NULL,
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  model TEXT NOT NULL,
  template_name TEXT,
  scope_json TEXT NOT NULL,
  expected_output_schema_json TEXT NOT NULL,
  max_steps INTEGER NOT NULL,
  final_output TEXT NOT NULL,
  error TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_tools (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  name TEXT NOT NULL,
  language TEXT NOT NULL,
  description TEXT NOT NULL,
  script TEXT NOT NULL,
  args_schema_json TEXT NOT NULL,
  created_by_agent_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(session_id, name)
);

CREATE INDEX IF NOT EXISTS idx_session_messages_time ON session_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_scope_session ON ac_memory_entries(scope, session_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_scope_user ON ac_memory_entries(scope, user_key, updated_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_parent ON agent_runs(parent_agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_events_agent ON agent_events(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_tools_session_name ON agent_tools(session_id, name);
"""


@dataclass(slots=True)
class AgentStore:
    db_path: Path
    _conn: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # sessions
    def create_session(self, req: SessionCreateRequest) -> SessionView:
        sid = str(uuid.uuid4())
        now = utcnow_iso()
        title = req.title.strip()[:120]
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_sessions(id, title, created_at, updated_at) VALUES(?, ?, ?, ?)",
                (sid, title, now, now),
            )
            self._conn.commit()
        return self.get_session(sid)

    def get_session(self, session_id: str) -> SessionView:
        row = self._one("SELECT * FROM agent_sessions WHERE id = ?", (session_id,))
        if row is None:
            raise KeyError(session_id)
        return SessionView(
            id=str(row["id"]),
            title=str(row["title"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    def list_sessions(self, limit: int = 20) -> list[SessionView]:
        rows = self._all("SELECT * FROM agent_sessions ORDER BY updated_at DESC LIMIT ?", (max(1, limit),))
        return [
            SessionView(
                id=str(row["id"]),
                title=str(row["title"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                updated_at=datetime.fromisoformat(str(row["updated_at"])),
            )
            for row in rows
        ]

    def add_session_message(self, session_id: str, role: str, content: str) -> None:
        now = utcnow_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_messages(id, session_id, role, content, created_at) VALUES(?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), session_id, role, content, now),
            )
            self._conn.execute(
                "UPDATE agent_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            self._conn.commit()

    def list_session_messages(self, session_id: str, limit: int = 30) -> list[SessionMessageView]:
        rows = self._all(
            "SELECT role, content, created_at FROM session_messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, max(1, limit)),
        )
        rows.reverse()
        return [
            SessionMessageView(
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
            )
            for row in rows
        ]

    # memory
    def write_memory(self, req: MemoryWriteRequest) -> int:
        now = utcnow_iso()
        payload = self._dumps(req.value)
        tags = self._dumps(req.tags)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO ac_memory_entries(scope, user_key, session_id, mem_key, value_json, tags_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req.scope,
                    req.user_key if req.scope == "user" else None,
                    req.session_id if req.scope == "session" else None,
                    req.key,
                    payload,
                    tags,
                    now,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def read_memory(self, req: MemoryReadRequest) -> list[MemoryEntryView]:
        clauses = ["scope = ?"]
        params: list[Any] = [req.scope]
        if req.scope == "user":
            clauses.append("user_key = ?")
            params.append(req.user_key)
        if req.scope == "session":
            clauses.append("session_id = ?")
            params.append(req.session_id)
        if req.key:
            clauses.append("mem_key = ?")
            params.append(req.key)
        params.append(max(1, req.limit))
        query = "SELECT * FROM ac_memory_entries WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC LIMIT ?"
        rows = self._all(query, tuple(params))
        return [
            MemoryEntryView(
                id=int(row["id"]),
                scope=str(row["scope"]),
                key=str(row["mem_key"]),
                value=self._loads(str(row["value_json"])),
                tags=self._loads(str(row["tags_json"])),
                session_id=row["session_id"],
                user_key=row["user_key"],
                updated_at=datetime.fromisoformat(str(row["updated_at"])),
            )
            for row in rows
        ]

    # templates
    def upsert_template(self, req: AgentTemplateUpsertRequest) -> AgentTemplateView:
        now = utcnow_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_templates(name, description, system_prompt, default_scope_json, default_model, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    system_prompt = excluded.system_prompt,
                    default_scope_json = excluded.default_scope_json,
                    default_model = excluded.default_model,
                    updated_at = excluded.updated_at
                """,
                (
                    req.name,
                    req.description,
                    req.system_prompt,
                    self._dumps(req.default_scope),
                    req.default_model,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_template(req.name)

    def get_template(self, name: str) -> AgentTemplateView:
        row = self._one("SELECT * FROM agent_templates WHERE name = ?", (name,))
        if row is None:
            raise KeyError(name)
        return AgentTemplateView(
            name=str(row["name"]),
            description=str(row["description"]),
            system_prompt=str(row["system_prompt"]),
            default_scope=self._loads(str(row["default_scope_json"])),
            default_model=str(row["default_model"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    def list_templates(self) -> list[AgentTemplateView]:
        rows = self._all("SELECT * FROM agent_templates ORDER BY name ASC", ())
        return [
            AgentTemplateView(
                name=str(row["name"]),
                description=str(row["description"]),
                system_prompt=str(row["system_prompt"]),
                default_scope=self._loads(str(row["default_scope_json"])),
                default_model=str(row["default_model"]),
                created_at=datetime.fromisoformat(str(row["created_at"])),
                updated_at=datetime.fromisoformat(str(row["updated_at"])),
            )
            for row in rows
        ]

    # agents
    def create_agent_run(self, req: AgentStartRequest, model: str, depth: int) -> AgentView:
        now = utcnow_iso()
        agent_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_runs(
                  id, session_id, parent_agent_id, depth, name, goal, status, model, template_name,
                  scope_json, expected_output_schema_json, max_steps, final_output, error, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, '', '', ?, ?)
                """,
                (
                    agent_id,
                    req.session_id,
                    req.parent_agent_id,
                    depth,
                    req.name,
                    req.goal,
                    model,
                    req.template_name,
                    self._dumps(req.scope),
                    self._dumps(req.expected_output_schema),
                    req.max_steps,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_agent(agent_id)

    def get_agent(self, agent_id: str) -> AgentView:
        row = self._one("SELECT * FROM agent_runs WHERE id = ?", (agent_id,))
        if row is None:
            raise KeyError(agent_id)
        return self._row_to_agent(row)

    def update_agent_state(
        self,
        agent_id: str,
        state: str,
        *,
        final_output: str | None = None,
        error: str | None = None,
    ) -> AgentView:
        now = utcnow_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE agent_runs SET status = ?, final_output = COALESCE(?, final_output), error = COALESCE(?, error), updated_at = ? WHERE id = ?",
                (state, final_output, error, now, agent_id),
            )
            self._conn.commit()
        return self.get_agent(agent_id)

    def append_agent_event(self, agent_id: str, event_type: str, payload: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_events(agent_id, event_type, payload_json, created_at) VALUES(?, ?, ?, ?)",
                (agent_id, event_type, self._dumps(payload), utcnow_iso()),
            )
            self._conn.commit()

    def list_agent_events(self, agent_id: str, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT event_type, payload_json, created_at FROM agent_events WHERE agent_id = ? ORDER BY id ASC LIMIT ?",
            (agent_id, max(1, limit)),
        )
        return [
            {
                "event_type": str(row["event_type"]),
                "payload": self._loads(str(row["payload_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    # tools
    def upsert_tool(
        self,
        *,
        session_id: str | None,
        name: str,
        language: str,
        description: str,
        script: str,
        args_schema: dict[str, Any],
        created_by_agent_id: str | None,
    ) -> dict[str, Any]:
        now = utcnow_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO agent_tools(
                  session_id, name, language, description, script, args_schema_json, created_by_agent_id, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, name) DO UPDATE SET
                  language = excluded.language,
                  description = excluded.description,
                  script = excluded.script,
                  args_schema_json = excluded.args_schema_json,
                  created_by_agent_id = excluded.created_by_agent_id,
                  updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    name,
                    language,
                    description,
                    script,
                    self._dumps(args_schema),
                    created_by_agent_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        tool = self.get_tool(session_id=session_id, name=name)
        if tool is None:
            raise KeyError(name)
        return tool

    def get_tool(self, *, session_id: str | None, name: str) -> dict[str, Any] | None:
        row = self._one(
            "SELECT * FROM agent_tools WHERE session_id IS ? AND name = ?",
            (session_id, name),
        )
        if row is None and session_id is not None:
            row = self._one(
                "SELECT * FROM agent_tools WHERE session_id IS NULL AND name = ?",
                (name,),
            )
        if row is None:
            return None
        return self._row_to_tool(row)

    def list_tools(self, *, session_id: str | None, limit: int = 120) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self._all(
                "SELECT * FROM agent_tools WHERE session_id IS NULL ORDER BY name ASC LIMIT ?",
                (max(1, limit),),
            )
            return [self._row_to_tool(row) for row in rows]
        rows = self._all(
            """
            SELECT * FROM agent_tools
            WHERE session_id = ? OR session_id IS NULL
            ORDER BY CASE WHEN session_id = ? THEN 0 ELSE 1 END ASC, name ASC
            LIMIT ?
            """,
            (session_id, session_id, max(1, limit)),
        )
        return [self._row_to_tool(row) for row in rows]

    def max_agent_depth(self, parent_agent_id: str | None) -> int:
        if not parent_agent_id:
            return 0
        return int(self.get_agent(parent_agent_id).depth)

    # helpers
    def _row_to_agent(self, row: sqlite3.Row) -> AgentView:
        return AgentView(
            id=str(row["id"]),
            session_id=row["session_id"],
            parent_agent_id=row["parent_agent_id"],
            depth=int(row["depth"]),
            name=str(row["name"]),
            goal=str(row["goal"]),
            status=str(row["status"]),
            model=str(row["model"]),
            template_name=row["template_name"],
            scope=self._loads(str(row["scope_json"])),
            expected_output_schema=self._loads(str(row["expected_output_schema_json"])),
            max_steps=int(row["max_steps"]),
            final_output=str(row["final_output"] or ""),
            error=str(row["error"] or ""),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    def _row_to_tool(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "session_id": row["session_id"],
            "name": str(row["name"]),
            "language": str(row["language"]),
            "description": str(row["description"]),
            "script": str(row["script"]),
            "args_schema": self._loads(str(row["args_schema_json"])),
            "created_by_agent_id": row["created_by_agent_id"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _all(self, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return list(cur.fetchall())

    def _one(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchone()

    @staticmethod
    def _dumps(value: Any) -> str:
        import json

        return json.dumps(value, ensure_ascii=True)

    @staticmethod
    def _loads(raw: str) -> Any:
        import json

        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return raw
