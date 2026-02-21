from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL,
    active_model TEXT NOT NULL,
    thinking_level TEXT NOT NULL DEFAULT 'balanced',
    thinking_budget INTEGER NOT NULL DEFAULT 12000,
    show_think_details INTEGER NOT NULL DEFAULT 0,
    allow_model_command_create INTEGER NOT NULL DEFAULT 1,
    encourage_model_command_create INTEGER NOT NULL DEFAULT 1,
    policy_profile TEXT NOT NULL,
    project_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    scope TEXT NOT NULL,
    ttl_seconds INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, key)
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    session_id TEXT,
    project_id TEXT,
    tags_json TEXT NOT NULL,
    priority INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wiki_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    compacted_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS command_defs (
    name TEXT PRIMARY KEY,
    definition TEXT NOT NULL,
    runtime TEXT NOT NULL,
    args_schema_json TEXT NOT NULL,
    effect_schema_json TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    source TEXT NOT NULL,
    signature TEXT,
    signature_status TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS subagents (
    id TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL,
    model TEXT NOT NULL,
    perm TEXT NOT NULL,
    prompt TEXT NOT NULL,
    images_json TEXT NOT NULL DEFAULT '[]',
    shared_context_json TEXT NOT NULL,
    state TEXT NOT NULL,
    result TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    policy_decision TEXT NOT NULL,
    result_status TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tmp_writes (
    name TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS background_tasks (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_background_tasks_state_time ON background_tasks(state, updated_at);

CREATE TABLE IF NOT EXISTS leader_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    goal TEXT NOT NULL,
    planner_model TEXT NOT NULL,
    worker_model TEXT NOT NULL,
    state TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    verify_command TEXT,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_leader_runs_state_time ON leader_runs(state, updated_at);

CREATE TABLE IF NOT EXISTS leader_jobs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    job_index INTEGER NOT NULL,
    name TEXT NOT NULL,
    objective TEXT NOT NULL,
    scope TEXT NOT NULL,
    owned_paths_json TEXT NOT NULL,
    context_keys_json TEXT NOT NULL,
    task_id TEXT,
    state TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_leader_jobs_run_idx ON leader_jobs(run_id, job_index);
CREATE INDEX IF NOT EXISTS idx_leader_jobs_state_time ON leader_jobs(state, updated_at);

CREATE TABLE IF NOT EXISTS model_profiles (
    model TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    best_for TEXT NOT NULL,
    speed_tier INTEGER NOT NULL,
    quality_tier INTEGER NOT NULL,
    cost_tier INTEGER NOT NULL,
    max_context_hint INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_profiles_speed_quality ON model_profiles(speed_tier, quality_tier);

CREATE INDEX IF NOT EXISTS idx_memory_scope_project ON memory_entries(scope, project_id);
CREATE INDEX IF NOT EXISTS idx_wiki_project_topic ON wiki_docs(project_id, topic);
CREATE INDEX IF NOT EXISTS idx_tool_audit_name_time ON tool_audit(tool_name, created_at);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.executescript(SCHEMA_SQL)
        self._run_migrations()
        self._conn.commit()

    def _run_migrations(self) -> None:
        self._ensure_column(
            table="sessions",
            column="thinking_level",
            definition="TEXT NOT NULL DEFAULT 'balanced'",
        )
        self._ensure_column(
            table="sessions",
            column="title",
            definition="TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column(
            table="sessions",
            column="thinking_budget",
            definition="INTEGER NOT NULL DEFAULT 12000",
        )
        self._ensure_column(
            table="sessions",
            column="show_think_details",
            definition="INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            table="sessions",
            column="allow_model_command_create",
            definition="INTEGER NOT NULL DEFAULT 1",
        )
        self._ensure_column(
            table="sessions",
            column="encourage_model_command_create",
            definition="INTEGER NOT NULL DEFAULT 1",
        )
        self._ensure_column(
            table="subagents",
            column="images_json",
            definition="TEXT NOT NULL DEFAULT '[]'",
        )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in rows}
        if column in existing:
            return
        self._conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._lock.acquire()
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
        return cur

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return list(cur.fetchall())

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchone()



def utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
