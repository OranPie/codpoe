from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from poecoder.db import Database, utcnow_iso
from poecoder.models import SessionCreateRequest, SessionResponse
from poecoder.services.utils import dumps, loads, parse_dt


@dataclass(slots=True)
class SessionService:
    db: Database
    default_model: str

    def create(self, req: SessionCreateRequest) -> SessionResponse:
        session_id = str(uuid.uuid4())
        now = utcnow_iso()
        active_model = req.active_model or "auto"
        self.db.execute(
            """
            INSERT INTO sessions(id, mode, active_model, policy_profile, project_id, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, req.mode, active_model, req.policy_profile, req.project_id, now, now),
        )
        return self.get(session_id)

    def get(self, session_id: str) -> SessionResponse:
        row = self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            raise KeyError(f"session not found: {session_id}")
        return SessionResponse(
            id=row["id"],
            mode=row["mode"],
            active_model=row["active_model"],
            policy_profile=row["policy_profile"],
            project_id=row["project_id"],
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
        )

    def touch(self, session_id: str) -> None:
        self.db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (utcnow_iso(), session_id),
        )


    def change_model(self, session_id: str, model: str) -> SessionResponse:
        self.db.execute(
            "UPDATE sessions SET active_model = ?, updated_at = ? WHERE id = ?",
            (model, utcnow_iso(), session_id),
        )
        return self.get(session_id)

    def put_context(self, session_id: str, key: str, value: Any, scope: str = "turn", ttl_seconds: int | None = None) -> None:
        now = utcnow_iso()
        self.db.execute(
            """
            INSERT INTO context_entries(session_id, key, value_json, scope, ttl_seconds, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, key) DO UPDATE SET
                value_json = excluded.value_json,
                scope = excluded.scope,
                ttl_seconds = excluded.ttl_seconds,
                created_at = excluded.created_at
            """,
            (session_id, key, dumps(value), scope, ttl_seconds, now),
        )

    def get_context(self, session_id: str, keys: list[str] | None = None) -> dict[str, Any]:
        if keys:
            placeholders = ",".join("?" for _ in keys)
            rows = self.db.query_all(
                f"SELECT key, value_json FROM context_entries WHERE session_id = ? AND key IN ({placeholders})",
                (session_id, *keys),
            )
        else:
            rows = self.db.query_all(
                "SELECT key, value_json FROM context_entries WHERE session_id = ?",
                (session_id,),
            )
        return {row["key"]: loads(row["value_json"]) for row in rows}

    def reset_for_turn(self, session: SessionResponse) -> None:
        if session.mode == "coding":
            self.db.execute(
                "DELETE FROM context_entries WHERE session_id = ? AND scope = ?",
                (session.id, "turn"),
            )
            return

        rows = self.db.query_all(
            "SELECT id FROM context_entries WHERE session_id = ? ORDER BY created_at DESC",
            (session.id,),
        )
        keep = 12
        if len(rows) <= keep:
            return
        ids_to_remove = [row["id"] for row in rows[keep:]]
        placeholders = ",".join("?" for _ in ids_to_remove)
        self.db.execute(
            f"DELETE FROM context_entries WHERE id IN ({placeholders})",
            tuple(ids_to_remove),
        )
