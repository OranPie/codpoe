from __future__ import annotations

import json
import re
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
            INSERT INTO sessions(
                id, title, mode, active_model, thinking_level, thinking_budget, show_think_details,
                allow_model_command_create, encourage_model_command_create,
                policy_profile, project_id, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "",
                req.mode,
                active_model,
                req.thinking_level,
                req.thinking_budget,
                1 if req.show_think_details else 0,
                1 if req.allow_model_command_create else 0,
                1 if req.encourage_model_command_create else 0,
                req.policy_profile,
                req.project_id,
                now,
                now,
            ),
        )
        return self.get(session_id)

    def list(self, project_id: str | None = None, limit: int = 20) -> list[SessionResponse]:
        query = "SELECT * FROM sessions"
        params: tuple[object, ...]
        if project_id:
            query += " WHERE project_id = ?"
            params = (project_id, limit)
            query += " ORDER BY updated_at DESC LIMIT ?"
        else:
            params = (limit,)
            query += " ORDER BY updated_at DESC LIMIT ?"
        rows = self.db.query_all(query, params)
        return [self._row_to_session(row) for row in rows]

    def get(self, session_id: str) -> SessionResponse:
        row = self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            raise KeyError(f"session not found: {session_id}")
        return self._row_to_session(row)

    def update_title(self, session_id: str, title: str) -> SessionResponse:
        clean = self._normalize_title(title)
        self.db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (clean, utcnow_iso(), session_id),
        )
        return self.get(session_id)

    def maybe_update_title_from_turn(self, session_id: str, user_prompt: str, model_output: str) -> SessionResponse:
        current = self.get(session_id)
        if current.title.strip():
            return current
        candidate = self._derive_title(user_prompt, model_output)
        if not candidate:
            return current
        return self.update_title(session_id, candidate)

    @staticmethod
    def _row_to_session(row: object) -> SessionResponse:
        return SessionResponse(
            id=row["id"],
            title=str(row["title"] or ""),
            mode=row["mode"],
            active_model=row["active_model"],
            thinking_level=row["thinking_level"],
            thinking_budget=int(row["thinking_budget"]),
            show_think_details=bool(int(row["show_think_details"])),
            allow_model_command_create=bool(int(row["allow_model_command_create"])),
            encourage_model_command_create=bool(int(row["encourage_model_command_create"])),
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

    @staticmethod
    def _normalize_title(value: str, max_len: int = 72) -> str:
        text = re.sub(r"\s+", " ", value).strip()
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3].rstrip() + "..."

    def _derive_title(self, user_prompt: str, model_output: str) -> str:
        lines = [line.strip() for line in (model_output or "").splitlines() if line.strip()]
        for raw in lines:
            text = raw
            lowered = text.lower()
            if lowered.startswith("@tool "):
                continue
            if lowered.startswith("thinking...") or lowered.startswith("generating..."):
                continue
            text = re.sub(r"^#+\s*", "", text)
            text = text.strip("-*` \t")
            if len(text) >= 4:
                return self._normalize_title(text)
        return self._normalize_title(user_prompt)


    def change_model(self, session_id: str, model: str) -> SessionResponse:
        self.db.execute(
            "UPDATE sessions SET active_model = ?, updated_at = ? WHERE id = ?",
            (model, utcnow_iso(), session_id),
        )
        return self.get(session_id)

    def update_thinking(self, session_id: str, level: str, budget: int) -> SessionResponse:
        self.db.execute(
            "UPDATE sessions SET thinking_level = ?, thinking_budget = ?, updated_at = ? WHERE id = ?",
            (level, budget, utcnow_iso(), session_id),
        )
        return self.get(session_id)

    def update_think_details(self, session_id: str, show_think_details: bool) -> SessionResponse:
        self.db.execute(
            "UPDATE sessions SET show_think_details = ?, updated_at = ? WHERE id = ?",
            (1 if show_think_details else 0, utcnow_iso(), session_id),
        )
        return self.get(session_id)

    def update_command_policy(self, session_id: str, allow_create: bool, encourage_create: bool) -> SessionResponse:
        self.db.execute(
            """
            UPDATE sessions
            SET allow_model_command_create = ?, encourage_model_command_create = ?, updated_at = ?
            WHERE id = ?
            """,
            (1 if allow_create else 0, 1 if encourage_create else 0, utcnow_iso(), session_id),
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

    def select_context_for_prompt(
        self,
        session_id: str,
        prompt: str,
        keys: list[str] | None = None,
        max_items: int = 20,
        max_value_chars: int = 10000,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if keys:
            selected = self.get_context(session_id, keys)
            compacted = {
                key: self._compact_value(value, max_chars=1200)
                for key, value in selected.items()
            }
            total = sum(len(json.dumps(val, ensure_ascii=False)) for val in compacted.values())
            return compacted, {
                "source": "explicit_keys",
                "requested_keys": len(keys),
                "selected_items": len(compacted),
                "dropped_items": max(0, len(keys) - len(compacted)),
                "selected_chars": total,
            }

        rows = self.db.query_all(
            """
            SELECT key, value_json, scope, created_at
            FROM context_entries
            WHERE session_id = ?
            ORDER BY created_at DESC
            """,
            (session_id,),
        )
        prompt_tokens = self._tokens(prompt)
        ranked: list[tuple[float, str, Any]] = []
        for idx, row in enumerate(rows):
            key = row["key"]
            value = loads(row["value_json"])
            scope = str(row["scope"])
            score = self._score_entry(prompt_tokens, key, value, scope, idx)
            ranked.append((score, key, value))
        ranked.sort(key=lambda item: item[0], reverse=True)

        selected: dict[str, Any] = {}
        total_chars = 0
        for _, key, value in ranked[:max_items]:
            compacted = self._compact_value(value, max_chars=1200)
            serialized = json.dumps(compacted, ensure_ascii=False)
            if selected and total_chars + len(serialized) > max_value_chars:
                continue
            selected[key] = compacted
            total_chars += len(serialized)
        diagnostics = {
            "source": "auto_ranked",
            "candidate_items": len(rows),
            "selected_items": len(selected),
            "dropped_items": max(0, len(rows) - len(selected)),
            "selected_chars": total_chars,
        }
        return selected, diagnostics

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

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())}

    def _score_entry(self, prompt_tokens: set[str], key: str, value: Any, scope: str, recency_idx: int) -> float:
        score = 0.0
        key_tokens = self._tokens(key)
        overlap = len(prompt_tokens.intersection(key_tokens))
        score += overlap * 2.5

        try:
            preview = json.dumps(value, ensure_ascii=False)
        except TypeError:
            preview = str(value)
        value_tokens = self._tokens(preview[:1200])
        score += min(8, len(prompt_tokens.intersection(value_tokens)))

        if scope == "pinned":
            score += 4.0
        elif scope == "turn":
            score += 2.0
        score += max(0.0, 3.0 - (recency_idx / 6.0))
        return score

    def _compact_value(self, value: Any, max_chars: int = 1200, depth: int = 0) -> Any:
        if depth >= 5:
            return "<truncated-depth>"
        if isinstance(value, str):
            if len(value) <= max_chars:
                return value
            return value[: max_chars - 3] + "..."
        if isinstance(value, list):
            limit = 12 if depth <= 1 else 6
            items = [self._compact_value(item, max_chars=max_chars // 2, depth=depth + 1) for item in value[:limit]]
            if len(value) > limit:
                items.append(f"... ({len(value) - limit} more)")
            return items
        if isinstance(value, dict):
            limit = 20 if depth == 0 else 10
            out: dict[str, Any] = {}
            for idx, (k, v) in enumerate(value.items()):
                if idx >= limit:
                    out["..."] = f"({len(value) - limit} more keys)"
                    break
                out[str(k)] = self._compact_value(v, max_chars=max_chars // 2, depth=depth + 1)
            return out
        return value
