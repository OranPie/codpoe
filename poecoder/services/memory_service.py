from __future__ import annotations

from dataclasses import dataclass

from poecoder.db import Database, utcnow_iso
from poecoder.models import MemoryEditRequest, MemoryEntryView, MemoryReadRequest, MemoryWriteRequest
from poecoder.services.utils import dumps, loads, parse_dt


@dataclass(slots=True)
class MemoryService:
    db: Database

    def write(self, req: MemoryWriteRequest) -> int:
        now = utcnow_iso()
        cur = self.db.execute(
            """
            INSERT INTO memory_entries(scope, session_id, project_id, tags_json, priority, content, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                req.scope,
                req.session_id,
                req.project_id,
                dumps(req.tags),
                req.priority,
                req.content,
                now,
                now,
            ),
        )
        return int(cur.lastrowid)

    def read(self, req: MemoryReadRequest) -> list[MemoryEntryView]:
        filters = []
        params: list[object] = []

        if req.scope:
            filters.append("scope = ?")
            params.append(req.scope)
        if req.session_id:
            filters.append("session_id = ?")
            params.append(req.session_id)
        if req.project_id:
            filters.append("project_id = ?")
            params.append(req.project_id)
        if req.query:
            filters.append("content LIKE ?")
            params.append(f"%{req.query}%")
        if req.min_priority is not None:
            filters.append("priority >= ?")
            params.append(req.min_priority)

        where = " AND ".join(filters) if filters else "1=1"
        fetch_limit = req.limit
        if req.tags_any:
            fetch_limit = max(req.limit * 6, req.limit)
        params.append(fetch_limit)
        rows = self.db.query_all(
            f"""
            SELECT * FROM memory_entries
            WHERE {where}
            ORDER BY priority DESC, updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        want_tags = {tag.strip() for tag in req.tags_any if isinstance(tag, str) and tag.strip()}
        out: list[MemoryEntryView] = []
        for row in rows:
            tags = loads(row["tags_json"])
            if want_tags and not any(str(tag) in want_tags for tag in tags):
                continue
            content = str(row["content"])
            if not req.include_content:
                content = ""
            elif req.max_content_chars is not None and len(content) > req.max_content_chars:
                cap = max(int(req.max_content_chars), 1)
                if cap <= 3:
                    content = content[:cap]
                else:
                    content = content[: cap - 3] + "..."
            out.append(
                MemoryEntryView(
                    id=row["id"],
                    scope=row["scope"],
                    session_id=row["session_id"],
                    project_id=row["project_id"],
                    tags=tags,
                    priority=row["priority"],
                    content=content,
                    created_at=parse_dt(row["created_at"]),
                    updated_at=parse_dt(row["updated_at"]),
                )
            )
            if len(out) >= req.limit:
                break
        return out

    def edit(self, req: MemoryEditRequest) -> int:
        targets = self._target_ids(req)
        if not targets:
            return 0

        changed = 0
        now = utcnow_iso()
        for entry_id in targets:
            row = self.db.query_one("SELECT content FROM memory_entries WHERE id = ?", (entry_id,))
            if row is None:
                continue
            content = row["content"]
            if req.operation == "delete":
                self.db.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
                changed += 1
                continue
            if req.operation == "replace":
                new_content = req.payload
            else:
                new_content = content + req.payload
            self.db.execute(
                "UPDATE memory_entries SET content = ?, updated_at = ? WHERE id = ?",
                (new_content, now, entry_id),
            )
            changed += 1
        return changed

    def _target_ids(self, req: MemoryEditRequest) -> list[int]:
        if req.entry_id is not None:
            return [req.entry_id]
        if not req.query:
            return []
        params: list[object] = [f"%{req.query}%"]
        sql = "SELECT id FROM memory_entries WHERE content LIKE ?"
        if req.scope:
            sql += " AND scope = ?"
            params.append(req.scope)
        rows = self.db.query_all(sql, tuple(params))
        return [int(row["id"]) for row in rows]
