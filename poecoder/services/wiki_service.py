from __future__ import annotations

from dataclasses import dataclass

from poecoder.db import Database, utcnow_iso
from poecoder.services.utils import dumps, loads


@dataclass(slots=True)
class WikiService:
    db: Database

    def ingest(self, project_id: str, topic: str, content: str, source: str) -> int:
        now = utcnow_iso()
        cur = self.db.execute(
            """
            INSERT INTO wiki_docs(project_id, topic, content, meta_json, compacted_at, created_at, updated_at)
            VALUES(?, ?, ?, ?, NULL, ?, ?)
            """,
            (project_id, topic, content, dumps({"source": source}), now, now),
        )
        return int(cur.lastrowid)

    def query(self, project_id: str, query: str, limit: int = 10) -> list[dict[str, object]]:
        pattern = f"%{query}%"
        rows = self.db.query_all(
            """
            SELECT * FROM wiki_docs
            WHERE project_id = ? AND (topic LIKE ? OR content LIKE ?)
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (project_id, pattern, pattern, limit),
        )
        results: list[dict[str, object]] = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "topic": row["topic"],
                    "content": row["content"],
                    "meta": loads(row["meta_json"]),
                    "compacted_at": row["compacted_at"],
                }
            )
        return results

    def compact(self, project_id: str) -> dict[str, int]:
        rows = self.db.query_all(
            "SELECT id, topic, content FROM wiki_docs WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        )
        seen: set[tuple[str, str]] = set()
        removed = 0
        compacted = 0
        for row in rows:
            topic = row["topic"].strip().lower()
            content = " ".join(row["content"].split())
            key = (topic, content)
            if key in seen:
                self.db.execute("DELETE FROM wiki_docs WHERE id = ?", (row["id"],))
                removed += 1
                continue
            seen.add(key)

            summary = self._compact_text(content)
            if summary != row["content"]:
                self.db.execute(
                    "UPDATE wiki_docs SET content = ?, compacted_at = ?, updated_at = ? WHERE id = ?",
                    (summary, utcnow_iso(), utcnow_iso(), row["id"]),
                )
                compacted += 1
        return {"removed": removed, "compacted": compacted}

    @staticmethod
    def _compact_text(text: str, max_len: int = 320) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= max_len:
            return normalized
        sentence_end = normalized.find(". ")
        if 0 < sentence_end < max_len:
            return normalized[: sentence_end + 1]
        return normalized[:max_len].rstrip() + "..."
