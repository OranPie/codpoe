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

    def query(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        topic: str | None = None,
        include_content: bool = True,
        include_meta: bool = True,
        max_content_chars: int | None = None,
    ) -> list[dict[str, object]]:
        pattern = f"%{query}%"
        clauses = ["project_id = ?", "(topic LIKE ? OR content LIKE ?)"]
        params: list[object] = [project_id, pattern, pattern]
        if topic:
            clauses.append("topic LIKE ?")
            params.append(f"%{topic}%")
        params.append(limit)
        rows = self.db.query_all(
            f"""
            SELECT * FROM wiki_docs
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        results: list[dict[str, object]] = []
        for row in rows:
            content = str(row["content"])
            if max_content_chars is not None and len(content) > max_content_chars:
                cap = max(int(max_content_chars), 1)
                if cap <= 3:
                    content = content[:cap]
                else:
                    content = content[: cap - 3] + "..."
            payload: dict[str, object] = {
                "id": row["id"],
                "project_id": row["project_id"],
                "topic": row["topic"],
                "compacted_at": row["compacted_at"],
            }
            if include_content:
                payload["content"] = content
            if include_meta:
                payload["meta"] = loads(row["meta_json"])
            results.append(
                payload
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
