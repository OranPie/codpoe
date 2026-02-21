from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from poecoder.db import Database, utcnow_iso


@dataclass(slots=True)
class AuditService:
    db: Database

    def log(
        self,
        actor: str,
        tool_name: str,
        args: dict,
        policy_decision: str,
        result_status: str,
        duration_ms: int,
    ) -> None:
        serialized = json.dumps(args, sort_keys=True, ensure_ascii=True)
        args_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.db.execute(
            """
            INSERT INTO tool_audit(actor, tool_name, args_hash, policy_decision, result_status, duration_ms, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (actor, tool_name, args_hash, policy_decision, result_status, duration_ms, utcnow_iso()),
        )
