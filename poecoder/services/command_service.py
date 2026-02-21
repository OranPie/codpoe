from __future__ import annotations

from dataclasses import dataclass

from poecoder.db import Database, utcnow_iso
from poecoder.policy import PolicyEngine
from poecoder.services.utils import dumps, loads


@dataclass(slots=True)
class CommandService:
    db: Database
    policy: PolicyEngine

    def install(
        self,
        name: str,
        definition: str,
        runtime: str,
        args_schema: dict,
        effect_schema: dict,
        capabilities: list[str],
        source: str,
        signature: str | None,
    ) -> dict[str, object]:
        cap_check = self.policy.check_capabilities(capabilities)
        if not cap_check.allowed:
            raise PermissionError(cap_check.reason)

        payload = f"{name}|{runtime}|{definition}|{args_schema}|{effect_schema}|{capabilities}"
        signature_status = self.policy.signature_status(payload, signature)
        now = utcnow_iso()
        row = self.db.query_one("SELECT version FROM command_defs WHERE name = ?", (name,))
        version = 1 if row is None else int(row["version"]) + 1

        self.db.execute(
            """
            INSERT INTO command_defs(
                name, definition, runtime, args_schema_json, effect_schema_json,
                capabilities_json, source, signature, signature_status, version, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                definition = excluded.definition,
                runtime = excluded.runtime,
                args_schema_json = excluded.args_schema_json,
                effect_schema_json = excluded.effect_schema_json,
                capabilities_json = excluded.capabilities_json,
                source = excluded.source,
                signature = excluded.signature,
                signature_status = excluded.signature_status,
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            (
                name,
                definition,
                runtime,
                dumps(args_schema),
                dumps(effect_schema),
                dumps(capabilities),
                source,
                signature,
                signature_status,
                version,
                now,
                now,
            ),
        )
        return self.get(name)

    def get(self, name: str) -> dict[str, object]:
        row = self.db.query_one("SELECT * FROM command_defs WHERE name = ?", (name,))
        if row is None:
            raise KeyError(name)
        return self._row_to_dict(row)

    def list(self) -> list[dict[str, object]]:
        rows = self.db.query_all("SELECT * FROM command_defs ORDER BY name ASC")
        return [self._row_to_dict(row) for row in rows]

    def patch(self, name: str, patch: dict[str, object]) -> dict[str, object]:
        current = self.get(name)
        merged = {
            "definition": patch.get("definition", current["definition"]),
            "args_schema": patch.get("args_schema", current["args_schema"]),
            "effect_schema": patch.get("effect_schema", current["effect_schema"]),
            "capabilities": patch.get("capabilities", current["capabilities"]),
            "signature": patch.get("signature", current["signature"]),
            "runtime": current["runtime"],
            "source": current["source"],
        }
        return self.install(
            name=name,
            definition=str(merged["definition"]),
            runtime=str(merged["runtime"]),
            args_schema=dict(merged["args_schema"]),
            effect_schema=dict(merged["effect_schema"]),
            capabilities=list(merged["capabilities"]),
            source=str(merged["source"]),
            signature=(None if merged["signature"] is None else str(merged["signature"])),
        )

    def delete(self, name: str) -> bool:
        cur = self.db.execute("DELETE FROM command_defs WHERE name = ?", (name,))
        return cur.rowcount > 0

    @staticmethod
    def _row_to_dict(row: object) -> dict[str, object]:
        return {
            "name": row["name"],
            "definition": row["definition"],
            "runtime": row["runtime"],
            "args_schema": loads(row["args_schema_json"]),
            "effect_schema": loads(row["effect_schema_json"]),
            "capabilities": loads(row["capabilities_json"]),
            "source": row["source"],
            "signature": row["signature"],
            "signature_status": row["signature_status"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
