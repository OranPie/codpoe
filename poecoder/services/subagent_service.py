from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from poecoder.db import Database, utcnow_iso
from poecoder.prompts import compose_subagent_system_message
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.session_service import SessionService
from poecoder.services.utils import dumps, loads, parse_dt


@dataclass(slots=True)
class SubagentService:
    db: Database
    model_client: PoeModelClient
    session_service: SessionService
    model_catalog: ModelCatalog
    _tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    _system_messages: dict[str, str] = field(default_factory=dict)

    def start(
        self,
        parent_session_id: str,
        model: str,
        perm: str,
        prompt: str,
        context_share: list[str],
        images: list[str] | None = None,
        system_message_modifier: str | None = None,
    ) -> dict[str, Any]:
        self.model_catalog.ensure_supported(model)
        agent_id = str(uuid.uuid4())
        now = utcnow_iso()
        image_list = list(images or [])
        self.db.execute(
            """
            INSERT INTO subagents(id, parent_session_id, model, perm, prompt, images_json, shared_context_json, state, result, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (agent_id, parent_session_id, model, perm, prompt, dumps(image_list), dumps(context_share), "running", now, now),
        )
        self._system_messages[agent_id] = compose_subagent_system_message(perm, system_message_modifier)
        task = asyncio.create_task(self._run(agent_id))
        self._tasks[agent_id] = task
        return self.read(agent_id)

    async def wait(self, agent_id: str, timeout_s: int = 60) -> dict[str, Any]:
        task = self._tasks.get(agent_id)
        if task is None:
            return self.read(agent_id)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except asyncio.TimeoutError:
            pass
        return self.read(agent_id)

    def cancel(self, agent_id: str) -> dict[str, Any]:
        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
        self._set_state(agent_id, "cancelled", "cancelled by parent")
        return self.read(agent_id)

    def read(self, agent_id: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM subagents WHERE id = ?", (agent_id,))
        if row is None:
            raise KeyError(agent_id)
        return {
            "id": row["id"],
            "parent_session_id": row["parent_session_id"],
            "model": row["model"],
            "perm": row["perm"],
            "prompt": row["prompt"],
            "images": loads(row["images_json"]) if row["images_json"] else [],
            "state": row["state"],
            "result": row["result"],
            "shared_context": loads(row["shared_context_json"]),
            "created_at": parse_dt(row["created_at"]),
            "updated_at": parse_dt(row["updated_at"]),
        }

    async def _run(self, agent_id: str) -> None:
        row = self.db.query_one("SELECT * FROM subagents WHERE id = ?", (agent_id,))
        if row is None:
            return
        try:
            parent_session = self.session_service.get(row["parent_session_id"])
            shared_keys = loads(row["shared_context_json"])
            context = self.session_service.get_context(parent_session.id, shared_keys)
            system_message = self._system_messages.get(
                agent_id,
                compose_subagent_system_message(str(row["perm"]), None),
            )
            reply = await self.model_client.chat(
                model=row["model"],
                system_message=system_message,
                user_prompt=row["prompt"],
                context=context,
                images=loads(row["images_json"]) if row["images_json"] else [],
            )
            self._set_state(agent_id, "completed", reply.text)
        except asyncio.CancelledError:
            self._set_state(agent_id, "cancelled", "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_state(agent_id, "failed", f"subagent failed: {exc}")
        finally:
            self._tasks.pop(agent_id, None)
            self._system_messages.pop(agent_id, None)

    def _set_state(self, agent_id: str, state: str, result: str | None) -> None:
        self.db.execute(
            "UPDATE subagents SET state = ?, result = ?, updated_at = ? WHERE id = ?",
            (state, result, utcnow_iso(), agent_id),
        )
