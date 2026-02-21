from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from poecoder.db import Database, utcnow_iso
from poecoder.models import SubagentStartRequest, TaskStartSubagentRequest, TaskView, TurnRequest
from poecoder.services.subagent_service import SubagentService
from poecoder.services.turn_service import TurnService
from poecoder.services.utils import dumps, loads, parse_dt


@dataclass(slots=True)
class TaskService:
    db: Database
    turns: TurnService
    subagents: SubagentService
    _runtime_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    _subagent_ids: dict[str, str] = field(default_factory=dict)

    async def start_turn(self, req: TurnRequest) -> TaskView:
        task_id = self._create_task_record("turn", req.model_dump(mode="json"))
        task = asyncio.create_task(self._run_turn(task_id, req))
        self._runtime_tasks[task_id] = task
        return self.get(task_id)

    async def start_subagent(self, req: TaskStartSubagentRequest) -> TaskView:
        task_id = self._create_task_record("subagent", req.model_dump(mode="json"))
        task = asyncio.create_task(self._run_subagent(task_id, req))
        self._runtime_tasks[task_id] = task
        return self.get(task_id)

    def get(self, task_id: str) -> TaskView:
        row = self.db.query_one("SELECT * FROM background_tasks WHERE id = ?", (task_id,))
        if row is None:
            raise KeyError(task_id)
        return TaskView(
            id=row["id"],
            task_type=row["task_type"],
            state=row["state"],
            payload=loads(row["payload_json"]),
            result=None if row["result_json"] is None else loads(row["result_json"]),
            error=row["error"],
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
        )

    def list(self, limit: int = 50, state: str | None = None, task_type: str | None = None) -> list[TaskView]:
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        params.append(limit)
        rows = self.db.query_all(
            f"SELECT * FROM background_tasks WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC LIMIT ?",
            tuple(params),
        )
        return [
            TaskView(
                id=row["id"],
                task_type=row["task_type"],
                state=row["state"],
                payload=loads(row["payload_json"]),
                result=None if row["result_json"] is None else loads(row["result_json"]),
                error=row["error"],
                created_at=parse_dt(row["created_at"]),
                updated_at=parse_dt(row["updated_at"]),
            )
            for row in rows
        ]

    def cancel(self, task_id: str) -> TaskView:
        task = self._runtime_tasks.get(task_id)
        if task and not task.done():
            task.cancel()
        subagent_id = self._subagent_ids.get(task_id)
        if subagent_id:
            try:
                self.subagents.cancel(subagent_id)
            except KeyError:
                pass
        self._set_state(task_id, "cancelled", result=None, error="cancelled by user")
        return self.get(task_id)

    def read_output(self, task_id: str) -> dict[str, Any]:
        task = self.get(task_id)
        return {
            "task_id": task.id,
            "task_type": task.task_type,
            "state": task.state,
            "result": task.result,
            "error": task.error,
            "updated_at": task.updated_at,
        }

    def _create_task_record(self, task_type: str, payload: dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        now = utcnow_iso()
        self.db.execute(
            """
            INSERT INTO background_tasks(id, task_type, state, payload_json, result_json, error, created_at, updated_at)
            VALUES(?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (task_id, task_type, "queued", dumps(payload), now, now),
        )
        return task_id

    async def _run_turn(self, task_id: str, req: TurnRequest) -> None:
        self._set_state(task_id, "running", result=None, error=None)
        try:
            result = await self.turns.execute(req)
            self._set_state(task_id, "completed", result=result.model_dump(mode="json"), error=None)
        except asyncio.CancelledError:
            self._set_state(task_id, "cancelled", result=None, error="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_state(task_id, "failed", result=None, error=f"turn task failed: {exc}")
        finally:
            self._runtime_tasks.pop(task_id, None)

    async def _run_subagent(self, task_id: str, req: TaskStartSubagentRequest) -> None:
        self._set_state(task_id, "running", result=None, error=None)
        try:
            started = self.subagents.start(
                parent_session_id=req.parent_session_id,
                model=req.model,
                perm=req.perm,
                prompt=req.prompt,
                context_share=req.context_share,
                images=req.images,
                system_message_modifier=req.system_message_modifier,
            )
            subagent_id = str(started["id"])
            self._subagent_ids[task_id] = subagent_id
            snapshot = await self.subagents.wait(subagent_id, timeout_s=req.wait_timeout_s)
            self._set_state(
                task_id,
                "completed",
                result={"subagent_id": subagent_id, "subagent": snapshot},
                error=None,
            )
        except asyncio.CancelledError:
            self._set_state(task_id, "cancelled", result=None, error="cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_state(task_id, "failed", result=None, error=f"subagent task failed: {exc}")
        finally:
            self._runtime_tasks.pop(task_id, None)
            self._subagent_ids.pop(task_id, None)

    def _set_state(self, task_id: str, state: str, result: dict[str, Any] | None, error: str | None) -> None:
        self.db.execute(
            """
            UPDATE background_tasks
            SET state = ?, result_json = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (state, None if result is None else dumps(result), error, utcnow_iso(), task_id),
        )
