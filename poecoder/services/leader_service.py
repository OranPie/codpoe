from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from poecoder.db import Database, utcnow_iso
from poecoder.models import LeaderJobView, LeaderRunRequest, LeaderRunView, TaskView, TurnRequest
from poecoder.prompts import LEADER_SYSTEM_MESSAGE
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.session_service import SessionService
from poecoder.services.shell_service import ShellService
from poecoder.services.task_service import TaskService
from poecoder.services.utils import dumps, loads, parse_dt

TERMINAL_STATES = {"completed", "failed", "cancelled"}


@dataclass(slots=True)
class LeaderService:
    db: Database
    sessions: SessionService
    tasks: TaskService
    shell: ShellService
    model_client: PoeModelClient
    model_catalog: ModelCatalog
    _runtime_tasks: dict[str, concurrent.futures.Future[Any]] = field(default_factory=dict)
    _bg_loop: asyncio.AbstractEventLoop | None = None
    _bg_thread: threading.Thread | None = None
    _bg_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, req: LeaderRunRequest) -> LeaderRunView:
        session = self.sessions.get(req.session_id)
        planner_model = self._resolve_model(req.planner_model or session.active_model)
        worker_model = self._resolve_model(req.worker_model or planner_model)

        run_id = str(uuid.uuid4())
        now = utcnow_iso()
        self.db.execute(
            """
            INSERT INTO leader_runs(
                id, session_id, goal, planner_model, worker_model, state,
                plan_json, verify_command, result_json, error, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                run_id,
                req.session_id,
                req.goal,
                planner_model,
                worker_model,
                "queued",
                dumps({"jobs": []}),
                req.verify_command,
                now,
                now,
            ),
        )

        loop = self._ensure_background_loop()
        future = asyncio.run_coroutine_threadsafe(self._run(run_id, req, planner_model, worker_model), loop)
        self._runtime_tasks[run_id] = future
        future.add_done_callback(lambda _: self._runtime_tasks.pop(run_id, None))
        return self.read(run_id)

    async def wait(self, run_id: str, timeout_s: int = 120) -> LeaderRunView:
        future = self._runtime_tasks.get(run_id)
        if future is None:
            return self.read(run_id)
        try:
            await asyncio.to_thread(future.result, timeout_s)
        except concurrent.futures.TimeoutError:
            pass
        return self.read(run_id)

    def cancel(self, run_id: str) -> LeaderRunView:
        future = self._runtime_tasks.get(run_id)
        if future and not future.done():
            future.cancel()

        rows = self.db.query_all("SELECT task_id FROM leader_jobs WHERE run_id = ?", (run_id,))
        for row in rows:
            task_id = row["task_id"]
            if not task_id:
                continue
            try:
                self.tasks.cancel(str(task_id))
            except KeyError:
                continue

        self.db.execute(
            """
            UPDATE leader_jobs
            SET state = ?, error = ?, updated_at = ?
            WHERE run_id = ? AND state IN ('queued', 'planning', 'running')
            """,
            ("cancelled", "cancelled by user", utcnow_iso(), run_id),
        )
        self._set_run_state(run_id, "cancelled", result=None, error="cancelled by user", force=True)
        return self.read(run_id)

    def read(self, run_id: str) -> LeaderRunView:
        row = self.db.query_one("SELECT * FROM leader_runs WHERE id = ?", (run_id,))
        if row is None:
            raise KeyError(run_id)
        return LeaderRunView(
            id=row["id"],
            session_id=row["session_id"],
            goal=row["goal"],
            planner_model=row["planner_model"],
            worker_model=row["worker_model"],
            state=row["state"],
            plan=loads(row["plan_json"]) if row["plan_json"] else {},
            verify_command=row["verify_command"],
            result=loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
        )

    def list_jobs(self, run_id: str) -> list[LeaderJobView]:
        self.read(run_id)
        rows = self.db.query_all(
            "SELECT * FROM leader_jobs WHERE run_id = ? ORDER BY job_index ASC",
            (run_id,),
        )
        return [self._job_row_to_view(row) for row in rows]

    async def _run(self, run_id: str, req: LeaderRunRequest, planner_model: str, worker_model: str) -> None:
        try:
            self._set_run_state(run_id, "planning", result=None, error=None)
            plan = await self._build_plan(req, planner_model)
            self._set_plan(run_id, plan)
            self._set_run_state(run_id, "running", result=None, error=None)

            job_ids: list[str] = []
            for idx, job in enumerate(plan.get("jobs", []), start=1):
                job_id = str(uuid.uuid4())
                job_ids.append(job_id)
                self._create_job(
                    job_id=job_id,
                    run_id=run_id,
                    job_index=idx,
                    name=str(job.get("name", f"job-{idx}")),
                    objective=str(job.get("objective", "")),
                    scope=str(job.get("scope", "")),
                    owned_paths=[str(path) for path in job.get("owned_paths", [])],
                    context_keys=[str(key) for key in job.get("context_keys", [])],
                )

            semaphore = asyncio.Semaphore(req.max_parallel)
            workers = [
                asyncio.create_task(self._run_job(run_id, job_id, req, worker_model, semaphore))
                for job_id in job_ids
            ]
            if workers:
                await asyncio.gather(*workers)

            jobs = self.list_jobs(run_id)
            incomplete_jobs = [job for job in jobs if job.state != "completed"]
            verify_result: dict[str, Any] | None = None
            if req.verify_command and not incomplete_jobs:
                verify_result = (
                    await self.shell.run(
                        session_id=req.session_id,
                        command=req.verify_command,
                        danger_level=req.verify_danger_level,
                        cwd=req.verify_cwd,
                        timeout_s=req.verify_timeout_s,
                    )
                ).model_dump(mode="json")

            final_state = "completed"
            final_error: str | None = None
            if incomplete_jobs:
                states = {job.state for job in incomplete_jobs}
                if "failed" in states:
                    final_state = "failed"
                    final_error = f"{len([j for j in incomplete_jobs if j.state == 'failed'])} leader jobs failed"
                elif "cancelled" in states:
                    final_state = "cancelled"
                    final_error = "cancelled"
                else:
                    final_state = "failed"
                    final_error = f"{len(incomplete_jobs)} leader jobs incomplete"
            if verify_result and (
                not verify_result.get("allowed", False)
                or int(verify_result.get("exit_code", 1)) != 0
            ):
                final_state = "failed"
                final_error = "verify command failed"

            self._set_run_state(
                run_id,
                final_state,
                result={"jobs": [job.model_dump(mode="json") for job in jobs], "verify": verify_result},
                error=final_error,
            )
        except asyncio.CancelledError:
            self._set_run_state(run_id, "cancelled", result=None, error="cancelled", force=True)
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_run_state(run_id, "failed", result=None, error=f"leader run failed: {exc}")

    async def _run_job(
        self,
        run_id: str,
        job_id: str,
        req: LeaderRunRequest,
        worker_model: str,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            row = self.db.query_one("SELECT * FROM leader_jobs WHERE id = ?", (job_id,))
            if row is None or row["state"] in TERMINAL_STATES:
                return

            scope = str(row["scope"])
            owned_paths = loads(row["owned_paths_json"]) if row["owned_paths_json"] else []
            context_keys = loads(row["context_keys_json"]) if row["context_keys_json"] else []
            system_message = self._compose_worker_system(scope=scope, owned_paths=owned_paths)
            user_prompt = self._compose_job_prompt(
                goal=req.goal,
                name=str(row["name"]),
                objective=str(row["objective"]),
                scope=scope,
                owned_paths=owned_paths,
            )

            self._set_job_state(job_id, "running", result=None, error=None)
            task_id: str | None = None
            try:
                task = await self.tasks.start_turn(
                    TurnRequest(
                        session_id=req.session_id,
                        user_prompt=user_prompt,
                        system_message=system_message,
                        context_keys=list(dict.fromkeys(req.context_keys + context_keys)),
                        metadata={
                            "leader_run_id": run_id,
                            "leader_job_id": job_id,
                            "scope": scope,
                            "owned_paths": owned_paths,
                            "isolation_rule": "do-not-touch-outside-owned-paths",
                            "worker_model": worker_model,
                        },
                    )
                )
                task_id = task.id
                self._set_job_task(job_id, task_id)
                snapshot = await self._wait_task(task_id, req.per_job_timeout_s)
                self._set_job_state(job_id, snapshot.state, result=snapshot.result, error=snapshot.error)
            except asyncio.CancelledError:
                if task_id:
                    try:
                        self.tasks.cancel(task_id)
                    except KeyError:
                        pass
                self._set_job_state(job_id, "cancelled", result=None, error="cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                if task_id:
                    try:
                        self.tasks.cancel(task_id)
                    except KeyError:
                        pass
                self._set_job_state(job_id, "failed", result=None, error=f"leader job failed: {exc}")

    async def _wait_task(self, task_id: str, timeout_s: int) -> TaskView:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            view = self.tasks.get(task_id)
            if view.state in TERMINAL_STATES:
                return view
            if loop.time() >= deadline:
                try:
                    self.tasks.cancel(task_id)
                except KeyError:
                    pass
                return self.tasks.get(task_id)
            await asyncio.sleep(0.2)

    async def _build_plan(self, req: LeaderRunRequest, planner_model: str) -> dict[str, Any]:
        if req.jobs:
            return {
                "source": "user",
                "planner_model": planner_model,
                "jobs": self._normalize_jobs([job.model_dump(mode="json") for job in req.jobs], req.goal),
            }

        context = self.sessions.get_context(req.session_id, req.context_keys) if req.context_keys else {}
        prompt = (
            "Design a parallel implementation plan for this coding goal. Return strict JSON only.\n\n"
            f"Goal:\n{req.goal}\n\n"
            "Required JSON shape:\n"
            "{\n"
            '  "interfaces": ["global contract bullets"],\n'
            '  "jobs": [\n'
            "    {\n"
            '      "name": "short job name",\n'
            '      "objective": "what to implement",\n'
            '      "scope": "owned sub-scope with non-interference rule",\n'
            '      "owned_paths": ["path/prefix/or/module"],\n'
            '      "context_keys": ["optional context key"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Constraints:\n"
            "- Clarify each sub-scope.\n"
            "- No overlapping ownership between jobs.\n"
            "- Each job must avoid influencing other scopes.\n"
            "- Keep interfaces compact and actionable."
        )
        reply = await self.model_client.chat(
            model=planner_model,
            system_message=LEADER_SYSTEM_MESSAGE,
            user_prompt=prompt,
            context={"context": context},
        )
        parsed = self._extract_json_object(reply.text)
        interfaces = parsed.get("interfaces")
        if not isinstance(interfaces, list):
            interfaces = []
        return {
            "source": "model",
            "planner_model": planner_model,
            "interfaces": [str(item) for item in interfaces][:32],
            "jobs": self._normalize_jobs(parsed.get("jobs", []), req.goal),
            "raw_plan_preview": reply.text[:2000],
        }

    def _normalize_jobs(self, jobs: list[dict[str, Any]], goal: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        owned_paths: set[str] = set()
        for idx, item in enumerate(jobs, start=1):
            if not isinstance(item, dict):
                item = {}
            name = str(item.get("name") or f"job-{idx}").strip() or f"job-{idx}"
            objective = str(item.get("objective") or goal).strip() or goal
            scope = str(item.get("scope") or f"scope-{idx}").strip()

            raw_paths = item.get("owned_paths")
            paths = [str(path).strip() for path in raw_paths] if isinstance(raw_paths, list) else []
            paths = [path for path in paths if path]
            if not paths:
                paths = [f"scope_{idx}"]

            deduped: list[str] = []
            overlaps: list[str] = []
            for path in paths:
                if path in owned_paths:
                    overlaps.append(path)
                    continue
                owned_paths.add(path)
                deduped.append(path)
            if not deduped:
                fallback = f"scope_{idx}_isolated"
                deduped = [fallback]
                owned_paths.add(fallback)
            if overlaps:
                scope = scope + f" (overlap removed: {', '.join(overlaps)})"

            raw_keys = item.get("context_keys")
            context_keys = [str(key).strip() for key in raw_keys] if isinstance(raw_keys, list) else []
            context_keys = [key for key in context_keys if key]

            normalized.append(
                {
                    "name": name,
                    "objective": objective,
                    "scope": scope,
                    "owned_paths": deduped,
                    "context_keys": context_keys,
                }
            )

        if normalized:
            return normalized
        return [
            {
                "name": "single-job",
                "objective": goal,
                "scope": "single-owner scope for full goal delivery",
                "owned_paths": ["workspace"],
                "context_keys": [],
            }
        ]

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if not candidate:
            return {}
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _resolve_model(self, preferred: str | None) -> str:
        candidate = (preferred or "").strip()
        if not candidate or candidate == "auto":
            models = self.model_catalog.list_models(refresh=False)
            if not models:
                models = self.model_catalog.list_models(refresh=True)
            if not models:
                raise ValueError("no models available")
            candidate = models[0]
        self.model_catalog.ensure_supported(candidate)
        return candidate

    @staticmethod
    def _compose_worker_system(scope: str, owned_paths: list[str]) -> str:
        owned = ", ".join(owned_paths) if owned_paths else "(none)"
        return (
            LEADER_SYSTEM_MESSAGE
            + "\n\nExecution role: scoped worker."
            + f"\nOwned scope: {scope}"
            + f"\nOwned paths/modules: {owned}"
            + "\nRules:"
            + "\n- Edit only owned scope and owned paths."
            + "\n- Do not modify or refactor outside your scope."
            + "\n- If blocked by cross-scope dependencies, report interface needs instead of editing outside scope."
        )

    @staticmethod
    def _compose_job_prompt(goal: str, name: str, objective: str, scope: str, owned_paths: list[str]) -> str:
        return (
            f"Global goal:\n{goal}\n\n"
            f"Assigned job: {name}\n"
            f"Objective: {objective}\n"
            f"Scope boundary: {scope}\n"
            f"Owned paths/modules: {', '.join(owned_paths)}\n\n"
            "Deliverables:\n"
            "1) Implement only inside owned scope.\n"
            "2) Keep interfaces explicit for other jobs.\n"
            "3) Return files touched, checks run, and unresolved dependencies."
        )

    def _create_job(
        self,
        job_id: str,
        run_id: str,
        job_index: int,
        name: str,
        objective: str,
        scope: str,
        owned_paths: list[str],
        context_keys: list[str],
    ) -> None:
        now = utcnow_iso()
        self.db.execute(
            """
            INSERT INTO leader_jobs(
                id, run_id, job_index, name, objective, scope,
                owned_paths_json, context_keys_json, task_id, state, result_json,
                error, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?)
            """,
            (
                job_id,
                run_id,
                job_index,
                name,
                objective,
                scope,
                dumps(owned_paths),
                dumps(context_keys),
                "queued",
                now,
                now,
            ),
        )

    def _set_job_task(self, job_id: str, task_id: str) -> None:
        self.db.execute(
            "UPDATE leader_jobs SET task_id = ?, updated_at = ? WHERE id = ?",
            (task_id, utcnow_iso(), job_id),
        )

    def _set_job_state(self, job_id: str, state: str, result: dict[str, Any] | None, error: str | None) -> None:
        self.db.execute(
            """
            UPDATE leader_jobs
            SET state = ?, result_json = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (state, None if result is None else dumps(result), error, utcnow_iso(), job_id),
        )

    def _set_plan(self, run_id: str, plan: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE leader_runs SET plan_json = ?, updated_at = ? WHERE id = ?",
            (dumps(plan), utcnow_iso(), run_id),
        )

    def _set_run_state(
        self,
        run_id: str,
        state: str,
        result: dict[str, Any] | None,
        error: str | None,
        force: bool = False,
    ) -> None:
        if not force and state != "cancelled":
            row = self.db.query_one("SELECT state FROM leader_runs WHERE id = ?", (run_id,))
            if row is not None and row["state"] == "cancelled":
                return
        self.db.execute(
            """
            UPDATE leader_runs
            SET state = ?, result_json = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (state, None if result is None else dumps(result), error, utcnow_iso(), run_id),
        )

    def _ensure_background_loop(self) -> asyncio.AbstractEventLoop:
        with self._bg_lock:
            if self._bg_loop and self._bg_loop.is_running():
                return self._bg_loop
            loop = asyncio.new_event_loop()

            def run_loop() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(target=run_loop, name="poecoder-leader-loop", daemon=True)
            thread.start()
            self._bg_loop = loop
            self._bg_thread = thread
            return loop

    @staticmethod
    def _job_row_to_view(row: Any) -> LeaderJobView:
        return LeaderJobView(
            id=row["id"],
            run_id=row["run_id"],
            job_index=int(row["job_index"]),
            name=row["name"],
            objective=row["objective"],
            scope=row["scope"],
            owned_paths=loads(row["owned_paths_json"]) if row["owned_paths_json"] else [],
            context_keys=loads(row["context_keys_json"]) if row["context_keys_json"] else [],
            task_id=row["task_id"],
            state=row["state"],
            result=loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
        )
