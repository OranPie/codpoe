from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import shlex
import sys
from getpass import getpass
from base64 import b64encode
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from poecoder.config import get_settings
from poecoder.i18n import Translator, is_supported_lang, normalize_lang, supported_langs
from poecoder.prompts import PLAN_SYSTEM_MESSAGE
from poecoder.services.model_clients import PoeModelClient


@dataclass(slots=True)
class CliState:
    backend_url: str
    session_id: str | None = None
    mode: str = "coding"
    active_model: str | None = None
    thinking_level: str = "balanced"
    thinking_budget: int = 12000
    allow_model_command_create: bool = True
    encourage_model_command_create: bool = True
    system_message: str = ""
    project_id: str = "default"
    local_context: dict[str, Any] = field(default_factory=dict)
    pending_images: list[str] = field(default_factory=list)


class Style:
    def __init__(self) -> None:
        self.use_color = sys.stdout.isatty()

    def _wrap(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def title(self, text: str) -> str:
        return self._wrap(text, "1;36")

    def ok(self, text: str) -> str:
        return self._wrap(text, "32")

    def info(self, text: str) -> str:
        return self._wrap(text, "36")

    def warn(self, text: str) -> str:
        return self._wrap(text, "33")

    def dim(self, text: str) -> str:
        return self._wrap(text, "2")

    def error(self, text: str) -> str:
        return self._wrap(text, "31")


class PoeCoderCLI:
    def __init__(self, backend_url: str, direct: bool, model: str | None, lang: str | None) -> None:
        self.settings = get_settings()
        self.direct = direct
        self.model = model or self.settings.default_large_model
        self.state = CliState(backend_url=backend_url)
        self.state.thinking_level = self.settings.default_thinking_level
        self.state.thinking_budget = self.settings.default_thinking_budget
        self.http = httpx.Client(timeout=90.0)
        self.async_http = httpx.AsyncClient(timeout=90.0)
        self.direct_model_client = PoeModelClient(self.settings.poe_api_url, self.settings.poe_api_key)
        self.style = Style()
        self.i18n = Translator(lang=normalize_lang(lang or self.settings.lang))

    def _t(self, key: str, **kwargs: Any) -> str:
        return self.i18n.t(key, **kwargs)

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        clean = text.replace("\n", " ").strip()
        if len(clean) <= limit:
            return clean
        if limit <= 3:
            return clean[:limit]
        return clean[: limit - 3] + "..."

    @staticmethod
    def _json_text(data: Any, indent: int | None = None) -> str:
        return json.dumps(data, ensure_ascii=False, indent=indent)

    @staticmethod
    def _format_timestamp(value: str | None) -> str:
        if not value:
            return "-"
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value

    def _print_table(self, headers: list[str], rows: list[list[str]]) -> None:
        if not rows:
            print(self.style.dim(self._t("msg.table_empty")))
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        print(sep)
        print("| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
        print(sep)
        for row in rows:
            print("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
        print(sep)

    def _render_models(self, models: list[str]) -> None:
        current = self.model if self.direct else (self.state.active_model or "auto")
        print(self.style.info(self._t("msg.models_header", count=len(models))))
        if not models:
            print(self.style.dim(self._t("msg.models_empty")))
            return
        rows: list[list[str]] = []
        for idx, name in enumerate(models, start=1):
            marker = "*" if name == current else " "
            rows.append([str(idx), marker, name])
        self._print_table(
            [self._t("table.no"), self._t("table.current"), self._t("table.model")],
            rows,
        )
        print(self.style.dim(self._t("msg.current_model", model=current)))
        print(
            self.style.dim(
                self._t(
                    "msg.current_thinking",
                    level=self.state.thinking_level,
                    budget=self.state.thinking_budget,
                )
            )
        )

    def _render_model_table(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            print(self.style.dim(self._t("msg.model_table_empty")))
            return
        table_rows: list[list[str]] = []
        for item in rows:
            table_rows.append(
                [
                    str(item.get("model", "")),
                    str(item.get("strategy", "")),
                    str(item.get("speed_tier", "")),
                    str(item.get("quality_tier", "")),
                    str(item.get("cost_tier", "")),
                ]
            )
        self._print_table(
            [
                self._t("table.model"),
                self._t("table.strategy"),
                self._t("table.speed"),
                self._t("table.quality"),
                self._t("table.cost"),
            ],
            table_rows,
        )

    def _render_task_list(self, tasks: list[dict[str, Any]]) -> None:
        print(self.style.info(self._t("msg.task_list_header", count=len(tasks))))
        if not tasks:
            print(self.style.dim(self._t("msg.no_tasks")))
            return
        rows: list[list[str]] = []
        for task in tasks:
            payload = task.get("payload", {}) or {}
            summary_raw = str(payload.get("user_prompt") or payload.get("prompt") or "")
            rows.append(
                [
                    self._shorten(str(task.get("id", "")), 12),
                    self._shorten(str(task.get("task_type", "")), 10),
                    self._shorten(str(task.get("state", "")), 10),
                    self._format_timestamp(task.get("updated_at")),
                    self._shorten(summary_raw, 46),
                ]
            )
        self._print_table(
            [
                self._t("table.id"),
                self._t("table.type"),
                self._t("table.state"),
                self._t("table.updated"),
                self._t("table.summary"),
            ],
            rows,
        )

    def _render_task_detail(self, task: dict[str, Any]) -> None:
        print(self.style.info(self._t("msg.task_detail", id=task.get("id", ""))))
        rows = [
            [self._t("table.type"), str(task.get("task_type", "-"))],
            [self._t("table.state"), str(task.get("state", "-"))],
            [self._t("table.created"), self._format_timestamp(task.get("created_at"))],
            [self._t("table.updated"), self._format_timestamp(task.get("updated_at"))],
        ]
        self._print_table([self._t("table.field"), self._t("table.value")], rows)

        payload = task.get("payload")
        if payload:
            print(self.style.dim(f"{self._t('table.payload')}:"))
            print(self._json_text(payload, indent=2))
        result = task.get("result")
        if result:
            print(self.style.dim(f"{self._t('table.result')}:"))
            print(self._json_text(result, indent=2))
        error = task.get("error")
        if error:
            print(self.style.error(f"{self._t('table.error')}: {error}"))

    def _render_task_output(self, payload: dict[str, Any]) -> None:
        print(
            self.style.info(
                self._t(
                    "msg.task_output_header",
                    id=payload.get("task_id", ""),
                    state=payload.get("state", "-"),
                )
            )
        )
        if payload.get("result") is not None:
            print(self.style.dim(f"{self._t('table.result')}:"))
            print(self._json_text(payload.get("result"), indent=2))
        elif payload.get("error"):
            print(self.style.error(f"{self._t('table.error')}: {payload.get('error')}"))
        else:
            print(self.style.dim(self._t("msg.task_output_pending")))

    def _render_leader_run(self, run: dict[str, Any]) -> None:
        print(self.style.info(self._t("msg.leader_run_header", id=run.get("id", ""))))
        rows = [
            [self._t("table.state"), str(run.get("state", "-"))],
            [self._t("table.model"), str(run.get("worker_model", "-"))],
            [self._t("table.created"), self._format_timestamp(run.get("created_at"))],
            [self._t("table.updated"), self._format_timestamp(run.get("updated_at"))],
        ]
        self._print_table([self._t("table.field"), self._t("table.value")], rows)
        goal = str(run.get("goal", "")).strip()
        if goal:
            print(self.style.dim(f"{self._t('table.goal')}: {goal}"))
        error = run.get("error")
        if error:
            print(self.style.error(f"{self._t('table.error')}: {error}"))

    def _render_leader_jobs(self, jobs: list[dict[str, Any]]) -> None:
        print(self.style.info(self._t("msg.leader_jobs_header", count=len(jobs))))
        if not jobs:
            print(self.style.dim(self._t("msg.table_empty")))
            return
        rows: list[list[str]] = []
        for job in jobs:
            rows.append(
                [
                    str(job.get("job_index", "")),
                    self._shorten(str(job.get("name", "")), 20),
                    self._shorten(str(job.get("state", "")), 12),
                    self._shorten(str(job.get("scope", "")), 42),
                ]
            )
        self._print_table(
            [self._t("table.no"), self._t("table.name"), self._t("table.state"), self._t("table.scope")],
            rows,
        )

    def _to_image_ref(self, value: str) -> str:
        src = value.strip()
        if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            return src
        path = Path(src).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(src)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    def _prompt_api_key(self) -> str:
        return getpass(self._t("cli.api_key_prompt")).strip()

    def _consume_pending_images(self) -> list[str]:
        if not self.state.pending_images:
            return []
        images = list(self.state.pending_images)
        self.state.pending_images.clear()
        return images

    def _render_backend_error(self, exc: Exception, startup: bool = False) -> None:
        print(
            self.style.error(
                self._t(
                    "msg.backend_unreachable",
                    url=self.state.backend_url,
                    error=str(exc),
                )
            )
        )
        print(self.style.dim(self._t("msg.backend_retry_hint")))
        if startup and not self.direct:
            self.direct = True
            print(self.style.warn(self._t("msg.backend_fallback_direct", model=self.model)))

    def start(self) -> None:
        print(self.style.title(self._t("cli.title")), self.style.dim(self._t("cli.type_help")))
        if not self.direct:
            try:
                self._ensure_session(self.state.mode)
            except httpx.HTTPError as exc:
                self._render_backend_error(exc, startup=True)
        while True:
            try:
                raw = input(self._t("cli.prompt")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.startswith("/"):
                try:
                    if self._handle_command(raw):
                        break
                except httpx.HTTPError as exc:
                    self._render_backend_error(exc)
                continue
            if self.direct:
                images = self._consume_pending_images()
                asyncio.run(self._run_direct_turn(raw, images))
            else:
                images = self._consume_pending_images()
                try:
                    asyncio.run(self._run_backend_turn(raw, images))
                except httpx.HTTPError as exc:
                    self._render_backend_error(exc)

    def _handle_command(self, raw: str) -> bool:
        parts = shlex.split(raw)
        cmd = parts[0]
        if cmd == "/quit":
            return True
        if cmd == "/help":
            print(self._t("cli.help"))
            return False
        if cmd == "/login":
            api_key = parts[1].strip() if len(parts) >= 2 else self._prompt_api_key()
            if not api_key:
                print(self.style.warn(self._t("msg.login_cancelled")))
                return False
            if self.direct:
                self.settings.poe_api_key = api_key
                self.direct_model_client.api_key = api_key
                print(self.style.ok(self._t("msg.login_direct_updated")))
                return False
            self.settings.poe_api_key = api_key
            self.direct_model_client.api_key = api_key
            try:
                resp = self.http.post(
                    f"{self.state.backend_url}/auth/poe/login",
                    json={"api_key": api_key},
                )
                resp.raise_for_status()
                print(self.style.ok(self._t("msg.login_backend_updated")))
            except httpx.HTTPError as exc:
                print(self.style.warn(self._t("msg.login_backend_failed_direct_only")))
                self._render_backend_error(exc)
            return False
        if cmd == "/lang" and len(parts) == 2:
            requested = parts[1]
            if not is_supported_lang(requested):
                print(self.style.warn(self._t("msg.lang_invalid", lang=requested)))
                print(self.style.dim(f"supported: {', '.join(supported_langs())}"))
                return False
            new_lang = self.i18n.set_lang(requested)
            label = self._t(f"lang.{new_lang}")
            print(self.style.ok(self._t("msg.lang_switched", lang=label)))
            return False
        if cmd == "/system":
            self.state.system_message = raw[len("/system") :].strip()
            print(self.style.ok(self._t("msg.system_updated")))
            return False
        if cmd == "/mode" and len(parts) == 2:
            alias = parts[1].lower()
            target_mode = "planning" if alias == "plan" else alias
            if target_mode not in {"coding", "chat", "planning", "leader"}:
                print(self.style.warn(self._t("msg.invalid_mode", mode=parts[1])))
                return False
            self.state.mode = target_mode
            if not self.direct:
                self._ensure_session(self.state.mode)
            print(self.style.ok(self._t("msg.mode_set", mode=self.state.mode)))
            return False
        if cmd == "/thinking" and len(parts) >= 2:
            level = parts[1]
            budget = self.state.thinking_budget
            if len(parts) >= 3:
                try:
                    budget = int(parts[2])
                except ValueError:
                    print(self.style.warn(self._t("msg.invalid_number")))
                    return False
            if level not in {"quick", "balanced", "deep"}:
                print(self.style.warn(self._t("msg.invalid_thinking_level", level=level)))
                return False
            self.state.thinking_level = level
            self.state.thinking_budget = budget
            if self.direct:
                print(
                    self.style.ok(
                        self._t("msg.thinking_updated", level=level, budget=budget)
                    )
                )
                return False
            resp = self.http.post(
                f"{self.state.backend_url}/sessions/{self.state.session_id}/thinking",
                json={"thinking_level": level, "thinking_budget": budget},
            )
            resp.raise_for_status()
            print(self.style.ok(self._t("msg.thinking_updated", level=level, budget=budget)))
            return False
        if cmd == "/commandpolicy" and len(parts) >= 2:
            flag = parts[1].lower()
            if flag not in {"allow", "deny"}:
                print(self.style.warn(self._t("msg.command_policy_usage")))
                return False
            allow = flag == "allow"
            encourage = self.state.encourage_model_command_create
            if len(parts) >= 3:
                sub = parts[2].lower()
                if sub in {"encourage", "on", "true"}:
                    encourage = True
                elif sub in {"noencourage", "off", "false"}:
                    encourage = False
                else:
                    print(self.style.warn(self._t("msg.command_policy_usage")))
                    return False
            self.state.allow_model_command_create = allow
            self.state.encourage_model_command_create = encourage
            if self.direct:
                print(
                    self.style.ok(
                        self._t("msg.command_policy_updated", allow=str(allow).lower(), encourage=str(encourage).lower())
                    )
                )
                return False
            resp = self.http.post(
                f"{self.state.backend_url}/sessions/{self.state.session_id}/command-policy",
                json={
                    "allow_model_command_create": allow,
                    "encourage_model_command_create": encourage,
                },
            )
            resp.raise_for_status()
            print(
                self.style.ok(
                    self._t("msg.command_policy_updated", allow=str(allow).lower(), encourage=str(encourage).lower())
                )
            )
            return False
        if cmd == "/plan":
            self.state.mode = "planning"
            self.state.system_message = PLAN_SYSTEM_MESSAGE
            if not self.direct:
                self._ensure_session(self.state.mode)
            print(self.style.ok(self._t("msg.plan_mode_enabled")))
            return False
        if cmd == "/image" and len(parts) == 2:
            try:
                image_ref = self._to_image_ref(parts[1])
            except FileNotFoundError:
                print(self.style.warn(self._t("msg.image_not_found", path=parts[1])))
                return False
            self.state.pending_images.append(image_ref)
            print(self.style.ok(self._t("msg.image_added", count=len(self.state.pending_images))))
            return False
        if cmd == "/images":
            if not self.state.pending_images:
                print(self.style.dim(self._t("msg.images_empty")))
                return False
            rows = []
            for idx, image in enumerate(self.state.pending_images, start=1):
                label = image if image.startswith("http") else "data:image/*"
                rows.append([str(idx), self._shorten(label, 64)])
            self._print_table([self._t("table.no"), self._t("table.image")], rows)
            return False
        if cmd == "/clearimages":
            self.state.pending_images.clear()
            print(self.style.ok(self._t("msg.images_cleared")))
            return False
        if cmd in {"/listmodels", "/models"}:
            if self.direct:
                self._render_models(self.settings.supported_models)
                return False
            resp = self.http.get(f"{self.state.backend_url}/models", params={"refresh": "true"})
            resp.raise_for_status()
            data = resp.json()
            self._render_models(data.get("models", []))
            return False
        if cmd == "/modeltable":
            if self.direct:
                print(self.style.warn(self._t("msg.modeltable_backend_only")))
                return False
            resp = self.http.get(f"{self.state.backend_url}/models/table")
            resp.raise_for_status()
            self._render_model_table(resp.json())
            return False
        if cmd in {"/changemodel", "/model"} and len(parts) == 2:
            target = parts[1]
            if self.direct:
                self.model = target
                print(self.style.ok(self._t("msg.direct_model_set", model=self.model)))
                return False
            resp = self.http.post(
                f"{self.state.backend_url}/sessions/{self.state.session_id}/change-model",
                json={"model": target},
            )
            resp.raise_for_status()
            updated = resp.json()
            self.state.active_model = updated.get("active_model")
            print(self.style.ok(self._t("msg.session_model_changed", model=updated["active_model"])))
            return False
        if cmd == "/balance":
            if self.direct:
                print(self.style.warn(self._t("msg.balance_backend_only")))
                return False
            resp = self.http.get(f"{self.state.backend_url}/usage/current_balance")
            resp.raise_for_status()
            data = resp.json()
            points = data.get("current_point_balance")
            print(self.style.info(self._t("msg.current_balance", points=points)))
            return False
        if cmd == "/context" and len(parts) >= 3:
            key = parts[1]
            try:
                value = json.loads(" ".join(parts[2:]))
            except json.JSONDecodeError:
                print(self.style.warn(self._t("msg.context_json_invalid")))
                return False
            if self.direct:
                self.state.local_context[key] = value
                print(self.style.ok(self._t("msg.local_context_updated")))
                return False
            self.http.put(
                f"{self.state.backend_url}/sessions/{self.state.session_id}/context",
                json={"key": key, "value": value, "scope": "pinned"},
            ).raise_for_status()
            print(self.style.ok(self._t("msg.context_stored")))
            return False
        if cmd == "/memory" and len(parts) >= 3:
            scope = parts[1]
            text = " ".join(parts[2:])
            if self.direct:
                print(self.style.warn(self._t("msg.memory_backend_only")))
                return False
            payload = {
                "scope": scope,
                "content": text,
                "session_id": self.state.session_id,
                "project_id": self.state.project_id,
                "priority": 0,
                "tags": [],
            }
            self.http.post(f"{self.state.backend_url}/memory/write", json=payload).raise_for_status()
            print(self.style.ok(self._t("msg.memory_stored")))
            return False
        if cmd == "/wiki" and len(parts) >= 3:
            topic = parts[1]
            text = " ".join(parts[2:])
            if self.direct:
                print(self.style.warn(self._t("msg.wiki_backend_only")))
                return False
            self.http.post(
                f"{self.state.backend_url}/wiki/ingest",
                json={"project_id": self.state.project_id, "topic": topic, "content": text, "source": "cli"},
            ).raise_for_status()
            print(self.style.ok(self._t("msg.wiki_updated")))
            return False
        if cmd == "/subagent" and len(parts) >= 4:
            if self.direct:
                print(self.style.warn(self._t("msg.subagent_backend_only")))
                return False
            model = parts[1]
            perm = parts[2]
            prompt = " ".join(parts[3:])
            resp = self.http.post(
                f"{self.state.backend_url}/subagents/start",
                json={
                    "parent_session_id": self.state.session_id,
                    "model": model,
                    "perm": perm,
                    "prompt": prompt,
                    "images": self._consume_pending_images(),
                    "context_share": [],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            print(self.style.ok(self._t("msg.subagent_started", id=data["id"])))
            return False
        if cmd == "/review" and len(parts) >= 2:
            if self.direct:
                print(self.style.warn(self._t("msg.review_backend_only")))
                return False
            prompt = raw[len("/review") :].strip()
            resp = self.http.post(
                f"{self.state.backend_url}/review",
                json={
                    "session_id": self.state.session_id,
                    "prompt": prompt,
                    "model": None,
                    "thinking_level": self.state.thinking_level,
                    "thinking_budget": self.state.thinking_budget,
                    "context_keys": [],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            print(self.style.info(self._t("msg.review_header", model=data.get("model"))))
            print(data.get("output_text", ""))
            return False
        if cmd == "/reviewsettings":
            if self.direct:
                print(self.style.warn(self._t("msg.review_backend_only")))
                return False
            if len(parts) == 1:
                resp = self.http.get(f"{self.state.backend_url}/review/settings")
                resp.raise_for_status()
                print(self._json_text(resp.json(), indent=2))
                return False
            if len(parts) >= 4:
                model = parts[1]
                level = parts[2]
                try:
                    budget = int(parts[3])
                except ValueError:
                    print(self.style.warn(self._t("msg.invalid_number")))
                    return False
                resp = self.http.post(
                    f"{self.state.backend_url}/review/settings",
                    json={"model": model, "thinking_level": level, "thinking_budget": budget},
                )
                resp.raise_for_status()
                print(self.style.ok(self._t("msg.review_settings_updated")))
                print(self._json_text(resp.json(), indent=2))
                return False
            print(self.style.warn(self._t("msg.review_settings_usage")))
            return False
        if cmd == "/bgturn" and len(parts) >= 2:
            if self.direct:
                print(self.style.warn(self._t("msg.task_backend_only")))
                return False
            prompt = raw[len("/bgturn") :].strip()
            resp = self.http.post(
                f"{self.state.backend_url}/tasks/turns/start",
                json={
                    "session_id": self.state.session_id,
                    "user_prompt": prompt,
                    "system_message": self.state.system_message or None,
                    "images": self._consume_pending_images(),
                    "thinking_level": self.state.thinking_level,
                    "thinking_budget": self.state.thinking_budget,
                    "context_keys": [],
                    "metadata": {},
                },
            )
            resp.raise_for_status()
            task = resp.json()
            print(self.style.ok(self._t("msg.task_started", id=task["id"])))
            return False
        if cmd == "/bgsubagent" and len(parts) >= 4:
            if self.direct:
                print(self.style.warn(self._t("msg.task_backend_only")))
                return False
            model = parts[1]
            perm = parts[2]
            prompt = " ".join(parts[3:])
            resp = self.http.post(
                f"{self.state.backend_url}/tasks/subagents/start",
                json={
                    "parent_session_id": self.state.session_id,
                    "model": model,
                    "perm": perm,
                    "prompt": prompt,
                    "images": self._consume_pending_images(),
                    "context_share": [],
                    "wait_timeout_s": 600,
                },
            )
            resp.raise_for_status()
            task = resp.json()
            print(self.style.ok(self._t("msg.task_started", id=task["id"])))
            return False
        if cmd == "/leader" and len(parts) >= 2:
            if self.direct:
                print(self.style.warn(self._t("msg.leader_backend_only")))
                return False
            goal = raw[len("/leader") :].strip()
            resp = self.http.post(
                f"{self.state.backend_url}/leader/start",
                json={
                    "session_id": self.state.session_id,
                    "goal": goal,
                    "context_keys": [],
                    "verify_command": None,
                },
            )
            resp.raise_for_status()
            run = resp.json()
            print(self.style.ok(self._t("msg.leader_started", id=run.get("id", ""))))
            self._render_leader_run(run)
            return False
        if cmd == "/leaderstatus" and len(parts) == 2:
            if self.direct:
                print(self.style.warn(self._t("msg.leader_backend_only")))
                return False
            run_id = parts[1]
            resp = self.http.get(f"{self.state.backend_url}/leader/{run_id}")
            resp.raise_for_status()
            self._render_leader_run(resp.json())
            return False
        if cmd == "/leaderjobs" and len(parts) == 2:
            if self.direct:
                print(self.style.warn(self._t("msg.leader_backend_only")))
                return False
            run_id = parts[1]
            resp = self.http.get(f"{self.state.backend_url}/leader/{run_id}/jobs")
            resp.raise_for_status()
            self._render_leader_jobs(resp.json())
            return False
        if cmd == "/leaderwait" and len(parts) in {2, 3}:
            if self.direct:
                print(self.style.warn(self._t("msg.leader_backend_only")))
                return False
            run_id = parts[1]
            timeout_s = 120
            if len(parts) == 3:
                try:
                    timeout_s = int(parts[2])
                except ValueError:
                    print(self.style.warn(self._t("msg.invalid_number")))
                    return False
            resp = self.http.post(
                f"{self.state.backend_url}/leader/{run_id}/wait",
                json={"timeout_s": timeout_s},
            )
            resp.raise_for_status()
            self._render_leader_run(resp.json())
            return False
        if cmd == "/leadercancel" and len(parts) == 2:
            if self.direct:
                print(self.style.warn(self._t("msg.leader_backend_only")))
                return False
            run_id = parts[1]
            resp = self.http.post(f"{self.state.backend_url}/leader/{run_id}/cancel")
            resp.raise_for_status()
            self._render_leader_run(resp.json())
            return False
        if cmd == "/tasks":
            if self.direct:
                print(self.style.warn(self._t("msg.task_backend_only")))
                return False
            params: dict[str, str] = {"limit": "20"}
            if len(parts) >= 2:
                params["state"] = parts[1]
            resp = self.http.get(f"{self.state.backend_url}/tasks", params=params)
            resp.raise_for_status()
            data = resp.json()
            self._render_task_list(data)
            return False
        if cmd == "/task" and len(parts) == 2:
            if self.direct:
                print(self.style.warn(self._t("msg.task_backend_only")))
                return False
            task_id = parts[1]
            resp = self.http.get(f"{self.state.backend_url}/tasks/{task_id}")
            resp.raise_for_status()
            self._render_task_detail(resp.json())
            return False
        if cmd in {"/readtaskoutput", "/taskoutput"} and len(parts) == 2:
            if self.direct:
                print(self.style.warn(self._t("msg.task_backend_only")))
                return False
            task_id = parts[1]
            resp = self.http.get(f"{self.state.backend_url}/tasks/{task_id}/output")
            resp.raise_for_status()
            self._render_task_output(resp.json())
            return False
        if cmd == "/canceltask" and len(parts) == 2:
            if self.direct:
                print(self.style.warn(self._t("msg.task_backend_only")))
                return False
            task_id = parts[1]
            resp = self.http.post(f"{self.state.backend_url}/tasks/{task_id}/cancel")
            resp.raise_for_status()
            self._render_task_detail(resp.json())
            return False
        if cmd == "/shell" and len(parts) >= 3:
            if self.direct:
                print(self.style.warn(self._t("msg.shell_backend_only")))
                return False
            danger = int(parts[1])
            command = " ".join(parts[2:])
            resp = self.http.post(
                f"{self.state.backend_url}/shell/run",
                json={
                    "session_id": self.state.session_id,
                    "command": command,
                    "danger_level": danger,
                    "timeout_s": 120,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            print(json.dumps(data, indent=2, ensure_ascii=True))
            return False

        print(self.style.warn(self._t("msg.unknown_command")))
        return False

    def _ensure_session(self, mode: str) -> None:
        resp = self.http.post(
            f"{self.state.backend_url}/sessions",
            json={
                "mode": mode,
                "project_id": self.state.project_id,
                "policy_profile": "research",
                "thinking_level": self.state.thinking_level,
                "thinking_budget": self.state.thinking_budget,
                "allow_model_command_create": self.state.allow_model_command_create,
                "encourage_model_command_create": self.state.encourage_model_command_create,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self.state.session_id = data["id"]
        self.state.active_model = data.get("active_model")
        self.state.thinking_level = data.get("thinking_level", self.state.thinking_level)
        self.state.thinking_budget = int(data.get("thinking_budget", self.state.thinking_budget))
        self.state.allow_model_command_create = bool(
            data.get("allow_model_command_create", self.state.allow_model_command_create)
        )
        self.state.encourage_model_command_create = bool(
            data.get("encourage_model_command_create", self.state.encourage_model_command_create)
        )
        print(self.style.dim(self._t("msg.session_id", session_id=self.state.session_id)))

    async def _run_backend_turn(self, prompt: str, images: list[str] | None = None) -> None:
        payload = {
            "session_id": self.state.session_id,
            "user_prompt": prompt,
            "system_message": self.state.system_message or None,
            "images": images or [],
            "thinking_level": self.state.thinking_level,
            "thinking_budget": self.state.thinking_budget,
            "context_keys": [],
            "metadata": {},
        }
        saw_delta = False
        async with self.async_http.stream(
            "POST",
            f"{self.state.backend_url}/turns/execute/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                etype = event.get("type")
                data = event.get("data")
                if etype == "status":
                    status_key = str(data)
                    label = self._t(f"status.{status_key}")
                    if label.startswith("status."):
                        label = status_key
                    print(self.style.dim(f"[{label}]"))
                    continue
                if etype == "model":
                    print(self.style.info(self._t("stream.model", model=data.get("model"))))
                    continue
                if etype == "tool":
                    name = data.get("name", "tool")
                    print(self.style.info(self._t("stream.tool", name=name)))
                    continue
                if etype == "delta":
                    if not saw_delta:
                        print(self.style.title(self._t("stream.assistant")), end="")
                        saw_delta = True
                    print(data, end="", flush=True)
                    continue
                if etype == "final":
                    if saw_delta:
                        print()
                        meta = data if isinstance(data, dict) else {}
                        if meta.get("model"):
                            self.state.active_model = str(meta.get("model"))
                        print(
                            self.style.dim(
                                self._t(
                                    "stream.done",
                                    model=meta.get("model"),
                                    tools=len(meta.get("tool_events", [])),
                                )
                            )
                        )
                    else:
                        text = data.get("output_text", "") if isinstance(data, dict) else ""
                        print(text)
                    continue

    async def _run_direct_turn(self, prompt: str, images: list[str] | None = None) -> None:
        direct_context = dict(self.state.local_context)
        direct_context["model_settings"] = {
            "thinking_level": self.state.thinking_level,
            "thinking_budget": self.state.thinking_budget,
        }
        reply = await self.direct_model_client.chat(
            model=self.model,
            system_message=self.state.system_message,
            user_prompt=prompt,
            context=direct_context,
            images=images or [],
        )
        print(self.style.title(self._t("stream.assistant")) + reply.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="PoeCoder CLI")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    parser.add_argument("--direct", action="store_true", help="Call model directly without backend")
    parser.add_argument("--model", default=None)
    parser.add_argument("--lang", default=None, help="CLI language: en or zh-cn")
    args = parser.parse_args()

    cli = PoeCoderCLI(backend_url=args.backend_url, direct=args.direct, model=args.model, lang=args.lang)
    cli.start()


if __name__ == "__main__":
    main()
