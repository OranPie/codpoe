from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any

import httpx

from poecoder.config import get_settings
from poecoder.services.model_clients import PoeModelClient


HELP_TEXT = """
Commands:
  /help                               Show this help
  /quit                               Exit CLI
  /system <text>                      Set system message
  /mode <coding|chat|planning>        Start new backend session in mode
  /listmodels                         List supported models
  /changemodel <name|auto>            Change active main model
  /balance                            Fetch current Poe point balance
  /context <key> <json>               Store context key/value in backend session
  /memory <scope> <text>              Write memory entry (session|project|global)
  /wiki <topic> <text>                Add project wiki note
  /subagent <model> <perm> <prompt>   Start subagent
  /shell <danger> <command>           Run shell command through policy engine
"""


@dataclass(slots=True)
class CliState:
    backend_url: str
    session_id: str | None = None
    mode: str = "coding"
    system_message: str = ""
    project_id: str = "default"
    local_context: dict[str, Any] = field(default_factory=dict)


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


class PoeCoderCLI:
    def __init__(self, backend_url: str, direct: bool, model: str | None) -> None:
        self.settings = get_settings()
        self.direct = direct
        self.model = model or self.settings.default_large_model
        self.state = CliState(backend_url=backend_url)
        self.http = httpx.Client(timeout=90.0)
        self.async_http = httpx.AsyncClient(timeout=90.0)
        self.direct_model_client = PoeModelClient(self.settings.poe_api_url, self.settings.poe_api_key)
        self.style = Style()

    def start(self) -> None:
        print(self.style.title("PoeCoder CLI"), self.style.dim("(type /help)"))
        if not self.direct:
            self._ensure_session(self.state.mode)
        while True:
            try:
                raw = input("poecoder> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.startswith("/"):
                if self._handle_command(raw):
                    break
                continue
            if self.direct:
                asyncio.run(self._run_direct_turn(raw))
            else:
                asyncio.run(self._run_backend_turn(raw))

    def _handle_command(self, raw: str) -> bool:
        parts = shlex.split(raw)
        cmd = parts[0]
        if cmd == "/quit":
            return True
        if cmd == "/help":
            print(HELP_TEXT.strip())
            return False
        if cmd == "/system":
            self.state.system_message = raw[len("/system") :].strip()
            print(self.style.ok("System message updated."))
            return False
        if cmd == "/mode" and len(parts) == 2:
            self.state.mode = parts[1]
            if not self.direct:
                self._ensure_session(self.state.mode)
            print(self.style.ok(f"Mode set to {self.state.mode}"))
            return False
        if cmd in {"/listmodels", "/models"}:
            if self.direct:
                print(json.dumps({"models": self.settings.supported_models}, ensure_ascii=True))
                return False
            resp = self.http.get(f"{self.state.backend_url}/models", params={"refresh": "true"})
            resp.raise_for_status()
            data = resp.json()
            print(self.style.info(f"models ({len(data.get('models', []))})"))
            print(json.dumps(data, ensure_ascii=True))
            return False
        if cmd in {"/changemodel", "/model"} and len(parts) == 2:
            target = parts[1]
            if self.direct:
                self.model = target
                print(self.style.ok(f"Direct model set to {self.model}"))
                return False
            resp = self.http.post(
                f"{self.state.backend_url}/sessions/{self.state.session_id}/change-model",
                json={"model": target},
            )
            resp.raise_for_status()
            updated = resp.json()
            print(self.style.ok(f"Session model changed to {updated['active_model']}"))
            return False
        if cmd == "/balance":
            if self.direct:
                print(self.style.warn("Balance requires backend mode."))
                return False
            resp = self.http.get(f"{self.state.backend_url}/usage/current_balance")
            resp.raise_for_status()
            data = resp.json()
            points = data.get("current_point_balance")
            print(self.style.info(f"Current balance: {points} points"))
            return False
        if cmd == "/context" and len(parts) >= 3:
            key = parts[1]
            try:
                value = json.loads(" ".join(parts[2:]))
            except json.JSONDecodeError:
                print(self.style.warn("Context value must be valid JSON"))
                return False
            if self.direct:
                self.state.local_context[key] = value
                print(self.style.ok("Local context updated."))
                return False
            self.http.put(
                f"{self.state.backend_url}/sessions/{self.state.session_id}/context",
                json={"key": key, "value": value, "scope": "pinned"},
            ).raise_for_status()
            print(self.style.ok("Context stored."))
            return False
        if cmd == "/memory" and len(parts) >= 3:
            scope = parts[1]
            text = " ".join(parts[2:])
            if self.direct:
                print(self.style.warn("Memory API requires backend mode."))
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
            print(self.style.ok("Memory stored."))
            return False
        if cmd == "/wiki" and len(parts) >= 3:
            topic = parts[1]
            text = " ".join(parts[2:])
            if self.direct:
                print(self.style.warn("Wiki API requires backend mode."))
                return False
            self.http.post(
                f"{self.state.backend_url}/wiki/ingest",
                json={"project_id": self.state.project_id, "topic": topic, "content": text, "source": "cli"},
            ).raise_for_status()
            print(self.style.ok("Wiki updated."))
            return False
        if cmd == "/subagent" and len(parts) >= 4:
            if self.direct:
                print(self.style.warn("Subagent API requires backend mode."))
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
                    "context_share": [],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            print(self.style.ok(f"Subagent started: {data['id']}"))
            return False
        if cmd == "/shell" and len(parts) >= 3:
            if self.direct:
                print(self.style.warn("Shell API requires backend mode."))
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

        print(self.style.warn("Unknown command. Use /help"))
        return False

    def _ensure_session(self, mode: str) -> None:
        resp = self.http.post(
            f"{self.state.backend_url}/sessions",
            json={"mode": mode, "project_id": self.state.project_id, "policy_profile": "research"},
        )
        resp.raise_for_status()
        self.state.session_id = resp.json()["id"]
        print(self.style.dim(f"session={self.state.session_id}"))

    async def _run_backend_turn(self, prompt: str) -> None:
        payload = {
            "session_id": self.state.session_id,
            "user_prompt": prompt,
            "system_message": self.state.system_message or None,
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
                    print(self.style.dim(f"[{data}]"))
                    continue
                if etype == "model":
                    print(self.style.info(f"model={data.get('model')}"))
                    continue
                if etype == "tool":
                    name = data.get("name", "tool")
                    print(self.style.info(f"tool:{name}"))
                    continue
                if etype == "delta":
                    if not saw_delta:
                        print(self.style.title("assistant> "), end="")
                        saw_delta = True
                    print(data, end="", flush=True)
                    continue
                if etype == "final":
                    if saw_delta:
                        print()
                        meta = data if isinstance(data, dict) else {}
                        print(self.style.dim(f"done model={meta.get('model')} tools={len(meta.get('tool_events', []))}"))
                    else:
                        text = data.get("output_text", "") if isinstance(data, dict) else ""
                        print(text)
                    continue

    async def _run_direct_turn(self, prompt: str) -> None:
        reply = await self.direct_model_client.chat(
            model=self.model,
            system_message=self.state.system_message,
            user_prompt=prompt,
            context=self.state.local_context,
        )
        print(self.style.title("assistant> ") + reply.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="PoeCoder CLI")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8765")
    parser.add_argument("--direct", action="store_true", help="Call model directly without backend")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    cli = PoeCoderCLI(backend_url=args.backend_url, direct=args.direct, model=args.model)
    cli.start()


if __name__ == "__main__":
    main()
