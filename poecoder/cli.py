from __future__ import annotations

import argparse
import curses
import json
import queue
import threading
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(slots=True)
class ApiClient:
    base_url: str
    timeout_s: int = 300
    _client: httpx.Client = field(init=False, repr=False)
    api: str = field(init=False)

    def __post_init__(self) -> None:
        self.api = self.base_url.rstrip("/") + "/agent-api"
        self._client = httpx.Client(timeout=self.timeout_s)

    def close(self) -> None:
        self._client.close()

    def about(self) -> dict[str, Any]:
        return self._get("/about")

    def create_session(self, title: str = "cui-session") -> dict[str, Any]:
        return self._post("/sessions", {"title": title})

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        resp = self._client.get(f"{self.api}/sessions", params={"limit": limit})
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def list_messages(self, session_id: str, limit: int = 80) -> list[dict[str, Any]]:
        resp = self._client.get(f"{self.api}/sessions/{session_id}/messages", params={"limit": limit})
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def turn(self, session_id: str, prompt: str, model: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        clean_model = (model or "").strip()
        if clean_model:
            payload["model"] = clean_model
        return self._post(f"/sessions/{session_id}/turn", payload)

    def arxiv(self, session_id: str, query: str, max_results: int = 5) -> dict[str, Any]:
        return self._post(
            "/workflows/arxiv",
            {"session_id": session_id, "query": query, "max_results": max_results},
        )

    def list_models(self, query: str = "", limit: int = 100, full: bool = False) -> dict[str, Any]:
        resp = self._client.get(
            f"{self.api}/models",
            params={"query": query, "limit": limit, "full": str(bool(full)).lower()},
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def get_agent(self, agent_id: str, event_limit: int = 200) -> dict[str, Any]:
        resp = self._client.get(f"{self.api}/agents/{agent_id}", params={"event_limit": event_limit})
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def cancel_agent(self, agent_id: str) -> dict[str, Any]:
        return self._post(f"/agents/{agent_id}/cancel", {})

    def turn_stream(self, session_id: str, prompt: str, model: str | None = None):
        request_payload: dict[str, Any] = {"prompt": prompt}
        clean_model = (model or "").strip()
        if clean_model:
            request_payload["model"] = clean_model

        with self._client.stream(
            "POST",
            f"{self.api}/sessions/{session_id}/turn/stream",
            json=request_payload,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            event_name = "message"
            data_lines: list[str] = []
            for raw_line in resp.iter_lines():
                line = raw_line.strip()
                if not line:
                    if not data_lines:
                        continue
                    joined = "\n".join(data_lines)
                    try:
                        parsed = json.loads(joined)
                    except Exception:  # noqa: BLE001
                        parsed = {"raw": joined}
                    yield {"event": event_name, "data": parsed}
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip() or "message"
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())

    def auth_poe(self, api_key: str) -> dict[str, Any]:
        return self._post("/auth/poe/login", {"api_key": api_key})

    def auth_openai(self, api_key: str) -> dict[str, Any]:
        return self._post("/auth/openai/login", {"api_key": api_key})

    def save_secrets(self, user_key: str) -> dict[str, Any]:
        return self._post("/auth/secrets/save", {"user_key": user_key})

    def load_secrets(self, user_key: str) -> dict[str, Any]:
        return self._post("/auth/secrets/load", {"user_key": user_key})

    def _get(self, path: str) -> dict[str, Any]:
        resp = self._client.get(f"{self.api}{path}")
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else {}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post(f"{self.api}{path}", json=payload)
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, dict) else {}


@dataclass(slots=True)
class CUIApp:
    api: ApiClient
    messages: list[dict[str, str]] = field(default_factory=list)
    input_buffer: str = ""
    status: str = "Ready"
    session_id: str = ""
    version: str = ""
    current_model: str = "auto"
    stream_mode: bool = True
    last_agent_id: str = ""
    running: bool = True
    scroll: int = 0
    stream_events_seen: int = 0
    _last_progress_signature: str = ""
    pending_ask: dict[str, Any] | None = None
    stream_inflight: bool = False
    popup_visible: bool = False
    popup_title: str = ""
    popup_lines: list[str] = field(default_factory=list)
    _stdscr: Any = None
    _colors_enabled: bool = False

    def run(self, stdscr: Any) -> None:
        self._stdscr = stdscr
        self._setup_screen()
        self._bootstrap()
        while self.running:
            self._render()
            try:
                key = stdscr.get_wch()
            except curses.error:
                continue
            self._handle_key(key)

    def _setup_screen(self) -> None:
        assert self._stdscr is not None
        stdscr = self._stdscr
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.timeout(-1)
        self._setup_colors()
        try:
            curses.curs_set(1)
        except curses.error:
            pass

    def _setup_colors(self) -> None:
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)   # header
            curses.init_pair(2, curses.COLOR_CYAN, -1)                   # system
            curses.init_pair(3, curses.COLOR_GREEN, -1)                  # assistant
            curses.init_pair(4, curses.COLOR_YELLOW, -1)                 # user
            curses.init_pair(5, curses.COLOR_MAGENTA, -1)                # progress
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)   # sidebar title
            curses.init_pair(7, curses.COLOR_RED, -1)                    # error
            curses.init_pair(8, curses.COLOR_YELLOW, -1)                 # ask
            self._colors_enabled = True
        except curses.error:
            self._colors_enabled = False

    def _role_attr(self, role: str) -> int:
        if not self._colors_enabled:
            return 0
        if role == "user":
            return curses.color_pair(4)
        if role == "assistant":
            return curses.color_pair(3)
        if role == "assistant_progress":
            return curses.color_pair(5)
        if role == "assistant_ask":
            return curses.color_pair(8) | curses.A_BOLD
        if role == "system_error":
            return curses.color_pair(7)
        if role == "system":
            return curses.color_pair(2)
        return 0

    def _bootstrap(self) -> None:
        try:
            about = self.api.about()
            self.version = str(about.get("version_tag", "unknown"))
            created = self.api.create_session("cui-session")
            self.session_id = str(created.get("id", ""))
            self._push_message(
                "system",
                "CUI ready. Commands: /help /new /sessions /switch /models /model /stream /agentinfo /ask /answer /skipask /authpoe /authopenai /secretssave /secretsload /arxiv /exit",
            )
            self._set_status("Connected")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"Bootstrap failed: {exc}")
            self._set_status("Offline")

    def _set_status(self, text: str) -> None:
        self.status = f"{text} @ {time.strftime('%H:%M:%S')}"

    def _open_popup(self, title: str, lines: list[str]) -> None:
        clean_title = (title or "").strip() or "Info"
        clean_lines = [str(item).strip() for item in lines if str(item).strip()]
        self.popup_title = clean_title[:80]
        self.popup_lines = clean_lines[:16] if clean_lines else ["(empty)"]
        self.popup_visible = True

    def _close_popup(self) -> None:
        self.popup_visible = False
        self.popup_title = ""
        self.popup_lines = []

    def _push_message(self, role: str, content: str) -> None:
        if not content.strip():
            return
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > 600:
            self.messages = self.messages[-600:]
        self.scroll = 0

    def _handle_key(self, key: Any) -> None:
        if key in (27, "\x1b"):
            if self.popup_visible:
                self._close_popup()
                self._set_status("Popup closed")
            return
        if key in ("\n", "\r"):
            self._submit_input()
            return

        if key == curses.KEY_RESIZE:
            return
        if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
            self.input_buffer = self.input_buffer[:-1]
            return
        if key == curses.KEY_UP:
            self.scroll += 1
            return
        if key == curses.KEY_DOWN:
            self.scroll = max(0, self.scroll - 1)
            return
        if key == curses.KEY_PPAGE:
            self.scroll += 10
            return
        if key == curses.KEY_NPAGE:
            self.scroll = max(0, self.scroll - 10)
            return
        if key == curses.KEY_F2:
            self._command_new_session("cui-session")
            return
        if key == curses.KEY_F5:
            self._reload_messages()
            return

        if isinstance(key, str) and key.isprintable():
            self.input_buffer += key

    def _submit_input(self) -> None:
        text = self.input_buffer.strip()
        self.input_buffer = ""
        if not text:
            return
        if text in {"/exit", "/quit"}:
            self.running = False
            return
        if text == "/help":
            self._push_message(
                "system",
                (
                    "Commands:\n"
                    "/new [title] - create and switch to new session\n"
                    "/sessions - list latest sessions\n"
                    "/switch <id-prefix> - switch session\n"
                    "/models [query] - list/search configured models\n"
                    "/models full [query] - fetch full model list (includes OpenAI remote)\n"
                    "/model [name] - show or change model for turns\n"
                    "/stream [on|off] - toggle turn event streaming\n"
                    "/agentinfo [agent_id] - show runshell/actions/cost info\n"
                    "/ask - show pending clarification request\n"
                    "/answer <text or option ids> - answer pending clarification\n"
                    "/skipask - clear pending clarification request\n"
                    "/authpoe <api_key> - set Poe API key\n"
                    "/authopenai <api_key> - set OpenAI API key\n"
                    "/secretssave <user_key> - save current keys to encrypted file\n"
                    "/secretsload <user_key> - load keys from encrypted file\n"
                    "/arxiv <query> - run multi-agent arxiv workflow\n"
                    "Keys: Esc stop stream/close popup, F2 new session, F5 reload, PgUp/PgDn scroll"
                ),
            )
            return
        if text.startswith("/new"):
            self._command_new_session(text[4:].strip() or "cui-session")
            return
        if text == "/sessions":
            self._command_sessions()
            return
        if text.startswith("/switch "):
            self._command_switch(text.split(" ", 1)[1].strip())
            return
        if text == "/models":
            self._command_models("", full=False)
            return
        if text.startswith("/models "):
            raw = text.split(" ", 1)[1].strip()
            if raw == "full":
                self._command_models("", full=True)
                return
            if raw.startswith("full "):
                self._command_models(raw.split(" ", 1)[1].strip(), full=True)
                return
            self._command_models(raw, full=False)
            return
        if text == "/model":
            self._push_message("system", f"Current model: {self.current_model}")
            return
        if text.startswith("/model "):
            self._command_model(text.split(" ", 1)[1].strip())
            return
        if text == "/stream":
            self._push_message("system", f"stream mode: {'on' if self.stream_mode else 'off'}")
            return
        if text.startswith("/stream "):
            self._command_stream_mode(text.split(" ", 1)[1].strip())
            return
        if text == "/agentinfo":
            self._command_agent_info(self.last_agent_id)
            return
        if text.startswith("/agentinfo "):
            self._command_agent_info(text.split(" ", 1)[1].strip())
            return
        if text == "/ask":
            self._command_show_pending_ask()
            return
        if text == "/skipask":
            self.pending_ask = None
            self._close_popup()
            self._push_message("system", "Pending ask cleared.")
            return
        if text == "/answer":
            self._push_message("system", "Usage: /answer <text or option ids>")
            return
        if text.startswith("/answer "):
            self._handle_pending_ask_input(text.split(" ", 1)[1].strip())
            return
        if text.startswith("/authpoe "):
            self._command_auth_poe(text.split(" ", 1)[1].strip())
            return
        if text.startswith("/authopenai "):
            self._command_auth_openai(text.split(" ", 1)[1].strip())
            return
        if text.startswith("/secretssave "):
            self._command_secrets_save(text.split(" ", 1)[1].strip())
            return
        if text.startswith("/secretsload "):
            self._command_secrets_load(text.split(" ", 1)[1].strip())
            return
        if text.startswith("/arxiv "):
            self._command_arxiv(text[len("/arxiv ") :].strip())
            return
        if text.startswith("/"):
            self._push_message("system", f"Unknown command: {text}. Use /help.")
            return
        if self.pending_ask is not None:
            self._handle_pending_ask_input(text)
            return

        self._execute_turn(text, user_visible=text)

    def _command_new_session(self, title: str) -> None:
        self._set_status("Creating session...")
        self._render()
        try:
            created = self.api.create_session(title)
            self.session_id = str(created.get("id", ""))
            self.last_agent_id = ""
            self.stream_events_seen = 0
            self._last_progress_signature = ""
            self.pending_ask = None
            self._close_popup()
            self.messages.clear()
            self._push_message("system", f"Switched to session {self.session_id}")
            self._set_status("Session created")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"Create session failed: {exc}")
            self._set_status("Error")

    def _command_sessions(self) -> None:
        self._set_status("Loading sessions...")
        self._render()
        try:
            rows = self.api.list_sessions(limit=12)
            if not rows:
                self._push_message("system", "No sessions found.")
                self._set_status("Done")
                return
            lines = ["Recent sessions:"]
            for item in rows:
                sid = str(item.get("id", ""))[:12]
                title = str(item.get("title", "") or "-")
                lines.append(f"- {sid}  {title}")
            self._push_message("system", "\n".join(lines))
            self._set_status("Done")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"List sessions failed: {exc}")
            self._set_status("Error")

    def _command_switch(self, prefix: str) -> None:
        target = prefix.strip()
        if not target:
            self._push_message("system", "Usage: /switch <id-prefix>")
            return
        self._set_status("Switching session...")
        self._render()
        try:
            rows = self.api.list_sessions(limit=40)
            matches = [item for item in rows if str(item.get("id", "")).startswith(target)]
            if len(matches) != 1:
                self._push_message("system", f"Switch needs exactly one match. found={len(matches)}")
                self._set_status("Ambiguous")
                return
            self.session_id = str(matches[0].get("id", ""))
            self.pending_ask = None
            self._close_popup()
            self._reload_messages()
            self._push_message("system", f"Switched to {self.session_id}")
            self._set_status("Session switched")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"Switch failed: {exc}")
            self._set_status("Error")

    def _command_models(self, query: str, *, full: bool) -> None:
        self._set_status("Loading models...")
        self._render()
        try:
            payload = self.api.list_models(query=query, limit=400 if full else 80, full=full)
            models = payload.get("models", [])
            if not isinstance(models, list):
                models = []
            if not models:
                self._push_message("system", f"No models found for query: {query or '(all)'}")
                self._set_status("No models")
                return
            source_meta = payload.get("source_meta", {})
            if not isinstance(source_meta, dict):
                source_meta = {}
            lines = [
                (
                    f"Models ({len(models)} shown, query='{query or '*'}', mode={'full' if full else 'configured'}). "
                    f"default_small={payload.get('default_small_model')} "
                    f"default_large={payload.get('default_large_model')}"
                )
            ]
            if full:
                lines.append(
                    "source: "
                    f"seeded={source_meta.get('seeded_count', 0)} "
                    f"openai_remote={source_meta.get('openai_remote_count', 0)}"
                )
                remote_error = str(source_meta.get("openai_remote_error", "")).strip()
                if remote_error:
                    lines.append(f"openai_remote_error: {remote_error}")
            for name in models[:40]:
                lines.append(f"- {name}")
            if len(models) > 40:
                lines.append(f"... and {len(models) - 40} more")
            self._push_message("system", "\n".join(lines))
            self._set_status("Models loaded")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"models query failed: {exc}")
            self._set_status("Error")

    def _command_stream_mode(self, value: str) -> None:
        clean = value.strip().lower()
        if clean in {"on", "1", "true"}:
            self.stream_mode = True
            self._push_message("system", "stream mode: on")
            return
        if clean in {"off", "0", "false"}:
            self.stream_mode = False
            self._push_message("system", "stream mode: off")
            return
        self._push_message("system", "Usage: /stream <on|off>")

    def _command_agent_info(self, agent_id: str) -> None:
        target = agent_id.strip()
        if not target:
            self._push_message("system", "No agent id. Run a turn first or use /agentinfo <agent_id>.")
            return
        self._set_status("Loading agent info...")
        self._render()
        try:
            payload = self.api.get_agent(target, event_limit=500)
            agent = payload.get("agent", {})
            if not isinstance(agent, dict):
                agent = {}
            events = payload.get("events", [])
            if not isinstance(events, list):
                events = []
            metrics = self._summarize_events(events)
            lines = [
                f"Agent {agent.get('id', target)} status={agent.get('status', 'unknown')} model={agent.get('model', '')}",
                (
                    f"runshell={metrics['runshell_count']} spawn={metrics['spawn_count']} ask={metrics['ask_count']} note={metrics.get('note_count', 0)} "
                    f"tool_define={metrics.get('tool_define_count', 0)} tool_call={metrics.get('tool_call_count', 0)} "
                    f"actions={metrics['model_actions']} total_tokens={metrics['total_tokens']} "
                    f"est_cost_usd={metrics['estimated_cost_usd']:.6f}"
                ),
            ]
            for item in metrics["last_runshell"]:
                lines.append(
                    (
                        f"- runshell progress={item.get('progress', '')} "
                        f"exit={item.get('exit_code')} allowed={item.get('allowed')} cmd={item.get('command')}"
                    )
                )
            for item in metrics.get("last_tool_calls", []):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    (
                        f"- tool_call name={item.get('tool_name', '')} progress={item.get('progress', '')} "
                        f"exit={item.get('exit_code')} allowed={item.get('allowed')} cmd={item.get('command', '')}"
                    )
                )
            lines.extend(self._summarize_spawn_details(events))
            self._push_message("system", "\n".join(lines))
            self._set_status("Agent info loaded")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"agentinfo failed: {exc}")
            self._set_status("Error")

    def _summarize_spawn_details(self, events: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if str(event.get("event_type", "")) != "spawn":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            child = str(payload.get("child_agent_id", "")).strip()
            status = str(payload.get("child_status", "")).strip()
            progress = str(payload.get("progress", "")).strip()
            child_summary = str(payload.get("child_command_summary", "")).strip()
            lines.append(f"- spawn child={child[:12]} status={status} progress={progress}")
            if child_summary:
                for line in child_summary.splitlines()[:6]:
                    clean = line.strip()
                    if clean:
                        lines.append(f"    {clean}")
        return lines

    def _command_show_pending_ask(self) -> None:
        if self.pending_ask is None:
            self._push_message("system", "No pending ask.")
            return
        ask_text = self._render_pending_ask_text(self.pending_ask)
        self._push_message("assistant_ask", ask_text)
        self._open_popup("Clarification Needed", ask_text.splitlines())

    def _execute_turn(self, prompt: str, *, user_visible: str) -> None:
        self._push_message("user", user_visible)
        if not self.session_id:
            self._push_message("system", "No session selected. Use /new.")
            return
        self._set_status("Thinking...")
        self._render()
        try:
            if self.stream_mode:
                payload = self._run_turn_stream(prompt)
            else:
                payload = self.api.turn(self.session_id, prompt, model=self.current_model)
                output = str(payload.get("output", "")).strip()
                self._push_message("assistant", output or "(empty)")
                self.last_agent_id = str(payload.get("agent_id", "")).strip()
                self._emit_agent_metrics(payload.get("agent_metrics", {}))
                self._set_pending_ask(payload.get("ask"))
                steps = payload.get("steps", [])
                if isinstance(steps, list) and steps:
                    self._set_status(" -> ".join(str(item) for item in steps[:4]))
                else:
                    self._set_status("Done")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"Turn failed: {exc}")
            self._set_status("Error")

    def _handle_pending_ask_input(self, raw: str) -> None:
        if self.pending_ask is None:
            self._push_message("system", "No pending ask. Use normal prompt.")
            return
        try:
            user_line, followup_prompt = self._build_ask_followup(raw, self.pending_ask)
        except ValueError as exc:
            self._push_message("system", str(exc))
            self._push_message("assistant_ask", self._render_pending_ask_text(self.pending_ask))
            return
        self.pending_ask = None
        self._close_popup()
        self._execute_turn(followup_prompt, user_visible=user_line)

    def _set_pending_ask(self, ask_payload: Any) -> None:
        if not isinstance(ask_payload, dict):
            return
        question = str(ask_payload.get("question", "")).strip()
        if not question:
            return
        self.pending_ask = ask_payload
        ask_text = self._render_pending_ask_text(ask_payload)
        self._push_message("assistant_ask", ask_text)
        self._open_popup("Clarification Needed", ask_text.splitlines())
        self._set_status("Awaiting clarification")

    def _render_pending_ask_text(self, ask_payload: dict[str, Any]) -> str:
        ask_id = str(ask_payload.get("ask_id", "")).strip() or "ask"
        question = str(ask_payload.get("question", "")).strip()
        mode = str(ask_payload.get("input_mode", "text")).strip() or "text"
        lines = [f"[ask:{ask_id}] {question}", f"mode={mode}"]
        options = ask_payload.get("options", [])
        if isinstance(options, list) and options:
            lines.append("choices:")
            for idx, item in enumerate(options[:8], start=1):
                if not isinstance(item, dict):
                    continue
                option_id = str(item.get("id", "")).strip() or str(idx)
                label = str(item.get("label", "")).strip()
                hint = str(item.get("hint", "")).strip()
                line = f"{idx}. {option_id} - {label}"
                if hint:
                    line += f" ({hint})"
                lines.append(line)
        allow_free = bool(ask_payload.get("allow_free_text", False))
        max_select = int(ask_payload.get("max_select", 1) or 1)
        lines.append(f"allow_free_text={allow_free} max_select={max_select}")
        lines.append("reply with plain text, or /answer ...")
        return "\n".join(lines)

    def _build_ask_followup(self, raw_input: str, ask_payload: dict[str, Any]) -> tuple[str, str]:
        text = raw_input.strip()
        if not text:
            raise ValueError("Ask answer cannot be empty.")
        ask_id = str(ask_payload.get("ask_id", "")).strip() or "ask"
        question = str(ask_payload.get("question", "")).strip()
        mode = str(ask_payload.get("input_mode", "text")).strip().lower() or "text"
        allow_free = bool(ask_payload.get("allow_free_text", False))
        max_select = int(ask_payload.get("max_select", 1) or 1)
        if max_select < 1:
            max_select = 1

        options_raw = ask_payload.get("options", [])
        options: list[dict[str, str]] = []
        if isinstance(options_raw, list):
            for item in options_raw[:8]:
                if not isinstance(item, dict):
                    continue
                option_id = str(item.get("id", "")).strip()
                label = str(item.get("label", "")).strip()
                if option_id and label:
                    options.append({"id": option_id, "label": label})
        if mode in {"single", "multiple"} and not options:
            mode = "text"

        if mode == "text":
            user_line = f"[answer:{ask_id}] {text}"
            prompt = (
                f"[ASK_RESPONSE]\nask_id: {ask_id}\nquestion: {question}\nmode: text\n"
                f"text: {text}\n\nContinue the task using this user clarification."
            )
            return user_line, prompt

        selection_part = text
        free_text = ""
        if "|" in text:
            selection_part, free_text = text.split("|", 1)
            selection_part = selection_part.strip()
            free_text = free_text.strip()

        selected = self._parse_ask_selections(selection_part, options)
        if mode == "single" and len(selected) != 1:
            raise ValueError("Single-choice ask needs exactly one option (index/id).")
        if mode == "multiple" and len(selected) > max_select:
            raise ValueError(f"Too many choices. max_select={max_select}.")

        if not selected and allow_free:
            free_text = free_text or text
        if not selected and not free_text:
            raise ValueError("Please select option ids/indexes, or provide text if allow_free_text=true.")

        selected_lines = "\n".join([f"- {item['id']}: {item['label']}" for item in selected]) or "- (none)"
        user_display = ", ".join([item["id"] for item in selected]) if selected else (free_text or text)
        user_line = f"[answer:{ask_id}] {user_display}"
        prompt = (
            f"[ASK_RESPONSE]\nask_id: {ask_id}\nquestion: {question}\nmode: {mode}\n"
            f"selected_options:\n{selected_lines}\nfree_text: {free_text}\n\n"
            "Continue the task using this answer. Do not ask the same clarification again unless still blocked."
        )
        return user_line, prompt

    @staticmethod
    def _parse_ask_selections(raw: str, options: list[dict[str, str]]) -> list[dict[str, str]]:
        tokens = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
        if not tokens:
            return []
        by_index = {str(idx): item for idx, item in enumerate(options, start=1)}
        by_id = {item["id"].lower(): item for item in options}
        selected: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for token in tokens:
            hit = by_index.get(token) or by_id.get(token.lower())
            if hit is None:
                continue
            option_id = hit["id"]
            if option_id in seen_ids:
                continue
            seen_ids.add(option_id)
            selected.append(hit)
        return selected

    def _run_turn_stream(self, prompt: str) -> dict[str, Any]:
        if not self.session_id:
            raise ValueError("session is required")
        final_payload: dict[str, Any] = {}
        stream_err: Exception | None = None
        stream_done = False
        stop_requested = False
        self._set_status("Streaming...")
        self._render()
        self.stream_events_seen = 0
        self._last_progress_signature = ""
        self.stream_inflight = True
        frame_q: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _reader() -> None:
            try:
                for frame in self.api.turn_stream(self.session_id, prompt, model=self.current_model):
                    frame_q.put(("frame", frame))
            except Exception as exc:  # noqa: BLE001
                frame_q.put(("error", exc))
            finally:
                frame_q.put(("done", None))

        worker = threading.Thread(target=_reader, daemon=True)
        worker.start()
        stdscr = self._stdscr
        if stdscr is not None:
            stdscr.timeout(100)

        try:
            while True:
                processed = False
                while True:
                    try:
                        item_type, payload = frame_q.get_nowait()
                    except queue.Empty:
                        break
                    processed = True
                    if item_type == "error":
                        if isinstance(payload, Exception):
                            stream_err = payload
                        else:
                            stream_err = RuntimeError(str(payload))
                        continue
                    if item_type == "done":
                        stream_done = True
                        continue
                    frame = payload if isinstance(payload, dict) else {}
                    self.stream_events_seen += 1
                    event = str(frame.get("event", "")).strip().lower()
                    data = frame.get("data", {})
                    if not isinstance(data, dict):
                        data = {}
                    if event == "started":
                        self.last_agent_id = str(data.get("agent_id", "")).strip()
                        self._push_message("assistant_progress", f"[init] agent started: {self.last_agent_id[:12]}")
                        self._set_status(f"agent={self.last_agent_id[:12]} streaming")
                    elif event == "action":
                        action = str(data.get("action", "")).strip()
                        step = data.get("step")
                        progress = str(data.get("progress", "")).strip()
                        detail = str(data.get("detail", "")).strip()
                        next_hint = str(data.get("next", "")).strip()
                        question = str(data.get("question", "")).strip()
                        cost = float(data.get("estimated_cost_usd", 0.0) or 0.0)
                        mid = f"[step {step}] action={action}"
                        if progress:
                            mid += f" | progress: {progress}"
                        if detail:
                            mid += f" | detail: {detail}"
                        if next_hint:
                            mid += f" | next: {next_hint}"
                        if action == "ask" and question:
                            mid += f" | question: {question}"
                        sign = f"{step}|{action}|{progress}|{detail}|{next_hint}|{question}"
                        if sign != self._last_progress_signature:
                            self._last_progress_signature = sign
                            self._push_message("assistant_progress", mid)
                        short = f"progress={progress}" if progress else ""
                        self._set_status(f"step={step} action={action} {short} est_cost=${cost:.6f}")
                    elif event == "runshell":
                        cmd = str(data.get("command", "")).strip()
                        exit_code = data.get("exit_code")
                        progress = str(data.get("progress", "")).strip()
                        desc = f"progress={progress}".strip()
                        self._push_message("assistant_progress", f"[runshell] {desc} | exit={exit_code} | cmd={cmd}")
                        for line in self._preview_lines_from_event(data.get("stdout_preview")):
                            self._push_message("assistant_progress", f"    [out] {line}")
                        for line in self._preview_lines_from_event(data.get("stderr_preview")):
                            self._push_message("assistant_progress", f"    [err] {line}")
                    elif event == "spawn":
                        child = str(data.get("child_agent_id", "")).strip()
                        status = str(data.get("child_status", "")).strip()
                        progress = str(data.get("progress", "")).strip()
                        child_summary = str(data.get("child_command_summary", "")).strip()
                        self._push_message(
                            "assistant_progress",
                            f"[spawn] progress={progress} child={child[:12]} status={status}",
                        )
                        if child_summary:
                            for line in child_summary.splitlines():
                                clean = line.strip()
                                if clean:
                                    self._push_message("assistant_progress", f"    [child-runshell] {clean}")
                    elif event == "ask":
                        self._set_pending_ask(data)
                    elif event == "note":
                        progress = str(data.get("progress", "")).strip()
                        detail = str(data.get("detail", "")).strip()
                        next_hint = str(data.get("next", "")).strip()
                        text = f"[note] progress={progress}"
                        if detail:
                            text += f" | detail={detail}"
                        if next_hint:
                            text += f" | next={next_hint}"
                        self._push_message("assistant_progress", text)
                    elif event == "tool_define":
                        tool_name = str(data.get("tool_name", "")).strip()
                        language = str(data.get("language", "")).strip()
                        progress = str(data.get("progress", "")).strip()
                        self._push_message(
                            "assistant_progress",
                            f"[tool-define] name={tool_name} lang={language} progress={progress}",
                        )
                    elif event == "tool_call":
                        tool_name = str(data.get("tool_name", "")).strip()
                        command = str(data.get("command", "")).strip()
                        exit_code = data.get("exit_code")
                        allowed = bool(data.get("allowed", True))
                        progress = str(data.get("progress", "")).strip()
                        self._push_message(
                            "assistant_progress",
                            f"[tool-call] tool={tool_name} progress={progress} exit={exit_code} allowed={allowed}",
                        )
                        if command:
                            self._push_message("assistant_progress", f"    [tool-cmd] {command}")
                        for line in self._preview_lines_from_event(data.get("stdout_preview")):
                            self._push_message("assistant_progress", f"    [tool-out] {line}")
                        for line in self._preview_lines_from_event(data.get("stderr_preview")):
                            self._push_message("assistant_progress", f"    [tool-err] {line}")
                    elif event == "final":
                        final_payload = data
                        output = str(data.get("output", "")).strip()
                        self._push_message("assistant", output or "(empty)")
                        self._emit_agent_metrics(data.get("agent_metrics", {}))
                        self._set_pending_ask(data.get("ask"))
                        self._set_status(str(data.get("status", "done")))
                    self._render()

                if stream_err is not None:
                    raise RuntimeError(f"stream failed: {stream_err}")

                if stdscr is not None:
                    try:
                        key = stdscr.get_wch()
                    except curses.error:
                        key = None
                    if key in (27, "\x1b") and not stop_requested:
                        stop_requested = True
                        if self.last_agent_id:
                            try:
                                self.api.cancel_agent(self.last_agent_id)
                                self._push_message("system", f"Stop requested for agent {self.last_agent_id[:12]}.")
                                self._open_popup(
                                    "Stopping Response",
                                    [
                                        f"Agent: {self.last_agent_id[:12]}",
                                        "Cancellation requested.",
                                        "Waiting for final event...",
                                    ],
                                )
                                self._set_status("Stopping response...")
                            except Exception as exc:  # noqa: BLE001
                                self._push_message("system", f"Stop failed: {exc}")
                                self._set_status("Stop failed")
                        else:
                            self._push_message("system", "Cannot stop yet: agent id not available.")

                if stream_done and frame_q.empty():
                    break
                if not processed:
                    time.sleep(0.03)
        finally:
            self.stream_inflight = False
            if stdscr is not None:
                stdscr.timeout(-1)

        if not final_payload:
            raise RuntimeError("stream ended without final payload")
        return final_payload

    def _emit_agent_metrics(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        runshell = int(payload.get("runshell_count", 0) or 0)
        spawn = int(payload.get("spawn_count", 0) or 0)
        ask = int(payload.get("ask_count", 0) or 0)
        note = int(payload.get("note_count", 0) or 0)
        tool_define = int(payload.get("tool_define_count", 0) or 0)
        tool_call = int(payload.get("tool_call_count", 0) or 0)
        actions = int(payload.get("model_actions", 0) or 0)
        total_tokens = int(payload.get("total_tokens", 0) or 0)
        est_cost = float(payload.get("estimated_cost_usd", 0.0) or 0.0)
        lines = [
            (
                f"[agent] actions={actions} runshell={runshell} spawn={spawn} "
                f"ask={ask} note={note} tool_define={tool_define} tool_call={tool_call} "
                f"total_tokens={total_tokens} est_cost_usd={est_cost:.6f}"
            )
        ]
        recent = payload.get("last_runshell", [])
        if isinstance(recent, list):
            for item in recent[:3]:
                if not isinstance(item, dict):
                    continue
                cmd = str(item.get("command", "")).strip()
                lines.append(
                    (
                        f"[agent] runshell progress={item.get('progress', '')} "
                        f"exit={item.get('exit_code')} allowed={item.get('allowed')} cmd={cmd}"
                    )
                )
        recent_tools = payload.get("last_tool_calls", [])
        if isinstance(recent_tools, list):
            for item in recent_tools[:3]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    (
                        f"[agent] tool_call name={item.get('tool_name', '')} "
                        f"progress={item.get('progress', '')} "
                        f"exit={item.get('exit_code')} allowed={item.get('allowed')}"
                    )
                )
        self._push_message("system", "\n".join(lines))

    def _summarize_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        model_actions = 0
        runshell_count = 0
        spawn_count = 0
        ask_count = 0
        note_count = 0
        tool_define_count = 0
        tool_call_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        estimated_cost_usd = 0.0
        last_runshell: list[dict[str, Any]] = []
        last_tool_calls: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            et = event.get("event_type")
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            if et == "model_action":
                model_actions += 1
                usage = payload.get("usage", {})
                if isinstance(usage, dict):
                    prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                estimated_cost_usd += float(payload.get("estimated_cost_usd", 0.0) or 0.0)
            elif et == "runshell":
                runshell_count += 1
                last_runshell.append(
                    {
                        "command": str(payload.get("command", "")),
                        "exit_code": payload.get("exit_code"),
                        "allowed": bool(payload.get("allowed", True)),
                        "progress": str(payload.get("progress", "")),
                    }
                )
            elif et == "spawn":
                spawn_count += 1
            elif et == "ask":
                ask_count += 1
            elif et == "note":
                note_count += 1
            elif et == "tool_define":
                tool_define_count += 1
            elif et == "tool_call":
                tool_call_count += 1
                last_tool_calls.append(
                    {
                        "tool_name": str(payload.get("tool_name", "")),
                        "command": str(payload.get("command", "")),
                        "exit_code": payload.get("exit_code"),
                        "allowed": bool(payload.get("allowed", True)),
                        "progress": str(payload.get("progress", "")),
                    }
                )
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "model_actions": model_actions,
            "runshell_count": runshell_count,
            "spawn_count": spawn_count,
            "ask_count": ask_count,
            "note_count": note_count,
            "tool_define_count": tool_define_count,
            "tool_call_count": tool_call_count,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 8),
            "last_runshell": last_runshell[-5:],
            "last_tool_calls": last_tool_calls[-5:],
        }

    @staticmethod
    def _preview_lines_from_event(raw: Any) -> list[str]:
        if isinstance(raw, list):
            out: list[str] = []
            for item in raw[:3]:
                line = str(item).strip()
                if line:
                    out.append(line[:220])
            return out
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            return [line.strip()[:220] for line in text.splitlines() if line.strip()][:3]
        return []

    def _command_model(self, raw_model: str) -> None:
        model = raw_model.strip()
        if not model:
            self._push_message("system", "Usage: /model <name|auto>")
            return
        self.current_model = model
        self._push_message("system", f"Model set to: {self.current_model}")
        self._set_status("Model updated")

    def _command_auth_poe(self, api_key: str) -> None:
        key = api_key.strip()
        if not key:
            self._push_message("system", "Usage: /authpoe <api_key>")
            return
        self._set_status("Updating Poe key...")
        self._render()
        try:
            self.api.auth_poe(key)
            self._push_message("system", "Poe API key set.")
            self._set_status("Auth updated")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"authpoe failed: {exc}")
            self._set_status("Error")

    def _command_auth_openai(self, api_key: str) -> None:
        key = api_key.strip()
        if not key:
            self._push_message("system", "Usage: /authopenai <api_key>")
            return
        self._set_status("Updating OpenAI key...")
        self._render()
        try:
            self.api.auth_openai(key)
            self._push_message("system", "OpenAI API key set.")
            self._set_status("Auth updated")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"authopenai failed: {exc}")
            self._set_status("Error")

    def _command_secrets_save(self, user_key: str) -> None:
        key = user_key.strip()
        if not key:
            self._push_message("system", "Usage: /secretssave <user_key>")
            return
        self._set_status("Saving secrets...")
        self._render()
        try:
            payload = self.api.save_secrets(key)
            self._push_message("system", f"Secrets saved: {payload.get('path', '')}")
            self._set_status("Secrets saved")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"secretssave failed: {exc}")
            self._set_status("Error")

    def _command_secrets_load(self, user_key: str) -> None:
        key = user_key.strip()
        if not key:
            self._push_message("system", "Usage: /secretsload <user_key>")
            return
        self._set_status("Loading secrets...")
        self._render()
        try:
            payload = self.api.load_secrets(key)
            msg = (
                "Secrets loaded. "
                f"poe={payload.get('poe_api_key_set')} "
                f"openai={payload.get('openai_api_key_set')}"
            )
            self._push_message("system", msg)
            self._set_status("Secrets loaded")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"secretsload failed: {exc}")
            self._set_status("Error")

    def _command_arxiv(self, query: str) -> None:
        if not query:
            self._push_message("system", "Usage: /arxiv <query>")
            return
        if not self.session_id:
            self._push_message("system", "No session selected. Use /new.")
            return
        self._push_message("user", f"/arxiv {query}")
        self._set_status("Running arxiv workflow...")
        self._render()
        try:
            payload = self.api.arxiv(self.session_id, query)
            final = str(payload.get("final_output", "")).strip()
            self._push_message("assistant", final or "(empty)")
            self._set_status("Workflow done")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"Workflow failed: {exc}")
            self._set_status("Error")

    def _reload_messages(self) -> None:
        if not self.session_id:
            return
        self._set_status("Reloading messages...")
        self._render()
        try:
            rows = self.api.list_messages(self.session_id, limit=120)
            self.messages = []
            for item in rows:
                role = str(item.get("role", "assistant"))
                content = str(item.get("content", ""))
                self._push_message(role, content)
            self._set_status("Reloaded")
        except Exception as exc:  # noqa: BLE001
            self._push_message("system", f"Reload failed: {exc}")
            self._set_status("Error")

    def _render(self) -> None:
        assert self._stdscr is not None
        stdscr = self._stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 10 or w < 50:
            self._draw_line(0, "Terminal too small. Resize to at least 40x8.", curses.A_BOLD)
            stdscr.refresh()
            return

        sid_short = self.session_id[:12] if self.session_id else "-"
        header = f"PoeCoder CUI {self.version} | session={sid_short} | model={self.current_model}"
        header_attr = curses.color_pair(1) | curses.A_BOLD if self._colors_enabled else curses.A_REVERSE
        self._draw_line(0, header, header_attr)
        sub = (
            f"stream={'on' if self.stream_mode else 'off'} | last_agent={self.last_agent_id[:12] or '-'} "
            f"| events={self.stream_events_seen} | pending_ask={'yes' if self.pending_ask else 'no'} | F2 new | F5 reload | /help"
        )
        self._draw_line(1, sub, curses.A_DIM)

        body_top = 2
        body_bottom = h - 3
        content_height = max(1, body_bottom - body_top + 1)
        sidebar_width = 0
        if w >= 110:
            sidebar_width = 34
        msg_width = w - 2 - sidebar_width
        lines = self._format_message_lines(max_width=max(20, msg_width))
        max_scroll = max(0, len(lines) - content_height)
        self.scroll = min(self.scroll, max_scroll)
        start = max(0, len(lines) - content_height - self.scroll)
        visible = lines[start : start + content_height]
        y = body_top
        for line, role in visible:
            self._draw_line(y, line, self._role_attr(role))
            y += 1

        if sidebar_width > 0:
            split_x = w - sidebar_width - 1
            for yy in range(body_top, body_bottom + 1):
                self._draw_line_at(yy, split_x, "|", curses.A_DIM)
            side_lines = self._sidebar_lines(max_lines=content_height, width=sidebar_width - 2)
            title_attr = curses.color_pair(6) | curses.A_BOLD if self._colors_enabled else curses.A_BOLD
            self._draw_line_at(body_top, split_x + 2, "Live Panel", title_attr)
            sy = body_top + 1
            for text, attr in side_lines[: max(0, content_height - 1)]:
                self._draw_line_at(sy, split_x + 2, text, attr)
                sy += 1

        self._draw_line(h - 2, self.status[: max(1, w - 1)], curses.A_DIM)
        prompt = f"> {self.input_buffer}"
        self._draw_line(h - 1, prompt[: max(1, w - 1)], curses.A_BOLD)
        try:
            stdscr.move(h - 1, min(len(prompt), w - 2))
        except curses.error:
            pass
        if self.popup_visible:
            try:
                stdscr.noutrefresh()
            except curses.error:
                pass
            self._draw_popup()
            try:
                curses.doupdate()
            except curses.error:
                pass
            return
        stdscr.refresh()

    def _format_message_lines(self, max_width: int) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = []
        role_map = {"user": "U", "assistant": "A", "assistant_ask": "Q", "system": "S"}
        body_width = max(10, max_width - 5)
        for msg in self.messages:
            msg_role = str(msg.get("role", "assistant"))
            role = role_map.get(msg_role, "P" if msg_role == "assistant_progress" else "?")
            content = (msg.get("content", "") or "").replace("\r\n", "\n").replace("\r", "\n")
            parts = content.split("\n") if content else [""]
            first = True
            render_role = msg_role
            if msg_role == "system" and ("failed" in content.lower() or "error" in content.lower()):
                render_role = "system_error"
            for part in parts:
                wrapped = textwrap.wrap(part, width=body_width) or [""]
                for chunk in wrapped:
                    prefix = f"[{role}] " if first else "    "
                    lines.append((prefix + chunk, render_role))
                    first = False
        if not lines:
            lines = [("[S] No messages yet.", "system")]
        return lines

    def _sidebar_lines(self, *, max_lines: int, width: int) -> list[tuple[str, int]]:
        base_attr = curses.A_DIM
        strong_attr = curses.A_BOLD
        if self._colors_enabled:
            base_attr = curses.color_pair(2)
            strong_attr = curses.color_pair(3) | curses.A_BOLD
        out: list[tuple[str, int]] = []
        push = lambda text, attr=base_attr: out.append((text[: max(1, width)], attr))
        push(f"session: {self.session_id[:18] or '-'}")
        push(f"model: {self.current_model}")
        push(f"stream: {'on' if self.stream_mode else 'off'}")
        push(f"last agent: {self.last_agent_id[:18] or '-'}")
        push(f"events: {self.stream_events_seen}")
        if self.pending_ask:
            ask_mode = str(self.pending_ask.get("input_mode", "text"))
            push(f"pending ask: {ask_mode}", strong_attr)
        else:
            push("pending ask: none")
        push("")
        push("Quick Cmds", strong_attr)
        push("/models full gpt")
        push("/model openai/gpt-5")
        push("/stream on|off")
        push("/agentinfo")
        push("/ask")
        push("/answer ...")
        push("/secretssave <key>")
        push("/secretsload <key>")
        if len(out) < max_lines:
            push("")
            push("Tip", strong_attr)
            push("Progress lines [P] are")
            push("LLM middle feedback.")
            push("Esc stops streaming")
        return out

    def _draw_line(self, y: int, text: str, attr: int = 0) -> None:
        assert self._stdscr is not None
        h, w = self._stdscr.getmaxyx()
        if y < 0 or y >= h:
            return
        out = text[: max(1, w - 1)]
        try:
            self._stdscr.addnstr(y, 0, out, max(1, w - 1), attr)
        except curses.error:
            return

    def _draw_line_at(self, y: int, x: int, text: str, attr: int = 0) -> None:
        assert self._stdscr is not None
        h, w = self._stdscr.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        out = text[: max(1, w - x - 1)]
        try:
            self._stdscr.addnstr(y, x, out, max(1, w - x - 1), attr)
        except curses.error:
            return

    def _draw_popup(self) -> None:
        assert self._stdscr is not None
        if not self.popup_visible:
            return
        h, w = self._stdscr.getmaxyx()
        if h < 8 or w < 40:
            return
        lines = self.popup_lines or ["(empty)"]
        max_line_width = max(len(line) for line in lines)
        popup_w = min(max(42, max_line_width + 6), max(20, w - 4))
        popup_h = min(max(7, len(lines) + 4), max(6, h - 4))
        y = max(1, (h - popup_h) // 2)
        x = max(2, (w - popup_w) // 2)
        try:
            win = curses.newwin(popup_h, popup_w, y, x)
            win.box()
            title = f" {self.popup_title[: max(1, popup_w - 6)]} "
            title_attr = curses.A_BOLD | (curses.color_pair(6) if self._colors_enabled else 0)
            win.addnstr(0, 2, title, max(1, popup_w - 4), title_attr)
            body_h = popup_h - 3
            for idx, line in enumerate(lines[:body_h], start=1):
                win.addnstr(idx, 2, line, max(1, popup_w - 4), curses.A_NORMAL)
            if popup_h >= 3:
                footer = "Esc close"
                win.addnstr(popup_h - 2, max(2, popup_w - len(footer) - 2), footer, len(footer), curses.A_DIM)
            win.noutrefresh()
        except curses.error:
            return


def run_simple(base_url: str) -> None:
    with httpx.Client(timeout=300) as client:
        api = base_url.rstrip("/") + "/agent-api"
        about = client.get(f"{api}/about")
        about.raise_for_status()
        meta = about.json()
        print(f"AgentCore {meta.get('version_tag')} - execution={meta.get('execution_primitive')}")

    api_client = ApiClient(base_url)
    current_model = "auto"
    try:
        session = api_client.create_session("simple-session")
        session_id = session["id"]
        print(f"session: {session_id}")
        print(
            "Type prompt. /exit to quit. /model <name>. /models [query]. /models full [query]. "
            "/arxiv <query> for workflow. If agent asks clarification, answer in next prompt."
        )
        while True:
            prompt = input("> ").strip()
            if not prompt:
                continue
            if prompt in {"/exit", "exit", "quit"}:
                print("bye")
                return
            if prompt == "/model":
                print(f"current model: {current_model}")
                continue
            if prompt.startswith("/model "):
                current_model = prompt.split(" ", 1)[1].strip() or "auto"
                print(f"model set: {current_model}")
                continue
            if prompt == "/models" or prompt.startswith("/models "):
                query = prompt.split(" ", 1)[1].strip() if " " in prompt else ""
                full = False
                if query == "full":
                    full = True
                    query = ""
                elif query.startswith("full "):
                    full = True
                    query = query.split(" ", 1)[1].strip()
                payload = api_client.list_models(query=query, limit=80 if full else 30, full=full)
                print(f"models ({payload.get('count', 0)} total):")
                for name in payload.get("models", [])[:30]:
                    print("-", name)
                continue
            if prompt.startswith("/arxiv "):
                query = prompt[len("/arxiv ") :].strip()
                payload = api_client.arxiv(session_id, query)
                print(textwrap.shorten(str(payload.get("final_output", "")), width=160, placeholder="..."))
                continue
            if prompt.startswith("/"):
                print(f"Unknown command: {prompt}.")
                continue
            payload = api_client.turn(session_id, prompt, model=current_model)
            print(payload.get("output", ""))
            ask_payload = payload.get("ask")
            if isinstance(ask_payload, dict) and str(ask_payload.get("question", "")).strip():
                print("[ask]", ask_payload.get("question", ""))
                options = ask_payload.get("options", [])
                if isinstance(options, list):
                    for idx, item in enumerate(options[:8], start=1):
                        if isinstance(item, dict):
                            print(f"  {idx}. {item.get('id')} - {item.get('label')}")
            metrics = payload.get("agent_metrics", {})
            if isinstance(metrics, dict):
                print(
                    "[agent]",
                    f"actions={metrics.get('model_actions', 0)}",
                    f"runshell={metrics.get('runshell_count', 0)}",
                    f"tokens={metrics.get('total_tokens', 0)}",
                    f"est_cost_usd={metrics.get('estimated_cost_usd', 0.0)}",
                )
    finally:
        api_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PoeCoder AgentCore CLI/CUI")
    parser.add_argument("--api", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", choices=["cui", "simple"], default="cui")
    args = parser.parse_args()

    if args.mode == "simple":
        run_simple(args.api)
        return

    api_client = ApiClient(args.api)
    try:
        curses.wrapper(lambda stdscr: CUIApp(api_client).run(stdscr))
    except KeyboardInterrupt:
        pass
    except Exception:
        # Fallback for terminals that do not support curses interaction.
        run_simple(args.api)
    finally:
        api_client.close()


if __name__ == "__main__":
    main()
