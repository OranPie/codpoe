# PoeCoder

PoeCoder is a Python coding assistant runtime with:
- A FastAPI backend that manages sessions, memory, wiki, tools, and subagents.
- Poe model calls via `fastapi-poe` (`https://creator.poe.com/api-reference/overview`).
- Optional OpenAI model calls via official `openai` Python SDK.
- A CLI with Codex-like turn loop and streaming output.
- Dual model path: backend-proxy model calls or direct model calls from CLI.

## Release advisory

- `v0.1.1` is now **unsuggested** due to known performance issues, higher token/cost overhead, and stability problems.
- Use `v0.1.2+` for improved turn stability, context efficiency, and tool-response handling.
- Docs index: `docs/README.md`

## Quick start

1. Install dependencies:
   - `python -m pip install -e .`
2. Run API:
   - `poecoder-api`
3. Run CLI:
   - `poecoder`

## Notes

- Storage uses SQLite at `~/.poecoder/poecoder.db` by default.
- Shell execution is policy-gated by danger level.
- Context is mode-based: coding clears per turn, chat/planning/leader keeps a short window.


## Notable CLI Commands

- `/listmodels [query]` to inspect supported models with optional substring filter.
- `/modeltable` to inspect/edit model strategy profiles used for auto model choice.
- `/changemodel <name|auto>` to switch main model during a session.
- `/login [api_key]` to set/update Poe API key (prompts securely when omitted).
- `/loginopenai [api_key]` to set/update OpenAI API key.
- `/sessions [limit]` and `/resume <session_id|index>` to list and resume backend sessions.
- OpenAI models should be selected with `openai/<model>` prefix (avoids name collisions with Poe model names).
- `/plan` to switch to planning mode with planning-focused system message.
- `/thinking <quick|balanced|deep> [budget]` to control model reasoning depth/token budget hints.
- `/thinkdetails <on|off>` to control whether model progress/thinking detail text is encouraged in outputs.
- `/commandpolicy <allow|deny> [encourage|noencourage]` to control model self-command creation autonomy.
- `/image <path|url>`, `/images`, `/clearimages` for image attachments on the next request.
- `/review <prompt>` to run reviewer-role analysis (also exposed as tool `Review`).

- Web tools: `GetWebRaw`, `GetWeb`, and `GetWebFile` are exposed via `/tools/invoke` and dedicated API routes; `GetWebRaw/GetWeb` support selector/regex filtering to cut token cost, and `GetWeb` can auto-download large pages for local reads.

- CLI supports `/balance` to read current Poe point balance.
- CLI output uses live streamed events (`status`, `tool`, `delta`) for better readability.
- Provider base URIs are configurable:
  - `POECODER_POE_API_URL` (default `https://api.poe.com/bot/`)
  - `POECODER_OPENAI_API_URL` (default `https://api.openai.com/v1`)
  - `POECODER_OPENAI_MODELS` accepts comma-separated OpenAI model names and auto-normalizes to `openai/<model>`.
  - `POECODER_SHOW_THINK_DETAILS` (`true`/`false`) sets default think-details output mode for new CLI sessions.
  - Runtime update endpoints: `POST /providers/poe/base-url`, `POST /providers/openai/base-url`.

- Base main/subagent system prompts live in `poecoder/prompts.py`.
- `StartSubAgent` supports `system_message_modifier` so the main model can shape subagent behavior safely.

- CLI i18n: set `POECODER_LANG=zh-cn` or run `poecoder --lang zh-cn` (you can also switch live with `/lang zh-cn`).

- Background tasks: `/bgturn`, `/bgsubagent`, `/tasks`, `/task`, `/canceltask`.
- Task output shortcuts: `/readtaskoutput <task_id>` and API `GET /tasks/{task_id}/output`.
- Model tools now support async task orchestration: `StartBackgroundTurn`, `StartBackgroundSubAgent`, `ReadTaskOutput`.
- Leader orchestration mode: `/mode leader`, `/leader`, `/leaderstatus`, `/leaderjobs`, `/leaderwait`, `/leadercancel`.
- Leader API endpoints: `/leader/start`, `/leader/{run_id}`, `/leader/{run_id}/jobs`, `/leader/{run_id}/wait`, `/leader/{run_id}/cancel`.
- Leader mode enforces scoped parallel jobs with explicit ownership and non-interference guidance per subtask.
- `GET /tools/catalog` exposes the command/tool reference so the model can follow exact command names and args.
- Tool `Help` provides per-tool usage guidance (`Help(tool_name)`) for complex commands.
- Context selection is relevance-ranked + compacted by default to reduce token waste while keeping important session context.
- Session titles are auto-derived from model conclusions after successful turns, then shown in session listings for quick resume.
- Turn streaming now uses model chunk streaming for lower latency (`delta` events are emitted as chunks arrive).
- GitHub Actions: CI runs tests on pushes/PRs to `main`; release workflow builds/tests and publishes GitHub Releases on `v*` tags.
