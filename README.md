# PoeCoder

PoeCoder is a Python coding assistant runtime with:
- A FastAPI backend that manages sessions, memory, wiki, tools, and subagents.
- Poe model calls via `fastapi-poe` (`https://creator.poe.com/api-reference/overview`).
- A CLI with Codex-like turn loop and streaming output.
- Dual model path: backend-proxy model calls or direct model calls from CLI.

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
- Context is mode-based: coding clears per turn, chat/planning keeps a short window.


## Notable CLI Commands

- `/listmodels` to inspect supported models.
- `/changemodel <name|auto>` to switch main model during a session.
- `/plan` to switch to planning mode with planning-focused system message.
- `/image <path|url>`, `/images`, `/clearimages` for image attachments on the next request.

- Web tools: `GetWebRaw`, `GetWeb`, and `GetWebFile` are exposed via `/tools/invoke` and dedicated API routes.

- CLI supports `/balance` to read current Poe point balance.
- CLI output uses live streamed events (`status`, `tool`, `delta`) for better readability.

- Base main/subagent system prompts live in `poecoder/prompts.py`.
- `StartSubAgent` supports `system_message_modifier` so the main model can shape subagent behavior safely.

- CLI i18n: set `POECODER_LANG=zh-cn` or run `poecoder --lang zh-cn` (you can also switch live with `/lang zh-cn`).

- Background tasks: `/bgturn`, `/bgsubagent`, `/tasks`, `/task`, `/canceltask`.
- Task output shortcuts: `/readtaskoutput <task_id>` and API `GET /tasks/{task_id}/output`.
- Model tools now support async task orchestration: `StartBackgroundTurn`, `StartBackgroundSubAgent`, `ReadTaskOutput`.
- Turn streaming now uses model chunk streaming for lower latency (`delta` events are emitted as chunks arrive).
