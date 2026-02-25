# PoeCoder AgentCore

Version tag: `v0.0.3`

This branch is a full rewrite to an **agent-driven backend**:
- Agent is core.
- Session is conversation continuity.
- Memory (`user` / `session`) is persistent context.
- Execution primitive is only **RunShell**.

## Core Ideas

- Runtime creates small, precise agents on demand.
- Complex tasks are solved by nested agents (max depth: 2).
- Built-in global templates:
  - `shell-reader`
  - `python-runner`
  - `wget-downloader`
  - `web-searcher`

## API

All new routes are under `/agent-api`.

- Sessions:
  - `POST /agent-api/sessions`
  - `GET /agent-api/sessions`
  - `GET /agent-api/sessions/{session_id}`
  - `POST /agent-api/sessions/{session_id}/turn`
  - `POST /agent-api/sessions/{session_id}/turn/stream` (SSE progress stream)
  - `GET /agent-api/sessions/{session_id}/messages`
  - turn responses may include `ask` payload for clarification (text/single/multiple choice)
- Models:
  - `GET /agent-api/models?query=<optional>&limit=<optional>&full=<true|false>`
    - `full=true` fetches full catalog (configured + OpenAI remote list when key exists)
- Agents:
  - `POST /agent-api/agents/start`
  - `GET /agent-api/agents/{agent_id}`
  - `GET /agent-api/agents/{agent_id}/events`
  - `POST /agent-api/agents/{agent_id}/cancel`
  - `POST /agent-api/agents/{agent_id}/wait`
- Templates:
  - `POST /agent-api/agents/templates/register`
  - `GET /agent-api/agents/templates`
- Memory:
  - `POST /agent-api/memory/user/write`
  - `POST /agent-api/memory/user/read`
  - `POST /agent-api/memory/session/write`
  - `POST /agent-api/memory/session/read`
- Shell:
  - `POST /agent-api/run-shell`
- Auth:
  - `POST /agent-api/auth/poe/login` with body `{"api_key":"..."}`
  - `POST /agent-api/auth/openai/login` with body `{"api_key":"..."}`
  - `POST /agent-api/auth/secrets/save` with body `{"user_key":"...","poe_api_key":"...","openai_api_key":"..."}`
  - `POST /agent-api/auth/secrets/load` with body `{"user_key":"..."}`
  - legacy aliases kept for migration: `/auth/secrets/save`, `/auth/secrets/load`
- Workflow:
  - `POST /agent-api/workflows/arxiv`
    - staged agents: `init -> arxiv-finder -> arxiv-download -> final-report`
    - stage outputs are persisted into session memory under `workflow.arxiv.*`
- Research (built-in parser to reduce token pressure):
  - `POST /agent-api/research/search-web`
  - `POST /agent-api/research/search-arxiv`
  - `POST /agent-api/research/get-web`
  - `POST /agent-api/research/download-urls`

## Quick Start

```bash
poecoder-api
```

Then:

```bash
poecoder
```

In CLI:
- normal prompt => conversation turn
- `/arxiv <query>` => multi-agent workflow (`init -> search -> download -> done`)

## CUI mode

Default `poecoder` now launches a curses CUI:
- `F2` create/switch new session
- `F5` reload session messages
- `F6` toggle progress density (`compact`/`verbose`)
- `PgUp/PgDn` scroll history
- `Up/Down` recall typed input history
- `Tab` completes slash commands
- `/help` for command list (`/models`, `/models full`, `/model`, `/stream`, `/progress`, `/runshell`, `/llm`, `/agentinfo`, `/ask`, `/answer`, `/skipask`, `/secretssave`, `/secretsload`, `/new`, `/sessions`, `/switch`, `/arxiv`, `/exit`)
- `/progress [compact|verbose]` controls streaming feedback density
- `/runshell <command>` executes shell directly without an LLM turn
- `/llm [on|off]` toggles LLM prompt turns (default: off in CUI)
- long-running shell/tool operations emit immediate "in progress" notes before completion
- live panel includes `Cmd Viz` showing command state (`>` running, `+` success, `!` error, `x` blocked)
- `Esc` during streaming sends cancel request for current agent; ask is shown as popup (not only chat line)
- turn view now shows agent/runshell metrics, token totals, and estimated cost (when usage is available)
- `/secretssave` and `/secretsload` without args now prompt for masked key input (no key echo in chat)
- spawn/runshell actions carry short `progress` updates for clearer middle feedback
- agent `note` actions are rendered directly as assistant feedback (not only operation-log progress lines)
- agent can now define and call reusable runtime tools (`define_tool` / `call_tool`, sh or python)
- defined tools are attached into model context each step (tool catalog with schema + script preview)
- non-final actions should include short `progress` text for middle feedback in stream UI
- ask flow supports clarification prompts with text/single/multiple-choice answers (`/ask`, `/answer`, `/skipask`)
- unknown slash input is treated as command error (not forwarded to LLM prompt)

Fallback plain mode:

```bash
poecoder --mode simple
```
