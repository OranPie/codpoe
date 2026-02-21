# Context and Turn Flow

This document explains what is carried into each model turn and what is intentionally excluded by default.

## Default turn protocol

- A turn is multi-stage:
  - first stage: model decides tool calls or final answer
  - tool stage: runtime executes tool calls
  - final stage: model receives tool outputs as new input and produces conclusion (or more tool calls)
- The protocol is not single-response. Multiple model passes may happen in one user turn.

## What is included in context by default

- `session`: current session settings and mode
- `selected_context`: auto-ranked compact context entries
- `conversation.previous_user_message`: latest prior user message carry-over
- `memory`: session/project/global/query memory hits
- `context_diagnostics`: selected vs dropped context metadata
- `command_catalog`: available tool names, args, effects
- `command_policy`, `metadata`, and model settings

## What is excluded by default

- Previous final assistant output (`last_model_output`) is excluded from default auto context selection.
- Stored tool payload entries (`tool:*`) are excluded from default auto context selection.
- These can still be loaded on demand using explicit keys or targeted tool calls.

## Tool result forwarding modes

Tool outputs are forwarded into the follow-up prompt using `metadata.tool_result_mode`:

- `auto` (default): compact only when payload is large
- `compact`: always compact tool payloads
- `full`: forward full payloads

Compaction keeps structure but truncates long strings/deep/large lists/maps to reduce token cost.

## Cost control behavior

- Large raw tool payloads are expensive when re-sent to the model.
- Runtime tracks forwarding stats (`original_tokens_total`, `forwarded_tokens_total`, per-tool alerts).
- CLI can surface these stats so user/model can choose `auto`, `compact`, or `full` intentionally.
