from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import fastapi_poe as fp


@dataclass(slots=True)
class ModelReply:
    text: str
    raw: dict[str, Any]


class PoeModelClient:
    def __init__(self, api_url: str, api_key: str | None) -> None:
        self.api_url = api_url
        self.api_key = api_key

    async def chat(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
    ) -> ModelReply:
        if not self.api_key:
            return self._mock_reply(model, system_message, user_prompt, context)

        payload = json.dumps(
            {
                "prompt": user_prompt,
                "context": context,
            },
            ensure_ascii=True,
        )
        messages = []
        if system_message:
            messages.append(fp.ProtocolMessage(role="system", content=system_message))
        messages.append(fp.ProtocolMessage(role="user", content=payload))

        chunks: list[str] = []
        events: list[dict[str, Any]] = []
        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name=model,
            api_key=self.api_key,
            base_url=self.api_url,
        ):
            if partial.text:
                chunks.append(partial.text)
            if partial.raw_response is not None:
                events.append({"raw_response": partial.raw_response})
            elif partial.data is not None:
                events.append({"data": partial.data})

        return ModelReply(text="".join(chunks), raw={"events": events})

    @staticmethod
    def _mock_reply(model: str, system_message: str, user_prompt: str, context: dict[str, Any]) -> ModelReply:
        lines = [
            f"[mock:{model}]",
            "No POE API key configured; returning local mock response.",
            f"System length={len(system_message)} user length={len(user_prompt)} context_keys={list(context.keys())}",
            "If you want tool execution, emit lines like:",
            "@tool Search {\"pattern\":\"TODO\",\"file_pattern\":\"*.py\"}",
        ]
        return ModelReply(text="\n".join(lines), raw={"mock": True})
