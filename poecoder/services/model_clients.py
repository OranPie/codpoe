from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
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
        images: list[str] | None = None,
    ) -> ModelReply:
        if not self.api_key:
            return self._mock_reply(model, system_message, user_prompt, context, images or [])

        chunks: list[str] = []
        events: list[dict[str, Any]] = []
        async for partial in self._iter_partials(model, system_message, user_prompt, context, images or []):
            if partial.text:
                chunks.append(partial.text)
            if partial.raw_response is not None:
                events.append({"raw_response": partial.raw_response})
            elif partial.data is not None:
                events.append({"data": partial.data})

        return ModelReply(text="".join(chunks), raw={"events": events})

    async def chat_stream(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str] | None = None,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            yield self._mock_reply(model, system_message, user_prompt, context, images or []).text
            return

        async for partial in self._iter_partials(model, system_message, user_prompt, context, images or []):
            if partial.text:
                yield partial.text

    async def _iter_partials(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str],
    ) -> AsyncIterator[fp.PartialResponse]:
        payload = json.dumps(
            {
                "prompt": user_prompt,
                "context": context,
            },
            ensure_ascii=True,
        )
        attachments = self._build_attachments(images)
        messages = []
        if system_message:
            messages.append(fp.ProtocolMessage(role="system", content=system_message))
        user_message = fp.ProtocolMessage(role="user", content=payload)
        if attachments:
            user_message.attachments = attachments
        messages.append(user_message)

        async for partial in fp.get_bot_response(
            messages=messages,
            bot_name=model,
            api_key=self.api_key or "",
            base_url=self.api_url,
        ):
            yield partial

    @staticmethod
    def _mock_reply(
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str],
    ) -> ModelReply:
        lines = [
            f"[mock:{model}]",
            "No POE API key configured; returning local mock response.",
            f"System length={len(system_message)} user length={len(user_prompt)} context_keys={list(context.keys())}",
            f"Image count={len(images)}",
            "If you want tool execution, emit lines like:",
            "@tool Search {\"pattern\":\"TODO\",\"file_pattern\":\"*.py\"}",
        ]
        return ModelReply(text="\n".join(lines), raw={"mock": True})

    def _build_attachments(self, images: list[str]) -> list[fp.Attachment]:
        attachments: list[fp.Attachment] = []
        for idx, src in enumerate(images):
            image = (src or "").strip()
            if not image:
                continue
            if image.startswith("http://") or image.startswith("https://") or image.startswith("data:"):
                content_type, name = self._guess_remote_attachment_meta(image, idx)
                attachments.append(
                    fp.Attachment(
                        url=image,
                        content_type=content_type,
                        name=name,
                    )
                )
                continue

            path = Path(image).expanduser()
            if not path.exists() or not path.is_file():
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data_url = f"data:{mime_type};base64,{b64encode(data).decode('ascii')}"
            attachments.append(
                fp.Attachment(
                    url=data_url,
                    content_type=mime_type,
                    name=path.name,
                )
            )
        return attachments

    @staticmethod
    def _guess_remote_attachment_meta(src: str, idx: int) -> tuple[str, str]:
        if src.startswith("data:"):
            prefix = src[5:].split(",", 1)[0]
            mime = prefix.split(";", 1)[0] if prefix else "application/octet-stream"
            ext = mimetypes.guess_extension(mime) or ""
            return mime, f"image_{idx + 1}{ext}"
        guessed = mimetypes.guess_type(src)[0] or "application/octet-stream"
        ext = mimetypes.guess_extension(guessed) or ""
        return guessed, f"image_{idx + 1}{ext}"
