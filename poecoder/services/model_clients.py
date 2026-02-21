from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fastapi_poe as fp


@dataclass(slots=True)
class ModelReply:
    text: str
    raw: dict[str, Any]


@dataclass(slots=True)
class ModelProviderError(Exception):
    model: str
    code: str
    detail: str
    hint: str = ""
    http_status: int = 502
    retryable: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.detail

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "provider": "poe",
            "model": self.model,
            "code": self.code,
            "detail": self.detail,
            "retryable": self.retryable,
        }
        if self.hint:
            payload["hint"] = self.hint
        if self.raw:
            payload["raw"] = self.raw
        return payload


class PoeModelClient:
    def __init__(
        self,
        api_url: str,
        api_key: str | None,
        openai_api_url: str = "https://api.openai.com/v1",
        openai_api_key: str | None = None,
        openai_models: list[str] | None = None,
    ) -> None:
        self.api_url = self._normalize_api_url(api_url)
        self.api_key = api_key
        self.openai_api_url = self._normalize_openai_api_url(openai_api_url)
        self.openai_api_key = openai_api_key
        self.openai_models = {self._strip_openai_prefix(name) for name in (openai_models or []) if name}

    def update_poe(self, api_key: str | None = None, base_url: str | None = None) -> None:
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.api_url = self._normalize_api_url(base_url)

    def update_openai(self, api_key: str | None = None, base_url: str | None = None) -> None:
        if api_key is not None:
            self.openai_api_key = api_key
        if base_url is not None:
            self.openai_api_url = self._normalize_openai_api_url(base_url)

    async def chat(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str] | None = None,
    ) -> ModelReply:
        provider, provider_model = self._resolve_provider(model)
        if provider == "openai":
            return await self._chat_openai(provider_model, system_message, user_prompt, context, images or [])

        if not self.api_key:
            return self._mock_reply(model, system_message, user_prompt, context, images or [])

        chunks: list[str] = []
        events: list[dict[str, Any]] = []
        async for partial in self._iter_partials(provider_model, system_message, user_prompt, context, images or []):
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
        provider, provider_model = self._resolve_provider(model)
        if provider == "openai":
            async for chunk in self._chat_stream_openai(provider_model, system_message, user_prompt, context, images or []):
                yield chunk
            return

        if not self.api_key:
            yield self._mock_reply(model, system_message, user_prompt, context, images or []).text
            return

        async for partial in self._iter_partials(provider_model, system_message, user_prompt, context, images or []):
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
        try:
            async for partial in fp.get_bot_response(
                messages=messages,
                bot_name=model,
                api_key=self.api_key or "",
                base_url=self.api_url,
            ):
                yield partial
        except Exception as exc:  # noqa: BLE001
            raise self._translate_provider_error(model, exc) from exc

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

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        base = (api_url or "").strip()
        if not base:
            return "https://api.poe.com/bot/"
        if base.startswith("https://api.poe.com") and "/bot/" not in base:
            base = base.rstrip("/") + "/bot/"
        if not base.endswith("/"):
            base += "/"
        return base

    @staticmethod
    def _normalize_openai_api_url(api_url: str) -> str:
        base = (api_url or "").strip()
        if not base:
            return "https://api.openai.com/v1"
        return base.rstrip("/")

    def _resolve_provider(self, model: str) -> tuple[str, str]:
        if model.startswith("openai/"):
            return "openai", model.split("/", 1)[1]
        if model.startswith("oa:"):
            return "openai", model.split(":", 1)[1]
        return "poe", model

    @staticmethod
    def _strip_openai_prefix(model: str) -> str:
        if model.startswith("openai/"):
            return model.split("/", 1)[1]
        if model.startswith("oa:"):
            return model.split(":", 1)[1]
        return model

    async def _chat_openai(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str],
    ) -> ModelReply:
        if not self.openai_api_key:
            raise ModelProviderError(
                model=model,
                code="openai_auth_error",
                detail="OpenAI API key is not configured.",
                hint="Set POECODER_OPENAI_API_KEY or call /auth/openai/login.",
                http_status=401,
                retryable=False,
            )
        messages = self._build_openai_messages(system_message, user_prompt, context, images)
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(
                model=model,
                code="openai_sdk_missing",
                detail="OpenAI Python SDK is not installed.",
                hint="Install with `python -m pip install openai`.",
                http_status=500,
                retryable=False,
            ) from exc

        client = AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_api_url, timeout=90.0)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            text = self._extract_openai_text(response)
            return ModelReply(text=text, raw={"provider": "openai"})
        except Exception as exc:  # noqa: BLE001
            raise self._translate_openai_error(model, exc) from exc
        finally:
            await client.close()

    async def _chat_stream_openai(
        self,
        model: str,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str],
    ) -> AsyncIterator[str]:
        if not self.openai_api_key:
            raise ModelProviderError(
                model=model,
                code="openai_auth_error",
                detail="OpenAI API key is not configured.",
                hint="Set POECODER_OPENAI_API_KEY or call /auth/openai/login.",
                http_status=401,
                retryable=False,
            )
        messages = self._build_openai_messages(system_message, user_prompt, context, images)
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(
                model=model,
                code="openai_sdk_missing",
                detail="OpenAI Python SDK is not installed.",
                hint="Install with `python -m pip install openai`.",
                http_status=500,
                retryable=False,
            ) from exc
        client = AsyncOpenAI(api_key=self.openai_api_key, base_url=self.openai_api_url, timeout=90.0)
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    yield content
                    continue
                if isinstance(content, list):
                    for part in content:
                        text = ""
                        if isinstance(part, dict):
                            text = str(part.get("text", ""))
                        else:
                            text = str(getattr(part, "text", ""))
                        if text:
                            yield text
        except Exception as exc:  # noqa: BLE001
            raise self._translate_openai_error(model, exc) from exc
        finally:
            await client.close()

    def _build_openai_messages(
        self,
        system_message: str,
        user_prompt: str,
        context: dict[str, Any],
        images: list[str],
    ) -> list[dict[str, Any]]:
        payload = json.dumps({"prompt": user_prompt, "context": context}, ensure_ascii=True)
        messages: list[dict[str, Any]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        attachments = self._build_attachments(images)
        if attachments:
            content: list[dict[str, Any]] = [{"type": "text", "text": payload}]
            for item in attachments:
                content.append({"type": "image_url", "image_url": {"url": item.url}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": payload})
        return messages

    @staticmethod
    def _extract_openai_text(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = str(part.get("text", ""))
                else:
                    text = str(getattr(part, "text", ""))
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content)

    def _translate_provider_error(self, model: str, exc: Exception) -> ModelProviderError:
        chain = self._exception_chain(exc)
        chain_text = " | ".join(f"{type(item).__name__}: {item}" for item in chain)
        lower = chain_text.lower()
        detail = f"Failed to query Poe model '{model}'."
        hint = (
            "Verify API key and model via /login and /listmodels. "
            "Poe can return non-SSE error payloads for auth/model/rate issues."
        )
        code = "poe_upstream_error"
        status = 502
        retryable = True

        if "content-type to contain 'text/event-stream'" in lower:
            code = "poe_non_sse_error"
            detail = f"Poe returned a non-stream response while querying '{model}'."
            hint = (
                "Poe Query API streams SSE for success. A non-SSE response usually means "
                "an upstream JSON/HTML error (invalid model, auth, quota, or temporary outage)."
            )
        elif "401" in lower or "unauthorized" in lower or "forbidden" in lower:
            code = "poe_auth_error"
            detail = "Poe API rejected the current API key."
            hint = "Run /login with a valid Poe API key, then retry."
            status = 401
            retryable = False
        elif "429" in lower or "rate limit" in lower:
            code = "poe_rate_limited"
            detail = "Poe API rate limit reached."
            hint = "Retry with backoff or switch to a lower-cost model."
            status = 429
        elif "404" in lower or "not found" in lower:
            code = "poe_model_not_found"
            detail = f"Model or bot '{model}' is not available for this Poe key."
            hint = "Use /listmodels and /changemodel to select a supported model."
            status = 400
            retryable = False

        return ModelProviderError(
            model=model,
            code=code,
            detail=detail,
            hint=hint,
            http_status=status,
            retryable=retryable,
            raw={"error_chain": chain_text[:1200]},
        )

    def _translate_openai_error(self, model: str, exc: Exception) -> ModelProviderError:
        chain = self._exception_chain(exc)
        chain_text = " | ".join(f"{type(item).__name__}: {item}" for item in chain)
        lower = chain_text.lower()
        detail = f"Failed to query OpenAI model '{model}'."
        hint = "Verify OpenAI API key and model, then retry."
        code = "openai_upstream_error"
        status = 502
        retryable = True

        if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
            code = "openai_auth_error"
            detail = "OpenAI rejected the API key."
            hint = "Call /auth/openai/login with a valid key."
            status = 401
            retryable = False
        elif "404" in lower or "model" in lower and "not found" in lower:
            code = "openai_model_not_found"
            detail = f"OpenAI model '{model}' is not available for this account."
            hint = "Use /listmodels and /changemodel to select a supported model."
            status = 400
            retryable = False
        elif "429" in lower or "rate limit" in lower:
            code = "openai_rate_limited"
            detail = "OpenAI rate limit reached."
            hint = "Retry with backoff or lower request rate."
            status = 429

        return ModelProviderError(
            model=model,
            code=code,
            detail=detail,
            hint=hint,
            http_status=status,
            retryable=retryable,
            raw={"error_chain": chain_text[:1200]},
        )

    @staticmethod
    def _exception_chain(exc: Exception) -> list[Exception]:
        chain: list[Exception] = []
        current: Exception | None = exc
        while current is not None and current not in chain:
            chain.append(current)
            cause = current.__cause__
            if isinstance(cause, Exception):
                current = cause
                continue
            context = current.__context__
            if isinstance(context, Exception):
                current = context
                continue
            current = None
        return chain
