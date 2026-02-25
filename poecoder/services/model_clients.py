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
    _OPENAI_REASONING_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
        # Source: OpenAI model/reference docs (gpt-5/o-series reasoning_effort support).
        ("gpt-5-pro", ("high",), "high"),
        ("gpt-5.2-codex-max", ("none", "low", "medium", "high", "xhigh"), "none"),
        ("gpt-5.2", ("none", "low", "medium", "high", "xhigh"), "none"),
        ("gpt-5.1", ("none", "low", "medium", "high"), "none"),
        ("gpt-5", ("minimal", "low", "medium", "high"), "medium"),
        ("o", ("low", "medium", "high"), "medium"),
    )

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
        self._openai_clients: dict[tuple[str, str], Any] = {}

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
        self._openai_clients.clear()

    async def fetch_full_model_catalog(
        self,
        *,
        seeded_models: list[str] | None = None,
        include_openai_remote: bool = True,
        remote_limit: int = 5000,
    ) -> dict[str, Any]:
        local: list[str] = []
        for item in seeded_models or []:
            name = str(item).strip()
            if name and name not in local:
                local.append(name)
        for item in sorted(self.openai_models):
            name = f"openai/{item}"
            if name not in local:
                local.append(name)

        openai_remote: list[str] = []
        remote_error = ""
        if include_openai_remote and self.openai_api_key:
            try:
                client = self._get_openai_client()
                response = await client.models.list()
                data = getattr(response, "data", None) or []
                for item in data:
                    model_id = ""
                    if isinstance(item, dict):
                        model_id = str(item.get("id", "")).strip()
                    else:
                        model_id = str(getattr(item, "id", "")).strip()
                    if not model_id:
                        continue
                    name = f"openai/{model_id}"
                    if name not in openai_remote:
                        openai_remote.append(name)
                    if len(openai_remote) >= max(1, remote_limit):
                        break
            except Exception as exc:  # noqa: BLE001
                remote_error = str(exc)

        merged = list(local)
        for item in openai_remote:
            if item not in merged:
                merged.append(item)

        return {
            "models": merged,
            "sources": {
                "seeded_count": len(local),
                "openai_remote_count": len(openai_remote),
                "openai_remote_error": remote_error,
            },
        }

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
        raw: dict[str, Any] = {"events": events}
        usage = self._extract_poe_usage(events)
        if usage:
            raw["usage"] = usage
        return ModelReply(text="".join(chunks), raw=raw)

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
            "Tool execution is disabled in mock mode.",
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
        client = self._get_openai_client()
        request_kwargs = self._openai_request_kwargs(model, context)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                **request_kwargs,
            )
            text = self._extract_openai_text(
                response,
                include_thinking=self._should_include_thinking(context),
            )
            raw: dict[str, Any] = {"provider": "openai"}
            usage = self._extract_openai_usage(response)
            if usage:
                raw["usage"] = usage
            return ModelReply(text=text, raw=raw)
        except Exception as exc:  # noqa: BLE001
            raise self._translate_openai_error(model, exc) from exc

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
        client = self._get_openai_client()
        request_kwargs = self._openai_request_kwargs(model, context)
        include_thinking = self._should_include_thinking(context)
        emitted_thinking_header = False
        emitted_answer = False
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **request_kwargs,
            )
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                thinking_chunk = self._extract_openai_thinking_from_delta(delta) if include_thinking else ""
                if thinking_chunk:
                    if not emitted_thinking_header:
                        emitted_thinking_header = True
                        yield "[thinking]\n"
                    yield thinking_chunk
                answer_chunk = self._extract_openai_text_from_delta(delta)
                if answer_chunk:
                    if emitted_thinking_header and not emitted_answer:
                        yield "\n\n"
                    emitted_answer = True
                    yield answer_chunk
        except Exception as exc:  # noqa: BLE001
            raise self._translate_openai_error(model, exc) from exc

    def _get_openai_client(self) -> Any:
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            raise ModelProviderError(
                model="openai",
                code="openai_sdk_missing",
                detail="OpenAI Python SDK is not installed.",
                hint="Install with `python -m pip install openai`.",
                http_status=500,
                retryable=False,
            ) from exc
        key = (self.openai_api_key or "", self.openai_api_url)
        cached = self._openai_clients.get(key)
        if cached is not None:
            return cached
        client = AsyncOpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_url,
            timeout=90.0,
            max_retries=2,
        )
        self._openai_clients[key] = client
        return client

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
    def _extract_openai_text(response: Any, include_thinking: bool = False) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            text_out = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = PoeModelClient._extract_text_part(part)
                if text:
                    parts.append(text)
            text_out = "".join(parts)
        else:
            text_out = str(content)

        if not include_thinking:
            return text_out
        thinking = PoeModelClient._extract_openai_thinking_from_message(message)
        if not thinking:
            return text_out
        if text_out:
            return f"[thinking]\n{thinking}\n\n{text_out}"
        return f"[thinking]\n{thinking}"

    @staticmethod
    def _extract_openai_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            total = int(usage.get("total_tokens", prompt + completion) or (prompt + completion))
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", prompt + completion) or (prompt + completion))
        if prompt == 0 and completion == 0 and total == 0:
            return {}
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    @staticmethod
    def _extract_poe_usage(events: list[dict[str, Any]]) -> dict[str, int]:
        for event in events:
            if not isinstance(event, dict):
                continue
            usage = PoeModelClient._find_usage_payload(event)
            if usage:
                return usage
        return {}

    @staticmethod
    def _find_usage_payload(node: Any) -> dict[str, int]:
        if isinstance(node, dict):
            prompt_val = node.get("prompt_tokens", node.get("input_tokens"))
            completion_val = node.get("completion_tokens", node.get("output_tokens"))
            total_val = node.get("total_tokens")
            if prompt_val is not None or completion_val is not None or total_val is not None:
                prompt = int(prompt_val or 0)
                completion = int(completion_val or 0)
                total = int(total_val or (prompt + completion))
                if prompt > 0 or completion > 0 or total > 0:
                    return {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": total,
                    }
            for value in node.values():
                usage = PoeModelClient._find_usage_payload(value)
                if usage:
                    return usage
            return {}
        if isinstance(node, list):
            for value in node:
                usage = PoeModelClient._find_usage_payload(value)
                if usage:
                    return usage
        return {}

    @staticmethod
    def _extract_text_part(part: Any) -> str:
        if isinstance(part, dict):
            part_type = str(part.get("type", "")).lower()
            if part_type not in {"", "text", "output_text"}:
                return ""
            return str(part.get("text", ""))
        part_type = str(getattr(part, "type", "")).lower()
        if part_type not in {"", "text", "output_text"}:
            return ""
        return str(getattr(part, "text", ""))

    @staticmethod
    def _extract_openai_text_from_delta(delta: Any) -> str:
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for part in content:
            text = PoeModelClient._extract_text_part(part)
            if text:
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_openai_thinking_from_message(message: Any) -> str:
        chunks: list[str] = []
        chunks.extend(PoeModelClient._flatten_text(getattr(message, "reasoning_content", None)))
        chunks.extend(PoeModelClient._flatten_text(getattr(message, "reasoning", None)))
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for part in content:
                part_type = ""
                if isinstance(part, dict):
                    part_type = str(part.get("type", "")).lower()
                else:
                    part_type = str(getattr(part, "type", "")).lower()
                if part_type in {"reasoning", "thinking"}:
                    chunks.extend(PoeModelClient._flatten_text(part))
        return PoeModelClient._join_unique_lines(chunks)

    @staticmethod
    def _extract_openai_thinking_from_delta(delta: Any) -> str:
        chunks: list[str] = []
        chunks.extend(PoeModelClient._flatten_text(getattr(delta, "reasoning_content", None)))
        chunks.extend(PoeModelClient._flatten_text(getattr(delta, "reasoning", None)))
        content = getattr(delta, "content", None)
        if isinstance(content, list):
            for part in content:
                part_type = ""
                if isinstance(part, dict):
                    part_type = str(part.get("type", "")).lower()
                else:
                    part_type = str(getattr(part, "type", "")).lower()
                if part_type in {"reasoning", "thinking"}:
                    chunks.extend(PoeModelClient._flatten_text(part))
        return PoeModelClient._join_unique_lines(chunks)

    @staticmethod
    def _flatten_text(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, dict):
            out: list[str] = []
            for key in ("text", "content", "summary", "reasoning", "output_text"):
                if key in value:
                    out.extend(PoeModelClient._flatten_text(value.get(key)))
            return out
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                out.extend(PoeModelClient._flatten_text(item))
            return out
        attrs = ("text", "content", "summary", "reasoning", "output_text")
        out: list[str] = []
        for attr in attrs:
            if hasattr(value, attr):
                out.extend(PoeModelClient._flatten_text(getattr(value, attr)))
        return out

    @staticmethod
    def _join_unique_lines(items: list[str]) -> str:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            clean = item.strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            out.append(clean)
        return "\n".join(out)

    @staticmethod
    def _should_include_thinking(context: dict[str, Any]) -> bool:
        model_settings = context.get("model_settings", {})
        if not isinstance(model_settings, dict):
            return False
        return bool(model_settings.get("show_think_details"))

    @staticmethod
    def _openai_request_kwargs(model: str, context: dict[str, Any]) -> dict[str, Any]:
        spec = PoeModelClient._resolve_openai_reasoning_spec(model)
        if spec is None:
            return {}
        allowed, default_value = spec
        level = PoeModelClient._extract_thinking_level(context)
        candidates_by_level = {
            "quick": ("none", "minimal", "low", default_value),
            "balanced": ("medium", "low", default_value),
            "deep": ("xhigh", "high", "medium", default_value),
        }
        for candidate in candidates_by_level.get(level, (default_value,)):
            if candidate in allowed:
                return {"reasoning_effort": candidate}
        if default_value in allowed:
            return {"reasoning_effort": default_value}
        return {"reasoning_effort": allowed[0]}

    @staticmethod
    def _extract_thinking_level(context: dict[str, Any]) -> str:
        model_settings = context.get("model_settings", {})
        if not isinstance(model_settings, dict):
            return "balanced"
        level = str(model_settings.get("thinking_level", "balanced")).strip().lower()
        if level in {"quick", "balanced", "deep"}:
            return level
        return "balanced"

    @staticmethod
    def _resolve_openai_reasoning_spec(model: str) -> tuple[tuple[str, ...], str] | None:
        normalized = model.strip().lower()
        for prefix, allowed, default_value in PoeModelClient._OPENAI_REASONING_SPECS:
            if normalized.startswith(prefix):
                return allowed, default_value
        return None

    @staticmethod
    def thinking_support_summary(model: str) -> str:
        raw = model.strip().lower()
        if raw.startswith("openai/"):
            model_name = raw.split("/", 1)[1]
            spec = PoeModelClient._resolve_openai_reasoning_spec(model_name)
            if spec is None:
                return "openai(default)"
            allowed, _default = spec
            return "reasoning:" + "/".join(allowed)
        return "prompt-only"

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
