from __future__ import annotations

import asyncio
import sys
import types

import fastapi_poe as fp

from poecoder.services.model_clients import PoeModelClient


def test_chat_builds_image_attachments(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured: dict[str, object] = {}

    async def fake_get_bot_response(messages, bot_name, api_key, base_url):
        captured["messages"] = messages
        yield fp.PartialResponse(text="ok")

    monkeypatch.setattr(fp, "get_bot_response", fake_get_bot_response)

    async def run() -> None:
        client = PoeModelClient(api_url="https://example.invalid", api_key="test-key")
        reply = await client.chat(
            model="assistant",
            system_message="sys",
            user_prompt="hello",
            context={},
            images=[str(image_path), "https://example.com/image.jpg"],
        )
        assert reply.text == "ok"

    asyncio.run(run())
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    user_message = messages[1]
    assert len(user_message.attachments) == 2
    assert user_message.attachments[0].url.startswith("data:image/png;base64,")
    assert user_message.attachments[1].url == "https://example.com/image.jpg"


def test_chat_routes_to_openai_when_model_has_prefix(monkeypatch) -> None:
    called: dict[str, str] = {}

    async def fake_openai(model, system_message, user_prompt, context, images):
        called["model"] = model
        return type("Reply", (), {"text": "oa-ok", "raw": {"provider": "openai"}})()

    async def run() -> None:
        client = PoeModelClient(
            api_url="https://api.poe.com/bot/",
            api_key="poe-key",
            openai_api_url="https://openai.local/v1/",
            openai_api_key="oa-key",
            openai_models=["gpt-4.1-mini"],
        )
        monkeypatch.setattr(client, "_chat_openai", fake_openai)
        reply = await client.chat(
            model="openai/gpt-4.1-mini",
            system_message="sys",
            user_prompt="hello",
            context={},
            images=[],
        )
        assert reply.text == "oa-ok"

    asyncio.run(run())
    assert called["model"] == "gpt-4.1-mini"


def test_update_provider_base_urls() -> None:
    client = PoeModelClient(
        api_url="https://api.poe.com",
        api_key="poe-key",
        openai_api_url="https://openai.local/v1/",
        openai_api_key="oa-key",
    )
    client.update_poe(base_url="https://poe.proxy.local")
    client.update_openai(base_url="https://openai.proxy.local/v1/")
    assert client.api_url == "https://poe.proxy.local/"
    assert client.openai_api_url == "https://openai.proxy.local/v1"


def test_bare_model_name_stays_on_poe_even_if_openai_models_configured(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def fake_get_bot_response(messages, bot_name, api_key, base_url):
        captured["bot_name"] = bot_name
        yield fp.PartialResponse(text="poe-ok")

    async def fail_openai(*args, **kwargs):
        raise AssertionError("openai path should require openai/ prefix")

    monkeypatch.setattr(fp, "get_bot_response", fake_get_bot_response)

    async def run() -> None:
        client = PoeModelClient(
            api_url="https://api.poe.com/bot/",
            api_key="poe-key",
            openai_api_url="https://openai.local/v1",
            openai_api_key="oa-key",
            openai_models=["openai/gpt-4.1-mini"],
        )
        monkeypatch.setattr(client, "_chat_openai", fail_openai)
        reply = await client.chat(
            model="gpt-4.1-mini",
            system_message="sys",
            user_prompt="hello",
            context={},
            images=[],
        )
        assert reply.text == "poe-ok"

    asyncio.run(run())
    assert captured["bot_name"] == "gpt-4.1-mini"


def test_openai_chat_captures_thinking_when_enabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            message = types.SimpleNamespace(content="final answer", reasoning_content="step-by-step trace")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))

    async def run() -> None:
        client = PoeModelClient(
            api_url="https://api.poe.com/bot/",
            api_key="poe-key",
            openai_api_url="https://openai.local/v1",
            openai_api_key="oa-key",
        )
        monkeypatch.setattr(client, "_get_openai_client", lambda: fake_client)
        reply = await client._chat_openai(
            model="gpt-5",
            system_message="sys",
            user_prompt="hello",
            context={"model_settings": {"show_think_details": True, "thinking_level": "deep"}},
            images=[],
        )
        assert "[thinking]" in reply.text
        assert "step-by-step trace" in reply.text
        assert reply.text.endswith("final answer")

    asyncio.run(run())
    assert captured["reasoning_effort"] == "high"


def test_openai_stream_emits_thinking_before_answer(monkeypatch) -> None:
    class FakeStream:
        def __init__(self, chunks):
            self._chunks = chunks
            self._idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx >= len(self._chunks):
                raise StopAsyncIteration
            item = self._chunks[self._idx]
            self._idx += 1
            return item

    class FakeCompletions:
        async def create(self, **kwargs):
            del kwargs
            chunks = [
                types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(reasoning_content="think-1"))]),
                types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="answer"))]),
            ]
            return FakeStream(chunks)

    fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))

    async def run() -> list[str]:
        client = PoeModelClient(
            api_url="https://api.poe.com/bot/",
            api_key="poe-key",
            openai_api_url="https://openai.local/v1",
            openai_api_key="oa-key",
        )
        monkeypatch.setattr(client, "_get_openai_client", lambda: fake_client)
        chunks: list[str] = []
        async for chunk in client._chat_stream_openai(
            model="gpt-5",
            system_message="sys",
            user_prompt="hello",
            context={"model_settings": {"show_think_details": True}},
            images=[],
        ):
            chunks.append(chunk)
        return chunks

    out = asyncio.run(run())
    assert "".join(out).startswith("[thinking]\nthink-1")
    assert "".join(out).endswith("answer")


def test_openai_client_cache_reuses_connection_and_resets_on_update(monkeypatch) -> None:
    init_calls = {"n": 0}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            init_calls["n"] += 1
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    client = PoeModelClient(
        api_url="https://api.poe.com/bot/",
        api_key="poe-key",
        openai_api_url="https://openai.local/v1",
        openai_api_key="oa-key",
    )

    first = client._get_openai_client()
    second = client._get_openai_client()
    assert first is second
    assert init_calls["n"] == 1

    client.update_openai(base_url="https://openai.proxy/v1")
    third = client._get_openai_client()
    assert third is not first
    assert init_calls["n"] == 2


def test_openai_reasoning_effort_uses_model_spec_matrix() -> None:
    deep = PoeModelClient._openai_request_kwargs(
        model="gpt-5.2-codex-max",
        context={"model_settings": {"thinking_level": "deep"}},
    )
    assert deep["reasoning_effort"] == "xhigh"

    quick = PoeModelClient._openai_request_kwargs(
        model="gpt-5",
        context={"model_settings": {"thinking_level": "quick"}},
    )
    assert quick["reasoning_effort"] in {"minimal", "low"}

    pro = PoeModelClient._openai_request_kwargs(
        model="gpt-5-pro",
        context={"model_settings": {"thinking_level": "quick"}},
    )
    assert pro["reasoning_effort"] == "high"


def test_thinking_support_summary_for_models() -> None:
    assert PoeModelClient.thinking_support_summary("openai/gpt-5.2") == "reasoning:none/low/medium/high/xhigh"
    assert PoeModelClient.thinking_support_summary("assistant") == "prompt-only"
