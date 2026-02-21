from __future__ import annotations

import asyncio

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
