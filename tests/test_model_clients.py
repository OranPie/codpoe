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
