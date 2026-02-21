from __future__ import annotations

import asyncio
import pytest

from poecoder.tools.web_tools import WebTools


def test_get_web_extracts_title_and_text(tmp_path):
    tools = WebTools(download_root=tmp_path)

    async def fake_raw(**_: object):
        return {
            "url": "https://example.com",
            "status_code": 200,
            "content_type": "text/html",
            "body": "<html><head><title>Demo</title></head><body><h1>Hello</h1></body></html>",
            "truncated": False,
        }

    tools.get_web_raw = fake_raw  # type: ignore[method-assign]
    result = asyncio.run(tools.get_web("https://example.com"))

    assert result["title"] == "Demo"
    assert "Hello" in result["text"]


def test_resolve_dir_blocks_escape(tmp_path):
    tools = WebTools(download_root=tmp_path)
    with pytest.raises(ValueError):
        tools._resolve_dir("/tmp")
