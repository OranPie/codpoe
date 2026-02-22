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
            "title": "Demo",
            "body": "<html><head><title>Demo</title></head><body><h1>Hello</h1></body></html>",
            "truncated": False,
            "filter": {"mode": "none"},
            "hint": "",
        }

    tools.get_web_raw = fake_raw  # type: ignore[method-assign]
    result = asyncio.run(tools.get_web("https://example.com"))

    assert result["title"] == "Demo"
    assert "Hello" in result["text"]


def test_regex_filter_returns_matches(tmp_path):
    tools = WebTools(download_root=tmp_path)
    body, meta = tools._filter_with_regex("alpha ID-12 and ID-34 end", r"ID-\d+", max_matches=1)
    assert body == "ID-12"
    assert meta["mode"] == "regex"
    assert meta["matches"] == 2
    assert meta["returned_matches"] == 1


def test_get_web_download_if_large(tmp_path):
    tools = WebTools(download_root=tmp_path)

    async def fake_raw(**_: object):
        return {
            "url": "https://example.com/huge",
            "status_code": 200,
            "content_type": "text/html",
            "title": "Huge",
            "body": "<html><body>" + ("x" * 5000) + "</body></html>",
            "truncated": True,
            "filter": {"mode": "none"},
            "hint": "Large page",
        }

    async def fake_file(**_: object):
        return {"saved_to": str(tmp_path / "downloads" / "huge.html"), "bytes": 123}

    tools.get_web_raw = fake_raw  # type: ignore[method-assign]
    tools.get_web_file = fake_file  # type: ignore[method-assign]
    result = asyncio.run(tools.get_web("https://example.com/huge", download_if_large=True))
    assert result["title"] == "Huge"
    assert result.get("downloaded_file", {}).get("bytes") == 123


def test_resolve_dir_blocks_escape(tmp_path):
    tools = WebTools(download_root=tmp_path)
    with pytest.raises(ValueError):
        tools._resolve_dir("/tmp")
