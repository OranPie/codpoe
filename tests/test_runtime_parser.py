from __future__ import annotations

from poecoder.tools.runtime import parse_tool_calls


def test_parse_tool_calls_accepts_prefixed_text() -> None:
    text = "Thinking... (1s elapsed)@tool ListFile {}"
    calls = parse_tool_calls(text)
    assert calls == [{"name": "ListFile", "args": {}}]


def test_parse_tool_calls_parses_multiple_calls() -> None:
    text = (
        "phase1 @tool Search {\"pattern\":\"TODO\",\"file_pattern\":\"*.py\"}\n"
        "phase2 @tool ListFile {\"recursive\":true}\n"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "Search"
    assert calls[1]["name"] == "ListFile"
