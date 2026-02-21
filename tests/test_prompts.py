from __future__ import annotations

from poecoder.prompts import MAIN_SYSTEM_MESSAGE, compose_subagent_system_message


def test_main_system_message_contains_tool_protocol() -> None:
    assert "@tool ToolName {json_args}" in MAIN_SYSTEM_MESSAGE
    assert "Do not fabricate" in MAIN_SYSTEM_MESSAGE


def test_subagent_modifier_is_included() -> None:
    msg = compose_subagent_system_message("readonly", "Focus on tests only.")
    assert "Permission: readonly" in msg
    assert "Focus on tests only." in msg
