from __future__ import annotations

from poecoder.prompts import (
    MAIN_SYSTEM_MESSAGE,
    PLAN_SYSTEM_MESSAGE,
    REVIEWER_SYSTEM_MESSAGE,
    compose_subagent_system_message,
    default_system_message_for_mode,
)


def test_main_system_message_contains_tool_protocol() -> None:
    assert "@tool ToolName {json_args}" in MAIN_SYSTEM_MESSAGE
    assert "Do not fabricate" in MAIN_SYSTEM_MESSAGE
    assert "command_catalog" in MAIN_SYSTEM_MESSAGE
    assert "InstallCommand" in MAIN_SYSTEM_MESSAGE


def test_subagent_modifier_is_included() -> None:
    msg = compose_subagent_system_message("readonly", "Focus on tests only.")
    assert "Permission: readonly" in msg
    assert "Focus on tests only." in msg


def test_planning_system_message_router() -> None:
    assert default_system_message_for_mode("planning") == PLAN_SYSTEM_MESSAGE
    assert default_system_message_for_mode("coding") == MAIN_SYSTEM_MESSAGE
    assert "Reviewer" in REVIEWER_SYSTEM_MESSAGE
