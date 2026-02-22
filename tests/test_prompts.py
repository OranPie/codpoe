from __future__ import annotations

from poecoder.prompts import (
    LEADER_SYSTEM_MESSAGE,
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
    assert "call a tool first" in MAIN_SYSTEM_MESSAGE
    assert "placeholder markdown" in MAIN_SYSTEM_MESSAGE
    assert "Turn protocol is multi-stage" in MAIN_SYSTEM_MESSAGE
    assert "multiple model turns" in MAIN_SYSTEM_MESSAGE
    assert "Do not ask \"should I proceed?\"" in MAIN_SYSTEM_MESSAGE
    assert "PoeCoder architecture flow" in MAIN_SYSTEM_MESSAGE
    assert "\"exit\", \"quit\", \"close session\"" in MAIN_SYSTEM_MESSAGE
    assert "previous_user_message" in MAIN_SYSTEM_MESSAGE
    assert "previous_turn_conclusion" in MAIN_SYSTEM_MESSAGE
    assert "not auto-carried into default context selection" in MAIN_SYSTEM_MESSAGE
    assert "Help {\"tool_name\":\"ToolName\"}" in MAIN_SYSTEM_MESSAGE
    assert "prefer filtered extraction" in MAIN_SYSTEM_MESSAGE
    assert "@ask {\"prompt\":\"...\",\"key\":\"...\"}" in MAIN_SYSTEM_MESSAGE
    assert "RunShell returns a terminal_id" in MAIN_SYSTEM_MESSAGE


def test_subagent_modifier_is_included() -> None:
    msg = compose_subagent_system_message("readonly", "Focus on tests only.")
    assert "Permission: readonly" in msg
    assert "Focus on tests only." in msg


def test_planning_system_message_router() -> None:
    assert default_system_message_for_mode("planning") == PLAN_SYSTEM_MESSAGE
    assert default_system_message_for_mode("leader") == LEADER_SYSTEM_MESSAGE
    assert default_system_message_for_mode("coding") == MAIN_SYSTEM_MESSAGE
    assert "Reviewer" in REVIEWER_SYSTEM_MESSAGE
