from __future__ import annotations

from datetime import datetime, timezone

from poecoder.models import RouterDecision, SessionResponse, TurnRequest
from poecoder.services.turn_service import TurnService
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


def test_parse_tool_calls_accepts_json_wrapper_fallback() -> None:
    text = '{"tool_name":"ListModels","args":{"refresh":false}}'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "ListModels", "args": {"refresh": False}}]


def test_tool_event_forwarding_compacts_large_payload_in_auto_mode() -> None:
    svc = TurnService(  # type: ignore[arg-type]
        sessions=None,
        memories=None,
        model_client=None,
        router=None,
        tools=None,
        model_catalog=None,
        model_profiles=None,
    )
    events = [
        {
            "name": "ListFile",
            "args": {"path": ".", "limit": 2000},
            "result": {"entries": [f"file_{i}.py" for i in range(1200)]},
        }
    ]
    forwarded, meta = svc._prepare_tool_events_for_prompt(events, {"tool_result_mode": "auto"})
    assert len(forwarded) == 1
    result = forwarded[0]["result"]
    assert isinstance(result, dict)
    assert "__truncated_items__" in result["entries"][-1]
    assert int(meta["compacted_events"]) == 1
    assert int(meta["forwarded_tokens_total"]) < int(meta["original_tokens_total"])


def test_tool_event_forwarding_keeps_full_payload_in_full_mode() -> None:
    svc = TurnService(  # type: ignore[arg-type]
        sessions=None,
        memories=None,
        model_client=None,
        router=None,
        tools=None,
        model_catalog=None,
        model_profiles=None,
    )
    events = [
        {
            "name": "ListFile",
            "args": {"path": ".", "limit": 400},
            "result": {"entries": [f"file_{i}.py" for i in range(400)]},
        }
    ]
    forwarded, meta = svc._prepare_tool_events_for_prompt(events, {"tool_result_mode": "full"})
    assert len(forwarded[0]["result"]["entries"]) == 400
    assert int(meta["compacted_events"]) == 0


def test_prepare_turn_includes_previous_user_message_in_context() -> None:
    class _Sessions:
        def __init__(self) -> None:
            now = datetime.now(tz=timezone.utc)
            self.session = SessionResponse(
                id="s1",
                title="",
                mode="coding",
                active_model="auto",
                thinking_level="balanced",
                thinking_budget=12000,
                show_think_details=False,
                allow_model_command_create=True,
                encourage_model_command_create=True,
                policy_profile="default",
                project_id="default",
                created_at=now,
                updated_at=now,
            )

        def get(self, session_id: str) -> SessionResponse:
            assert session_id == "s1"
            return self.session

        def reset_for_turn(self, session: SessionResponse) -> None:
            assert session.id == "s1"

        def get_context(self, session_id: str, keys: list[str] | None = None) -> dict[str, str]:
            assert session_id == "s1"
            return {"last_user_prompt": "previous user message"}

        def select_context_for_prompt(
            self,
            session_id: str,
            prompt: str,
            keys: list[str] | None = None,
            max_items: int = 20,
            max_value_chars: int = 10000,
        ) -> tuple[dict[str, str], dict[str, int]]:
            assert session_id == "s1"
            return {}, {"selected_items": 0}

    class _Memories:
        def read(self, req: object) -> list[object]:
            return []

    class _Router:
        def decide(self, prompt: str, context_size_hint: int = 0, tool_count_hint: int = 0) -> RouterDecision:
            return RouterDecision(
                classifier_model="assistant",
                selected_model="assistant",
                complexity="small",
                reason="stub",
            )

    class _Tools:
        def command_catalog(self) -> dict[str, list[dict[str, str]]]:
            return {"commands": []}

    class _Catalog:
        def list_models(self, refresh: bool = False) -> list[str]:
            return ["assistant"]

        def ensure_supported(self, model: str) -> None:
            return None

    class _Profiles:
        def ensure_seeded(self, available_models: list[str]) -> None:
            return None

        def choose_model(
            self,
            available_models: list[str],
            fallback_model: str,
            complexity: str,
            thinking_level: str,
            thinking_budget: int,
        ) -> str:
            return fallback_model

    svc = TurnService(  # type: ignore[arg-type]
        sessions=_Sessions(),
        memories=_Memories(),
        model_client=None,
        router=_Router(),
        tools=_Tools(),
        model_catalog=_Catalog(),
        model_profiles=_Profiles(),
    )
    req = TurnRequest(session_id="s1", user_prompt="now do it")
    _, context, _, _, _ = svc._prepare_turn(req)
    assert context["conversation"]["previous_user_message"] == "previous user message"
