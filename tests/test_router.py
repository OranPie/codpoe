from __future__ import annotations

from poecoder.router import ModelRouter


def test_router_marks_file_listing_prompt_as_medium() -> None:
    router = ModelRouter("assistant", "gpt-5.2-codex")
    decision = router.decide("list all cwd entries.")
    assert decision.complexity == "medium"
