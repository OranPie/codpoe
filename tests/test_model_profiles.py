from __future__ import annotations

from poecoder.db import Database
from poecoder.services.model_profile_service import ModelProfileService


def test_choose_model_prefers_fallback_on_score_tie(tmp_path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    svc = ModelProfileService(db=db)
    svc.upsert(
        model="assistant",
        strategy="s",
        best_for="x",
        speed_tier=2,
        quality_tier=5,
        cost_tier=4,
        max_context_hint=128000,
    )
    svc.upsert(
        model="gpt-5.2",
        strategy="s",
        best_for="x",
        speed_tier=2,
        quality_tier=5,
        cost_tier=4,
        max_context_hint=128000,
    )
    picked = svc.choose_model(
        available_models=["assistant", "gpt-5.2"],
        fallback_model="gpt-5.2",
        complexity="large",
        thinking_level="balanced",
        thinking_budget=12000,
    )
    assert picked == "gpt-5.2"


def test_choose_model_does_not_pick_assistant_for_large_tasks(tmp_path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    svc = ModelProfileService(db=db)
    svc.upsert(
        model="assistant",
        strategy="legacy",
        best_for="legacy",
        speed_tier=2,
        quality_tier=5,
        cost_tier=4,
        max_context_hint=128000,
    )
    svc.upsert(
        model="gpt-5.2-codex",
        strategy="deep",
        best_for="coding",
        speed_tier=2,
        quality_tier=5,
        cost_tier=4,
        max_context_hint=128000,
    )
    picked = svc.choose_model(
        available_models=["assistant", "gpt-5.2-codex"],
        fallback_model="gpt-5.2-codex",
        complexity="large",
        thinking_level="balanced",
        thinking_budget=12000,
    )
    assert picked == "gpt-5.2-codex"


def test_choose_model_prefers_codex_for_large_tasks(tmp_path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    svc = ModelProfileService(db=db)
    svc.upsert(
        model="gpt-5.2",
        strategy="deep",
        best_for="general",
        speed_tier=2,
        quality_tier=5,
        cost_tier=4,
        max_context_hint=128000,
    )
    svc.upsert(
        model="gpt-5.2-codex",
        strategy="deep",
        best_for="coding",
        speed_tier=2,
        quality_tier=5,
        cost_tier=4,
        max_context_hint=128000,
    )
    picked = svc.choose_model(
        available_models=["gpt-5.2", "gpt-5.2-codex"],
        fallback_model="gpt-5.2",
        complexity="large",
        thinking_level="balanced",
        thinking_budget=12000,
    )
    assert picked == "gpt-5.2-codex"
