from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poecoder.db import Database, utcnow_iso
from poecoder.services.utils import parse_dt


def _infer_profile(model: str) -> dict[str, Any]:
    key = model.lower()
    if "mini" in key or "small" in key or "haiku" in key:
        return {
            "strategy": "fast-iteration",
            "best_for": "quick reads, short edits, and cheap utility steps",
            "speed_tier": 5,
            "quality_tier": 2,
            "cost_tier": 1,
            "max_context_hint": 32000,
        }
    if "sonnet" in key or "4.1" in key:
        return {
            "strategy": "balanced-engineering",
            "best_for": "most coding tasks with good quality/speed balance",
            "speed_tier": 3,
            "quality_tier": 4,
            "cost_tier": 3,
            "max_context_hint": 64000,
        }
    return {
        "strategy": "deep-reasoning",
        "best_for": "planning, architecture, and difficult synthesis",
        "speed_tier": 2,
        "quality_tier": 5,
        "cost_tier": 4,
        "max_context_hint": 128000,
    }


@dataclass(slots=True)
class ModelProfileService:
    db: Database

    def ensure_seeded(self, models: list[str]) -> None:
        for model in models:
            row = self.db.query_one("SELECT model FROM model_profiles WHERE model = ?", (model,))
            if row is not None:
                continue
            profile = _infer_profile(model)
            self.upsert(model=model, **profile)

    def list(self) -> list[dict[str, Any]]:
        rows = self.db.query_all("SELECT * FROM model_profiles ORDER BY model ASC")
        return [self._row_to_dict(row) for row in rows]

    def get(self, model: str) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM model_profiles WHERE model = ?", (model,))
        if row is None:
            raise KeyError(model)
        return self._row_to_dict(row)

    def upsert(
        self,
        model: str,
        strategy: str,
        best_for: str,
        speed_tier: int,
        quality_tier: int,
        cost_tier: int,
        max_context_hint: int,
    ) -> dict[str, Any]:
        now = utcnow_iso()
        self.db.execute(
            """
            INSERT INTO model_profiles(
                model, strategy, best_for, speed_tier, quality_tier, cost_tier, max_context_hint, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model) DO UPDATE SET
                strategy = excluded.strategy,
                best_for = excluded.best_for,
                speed_tier = excluded.speed_tier,
                quality_tier = excluded.quality_tier,
                cost_tier = excluded.cost_tier,
                max_context_hint = excluded.max_context_hint,
                updated_at = excluded.updated_at
            """,
            (
                model,
                strategy,
                best_for,
                speed_tier,
                quality_tier,
                cost_tier,
                max_context_hint,
                now,
                now,
            ),
        )
        return self.get(model)

    def choose_model(
        self,
        available_models: list[str],
        fallback_model: str,
        complexity: str,
        thinking_level: str,
        thinking_budget: int,
    ) -> str:
        rows = self.list()
        by_model = {item["model"]: item for item in rows if item["model"] in available_models}
        if not by_model:
            return fallback_model

        budget_factor = 1 if thinking_budget >= 20000 else 0
        best_model = fallback_model
        best_score = -10**9
        for model in available_models:
            profile = by_model.get(model)
            if profile is None:
                continue
            score = 0
            speed = int(profile["speed_tier"])
            quality = int(profile["quality_tier"])
            cost = int(profile["cost_tier"])
            context = int(profile["max_context_hint"])

            if complexity == "large":
                score += quality * 4 + context // 20000
            elif complexity == "medium":
                score += quality * 2 + speed * 2
            else:
                score += speed * 3 - cost

            if thinking_level == "deep":
                score += quality * 4 + budget_factor * 2 - speed
            elif thinking_level == "quick":
                score += speed * 4 - cost * 2
            else:
                score += quality + speed

            if score > best_score:
                best_score = score
                best_model = model
        return best_model

    @staticmethod
    def _row_to_dict(row: object) -> dict[str, Any]:
        return {
            "model": row["model"],
            "strategy": row["strategy"],
            "best_for": row["best_for"],
            "speed_tier": int(row["speed_tier"]),
            "quality_tier": int(row["quality_tier"]),
            "cost_tier": int(row["cost_tier"]),
            "max_context_hint": int(row["max_context_hint"]),
            "created_at": parse_dt(row["created_at"]),
            "updated_at": parse_dt(row["updated_at"]),
        }
