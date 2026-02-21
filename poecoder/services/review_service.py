from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from poecoder.models import ReviewRequest
from poecoder.prompts import REVIEWER_SYSTEM_MESSAGE
from poecoder.services.model_catalog import ModelCatalog
from poecoder.services.model_clients import PoeModelClient
from poecoder.services.session_service import SessionService


@dataclass(slots=True)
class ReviewService:
    sessions: SessionService
    model_client: PoeModelClient
    model_catalog: ModelCatalog
    default_model: str
    default_thinking_level: str
    default_thinking_budget: int
    command_catalog_provider: Callable[[], list[dict[str, Any]]]

    def get_settings(self) -> dict[str, Any]:
        return {
            "model": self.default_model,
            "thinking_level": self.default_thinking_level,
            "thinking_budget": self.default_thinking_budget,
        }

    def update_settings(
        self,
        model: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        if model:
            self.default_model = model
        if thinking_level:
            self.default_thinking_level = thinking_level
        if thinking_budget is not None:
            self.default_thinking_budget = thinking_budget
        return self.get_settings()

    async def run(self, req: ReviewRequest) -> dict[str, Any]:
        session = self.sessions.get(req.session_id)
        selected_keys = req.context_keys if req.context_keys else None
        selected_context = self.sessions.get_context(session.id, selected_keys)

        model = req.model or self.default_model
        try:
            self.model_catalog.ensure_supported(model)
        except ValueError:
            available = self.model_catalog.list_models(refresh=True)
            if available:
                model = available[0]

        thinking_level = req.thinking_level or self.default_thinking_level
        thinking_budget = req.thinking_budget or self.default_thinking_budget
        context = {
            "session": session.model_dump(mode="json"),
            "selected_context": selected_context,
            "review_settings": {
                "thinking_level": thinking_level,
                "thinking_budget": thinking_budget,
            },
            "command_catalog": self.command_catalog_provider(),
        }
        effective_system = (
            REVIEWER_SYSTEM_MESSAGE
            + "\n\nReview settings:\n"
            + f"- thinking_level={thinking_level}\n"
            + f"- thinking_budget={thinking_budget}\n"
        )
        reply = await self.model_client.chat(
            model=model,
            system_message=effective_system,
            user_prompt=req.prompt,
            context=context,
        )
        return {
            "model": model,
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "output_text": reply.text,
            "raw": reply.raw,
        }
