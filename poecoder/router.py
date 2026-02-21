from __future__ import annotations

from poecoder.models import RouterDecision


class ModelRouter:
    def __init__(self, small_model: str, large_model: str) -> None:
        self.small_model = small_model
        self.large_model = large_model

    def decide(self, prompt: str, context_size_hint: int = 0, tool_count_hint: int = 0) -> RouterDecision:
        text = prompt.lower()
        complexity = "small"
        reason = "quick extraction or lookup"

        heavy_signals = (
            "refactor",
            "architecture",
            "multi-file",
            "subagent",
            "design",
            "recursive",
            "dependency",
        )
        medium_signals = ("implement", "fix", "test", "search", "replace")

        if any(signal in text for signal in heavy_signals) or context_size_hint > 8000 or tool_count_hint > 6:
            complexity = "large"
            reason = "complex synthesis or broad context"
        elif any(signal in text for signal in medium_signals) or context_size_hint > 3000:
            complexity = "medium"
            reason = "moderate implementation with tools"

        selected = self.large_model if complexity == "large" else self.small_model
        return RouterDecision(
            classifier_model=self.small_model,
            selected_model=selected,
            complexity=complexity,
            reason=reason,
        )
