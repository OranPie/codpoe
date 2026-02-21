from __future__ import annotations

from poecoder.config import _normalize_openai_models


def test_openai_models_are_prefixed() -> None:
    out = _normalize_openai_models("gpt-4.1-mini,openai/o3,oa:gpt-5-mini")
    assert out == ["openai/gpt-4.1-mini", "openai/o3", "openai/gpt-5-mini"]
