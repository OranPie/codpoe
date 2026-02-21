from __future__ import annotations

from poecoder.config import _normalize_openai_models, _parse_bool


def test_openai_models_are_prefixed() -> None:
    out = _normalize_openai_models("gpt-4.1-mini,openai/o3,oa:gpt-5-mini")
    assert out == ["openai/gpt-4.1-mini", "openai/o3", "openai/gpt-5-mini"]


def test_parse_bool_values() -> None:
    assert _parse_bool("true", False) is True
    assert _parse_bool("on", False) is True
    assert _parse_bool("0", True) is False
    assert _parse_bool("off", True) is False
    assert _parse_bool("unknown", True) is True
