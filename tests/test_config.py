from __future__ import annotations

from poecoder.config import _normalize_openai_models, _parse_bool, get_settings


def test_openai_models_are_prefixed() -> None:
    out = _normalize_openai_models("gpt-4.1-mini,openai/o3,oa:gpt-5-mini")
    assert out == ["openai/gpt-4.1-mini", "openai/o3", "openai/gpt-5-mini"]


def test_parse_bool_values() -> None:
    assert _parse_bool("true", False) is True
    assert _parse_bool("on", False) is True
    assert _parse_bool("0", True) is False
    assert _parse_bool("off", True) is False
    assert _parse_bool("unknown", True) is True


def test_settings_preinclude_model_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("POECODER_HOME", str(tmp_path))
    monkeypatch.delenv("POECODER_MODELS", raising=False)
    monkeypatch.delenv("POECODER_OPENAI_MODELS", raising=False)

    settings = get_settings()
    assert "gpt-5.2-codex" in settings.supported_models
    assert "openai/gpt-4.1-mini" in settings.supported_models
    assert "openai/o4-mini" in settings.openai_models
