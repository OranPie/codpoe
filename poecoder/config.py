from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PRESET_MODELS = [
    "assistant",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5",
    "claude-sonnet-4.5",
    "claude-opus-4.1",
    "gemini-2.5-pro",
]

DEFAULT_PRESET_OPENAI_MODELS = [
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/o4-mini",
]


@dataclass(slots=True)
class Settings:
    home_dir: Path
    db_path: Path
    poe_api_url: str
    poe_api_key: str | None
    openai_api_url: str
    openai_api_key: str | None
    openai_models: list[str]
    default_small_model: str
    default_large_model: str
    default_thinking_level: str
    default_thinking_budget: int
    default_show_think_details: bool
    reviewer_model: str
    reviewer_thinking_level: str
    reviewer_thinking_budget: int
    host: str
    port: int
    supported_models: list[str]
    lang: str



def _parse_models(raw: str, defaults: list[str]) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        items = defaults
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _normalize_openai_models(raw: str) -> list[str]:
    items = _parse_models(raw, [])
    out: list[str] = []
    for item in items:
        name = item.strip()
        if not name:
            continue
        if name.startswith("oa:"):
            name = f"openai/{name.split(':', 1)[1]}"
        if not name.startswith("openai/"):
            name = f"openai/{name}"
        if name not in out:
            out.append(name)
    return out


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def get_settings() -> Settings:
    home = Path(os.environ.get("POECODER_HOME", Path.home() / ".poecoder"))
    home.mkdir(parents=True, exist_ok=True)
    db_path = Path(os.environ.get("POECODER_DB_PATH", home / "poecoder.db"))
    default_small = os.environ.get("POECODER_SMALL_MODEL", "assistant")
    default_large = os.environ.get("POECODER_LARGE_MODEL", "gpt-5.2")
    default_thinking_level = os.environ.get("POECODER_THINKING_LEVEL", "balanced")
    default_thinking_budget = int(os.environ.get("POECODER_THINKING_BUDGET", "12000"))
    default_show_think_details = _parse_bool(os.environ.get("POECODER_SHOW_THINK_DETAILS"), False)
    reviewer_model = os.environ.get("POECODER_REVIEWER_MODEL", default_large)
    reviewer_thinking_level = os.environ.get("POECODER_REVIEWER_THINKING_LEVEL", "deep")
    reviewer_thinking_budget = int(os.environ.get("POECODER_REVIEWER_THINKING_BUDGET", "16000"))
    openai_models = _normalize_openai_models(os.environ.get("POECODER_OPENAI_MODELS", ""))
    merged_openai_models: list[str] = []
    for name in [*DEFAULT_PRESET_OPENAI_MODELS, *openai_models]:
        if name not in merged_openai_models:
            merged_openai_models.append(name)
    supported = _parse_models(
        os.environ.get("POECODER_MODELS", ""),
        [default_small, default_large, *DEFAULT_PRESET_MODELS, *merged_openai_models],
    )
    return Settings(
        db_path=db_path,
        poe_api_url=os.environ.get("POECODER_POE_API_URL", "https://api.poe.com/bot/"),
        poe_api_key=os.environ.get("POECODER_POE_API_KEY"),
        openai_api_url=os.environ.get("POECODER_OPENAI_API_URL", "https://api.openai.com/v1"),
        openai_api_key=os.environ.get("POECODER_OPENAI_API_KEY"),
        openai_models=merged_openai_models,
        default_small_model=default_small,
        default_large_model=default_large,
        default_thinking_level=default_thinking_level,
        default_thinking_budget=default_thinking_budget,
        default_show_think_details=default_show_think_details,
        reviewer_model=reviewer_model,
        reviewer_thinking_level=reviewer_thinking_level,
        reviewer_thinking_budget=reviewer_thinking_budget,
        host=os.environ.get("POECODER_HOST", "127.0.0.1"),
        port=int(os.environ.get("POECODER_PORT", "8765")),
        supported_models=supported,
        lang=os.environ.get("POECODER_LANG", "en"),
        home_dir=home,
    )
