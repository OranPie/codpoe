from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    db_path: Path
    poe_api_url: str
    poe_api_key: str | None
    default_small_model: str
    default_large_model: str
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


def get_settings() -> Settings:
    home = Path(os.environ.get("POECODER_HOME", Path.home() / ".poecoder"))
    home.mkdir(parents=True, exist_ok=True)
    db_path = Path(os.environ.get("POECODER_DB_PATH", home / "poecoder.db"))
    default_small = os.environ.get("POECODER_SMALL_MODEL", "assistant")
    default_large = os.environ.get("POECODER_LARGE_MODEL", "gpt-5.2")
    supported = _parse_models(
        os.environ.get("POECODER_MODELS", ""),
        [default_small, default_large],
    )
    return Settings(
        db_path=db_path,
        poe_api_url=os.environ.get("POECODER_POE_API_URL", "https://api.poe.com/bot/"),
        poe_api_key=os.environ.get("POECODER_POE_API_KEY"),
        default_small_model=default_small,
        default_large_model=default_large,
        host=os.environ.get("POECODER_HOST", "127.0.0.1"),
        port=int(os.environ.get("POECODER_PORT", "8765")),
        supported_models=supported,
        lang=os.environ.get("POECODER_LANG", "en"),
    )
