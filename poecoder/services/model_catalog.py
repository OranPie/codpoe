from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx


@dataclass(slots=True)
class ModelCatalog:
    supported_models: list[str]
    api_key: str | None = None
    models_url: str = "https://api.poe.com/v1/models"
    openai_api_key: str | None = None
    openai_api_url: str = "https://api.openai.com/v1"
    cache_ttl_s: int = 300
    _last_fetch_poe_at: float = field(default=0.0, init=False)
    _last_fetch_openai_at: float = field(default=0.0, init=False)
    _last_poe_error: str | None = field(default=None, init=False)
    _last_openai_error: str | None = field(default=None, init=False)

    def list_models(self, refresh: bool = False) -> list[str]:
        self._refresh_from_remote(force=refresh)
        self._refresh_openai_from_remote(force=refresh)
        return list(self.supported_models)

    def provider_status(self) -> dict[str, object]:
        openai_models = [name for name in self.supported_models if isinstance(name, str) and name.startswith("openai/")]
        return {
            "poe": {
                "api_key_configured": bool(self.api_key),
                "models_url": self.models_url,
                "last_fetch_at": self._last_fetch_poe_at,
                "last_error": self._last_poe_error,
            },
            "openai": {
                "api_key_configured": bool(self.openai_api_key),
                "base_url": self.openai_api_url,
                "last_fetch_at": self._last_fetch_openai_at,
                "last_error": self._last_openai_error,
                "model_count": len(openai_models),
                "models_preview": openai_models[:10],
            },
        }

    def ensure_supported(self, model: str) -> None:
        if model == "auto":
            return
        models = self.list_models(refresh=False)
        if model not in models:
            raise ValueError(f"unsupported model: {model}")

    def _refresh_from_remote(self, force: bool = False) -> None:
        if not self.api_key:
            return
        now = time.time()
        if not force and (now - self._last_fetch_poe_at) < self.cache_ttl_s and self.supported_models:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(self.models_url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            models = []
            for item in data.get("data", []):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id and model_id not in models:
                    models.append(model_id)
            if models:
                merged: list[str] = []
                for item in [*self.supported_models, *models]:
                    if item and item not in merged:
                        merged.append(item)
                self.supported_models = merged
            self._last_fetch_poe_at = now
            self._last_poe_error = None
        except Exception as exc:
            # Keep local fallback model list if remote fetch fails.
            self._last_fetch_poe_at = now
            self._last_poe_error = str(exc)[:240] or "failed to fetch Poe models"

    def update_openai(self, api_key: str | None = None, base_url: str | None = None) -> None:
        changed = False
        if api_key is not None:
            if api_key != self.openai_api_key:
                changed = True
            self.openai_api_key = api_key
        if base_url is not None:
            normalized = self._normalize_openai_api_url(base_url)
            if normalized != self.openai_api_url:
                changed = True
            self.openai_api_url = normalized
        if changed:
            self._last_fetch_openai_at = 0.0

    def _refresh_openai_from_remote(self, force: bool = False) -> None:
        if not self.openai_api_key:
            return
        now = time.time()
        if not force and (now - self._last_fetch_openai_at) < self.cache_ttl_s and self.supported_models:
            return
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Accept": "application/json",
        }
        models_url = self.openai_api_url.rstrip("/") + "/models"
        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(models_url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            models: list[str] = []
            for item in data.get("data", []):
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id:
                    continue
                if not self._looks_like_text_model(model_id):
                    continue
                name = f"openai/{model_id}"
                if name not in models:
                    models.append(name)
            if models:
                merged: list[str] = []
                for item in [*self.supported_models, *models]:
                    if item and item not in merged:
                        merged.append(item)
                self.supported_models = merged
            self._last_fetch_openai_at = now
            self._last_openai_error = None
        except Exception as exc:
            self._last_fetch_openai_at = now
            self._last_openai_error = str(exc)[:240] or "failed to fetch OpenAI models"

    @staticmethod
    def _normalize_openai_api_url(api_url: str) -> str:
        base = (api_url or "").strip()
        if not base:
            return "https://api.openai.com/v1"
        return base.rstrip("/")

    @staticmethod
    def _looks_like_text_model(model_id: str) -> bool:
        normalized = model_id.strip().lower()
        if not normalized:
            return False
        prefixes = ("gpt-", "o", "chatgpt")
        return normalized.startswith(prefixes)
