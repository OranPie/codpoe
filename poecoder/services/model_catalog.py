from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx


@dataclass(slots=True)
class ModelCatalog:
    supported_models: list[str]
    api_key: str | None = None
    models_url: str = "https://api.poe.com/v1/models"
    cache_ttl_s: int = 300
    _last_fetch_at: float = field(default=0.0, init=False)

    def list_models(self, refresh: bool = False) -> list[str]:
        self._refresh_from_remote(force=refresh)
        return list(self.supported_models)

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
        if not force and (now - self._last_fetch_at) < self.cache_ttl_s and self.supported_models:
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
                self._last_fetch_at = now
        except Exception:
            # Keep local fallback model list if remote fetch fails.
            self._last_fetch_at = now
