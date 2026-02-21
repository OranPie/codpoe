from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx


@dataclass(slots=True)
class UsageService:
    api_key: str | None
    balance_url: str = "https://api.poe.com/usage/current_balance"

    def get_current_balance(self) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("POE API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(self.balance_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        balance = data.get("current_point_balance")
        if not isinstance(balance, int):
            raise ValueError("balance response missing current_point_balance")

        return {
            "current_point_balance": balance,
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "source": self.balance_url,
        }
