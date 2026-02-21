from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def loads(value: str) -> Any:
    return json.loads(value)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)
