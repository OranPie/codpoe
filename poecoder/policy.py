from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from typing import Iterable


DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-rf\s+/\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
)


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    reason: str


class PolicyEngine:
    def __init__(self) -> None:
        self.command_signing_secret = os.environ.get("POECODER_COMMAND_SIGNING_SECRET")

    def check_shell(self, profile: str, command: str, danger_level: int) -> PolicyDecision:
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(command):
                return PolicyDecision(False, "hard-veto dangerous system command")

        profile = profile.lower()
        if profile == "strict" and danger_level > 0:
            return PolicyDecision(False, "strict profile only allows danger level 0")

        if profile == "default" and danger_level > 1:
            return PolicyDecision(False, "default profile blocks danger level 2")

        if profile in {"autonomous", "research"}:
            return PolicyDecision(True, "autonomous profile allows requested danger level")

        return PolicyDecision(True, "allowed by policy")

    def check_capabilities(self, capabilities: Iterable[str]) -> PolicyDecision:
        blocked = {"kernel", "rootfs"}
        overlap = blocked.intersection(capabilities)
        if overlap:
            return PolicyDecision(False, f"capabilities blocked: {','.join(sorted(overlap))}")
        return PolicyDecision(True, "capabilities allowed")

    def signature_status(self, payload: str, signature: str | None) -> str:
        if not signature:
            return "unsigned"
        if not self.command_signing_secret:
            return "unverified"
        expected = hmac.new(
            self.command_signing_secret.encode("utf-8"),
            payload.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return "verified" if hmac.compare_digest(signature, expected) else "invalid"
