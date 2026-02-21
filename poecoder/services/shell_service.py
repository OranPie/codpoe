from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from poecoder.models import ShellRunResponse
from poecoder.policy import PolicyEngine
from poecoder.services.session_service import SessionService


@dataclass(slots=True)
class ShellService:
    policy: PolicyEngine
    sessions: SessionService

    async def run(self, session_id: str, command: str, danger_level: int, cwd: str | None, timeout_s: int) -> ShellRunResponse:
        session = self.sessions.get(session_id)
        decision = self.policy.check_shell(session.policy_profile, command, danger_level)
        if not decision.allowed:
            return ShellRunResponse(allowed=False, policy_reason=decision.reason)

        workdir = Path(cwd) if cwd else Path.cwd()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            return ShellRunResponse(
                allowed=True,
                exit_code=proc.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                policy_reason=decision.reason,
            )
        except asyncio.TimeoutError:
            return ShellRunResponse(
                allowed=True,
                exit_code=124,
                stdout="",
                stderr="timed out",
                policy_reason=decision.reason,
            )
