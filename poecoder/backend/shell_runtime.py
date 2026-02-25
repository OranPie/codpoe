from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

from poecoder.backend.models import RunShellRequest, RunShellResult


@dataclass(slots=True)
class ShellRuntime:
    workspace_root: Path

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()

    async def run(self, req: RunShellRequest) -> RunShellResult:
        blocked = self._blocked_reason(req.command, req.danger_ack)
        if blocked:
            return RunShellResult(allowed=False, blocked_reason=blocked)

        cwd = self._resolve_cwd(req.cwd)
        if cwd is None:
            return RunShellResult(allowed=False, blocked_reason="cwd outside workspace")

        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                req.command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._safe_env(),
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=req.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return RunShellResult(
                    allowed=True,
                    exit_code=124,
                    stdout="",
                    stderr=f"command timed out after {req.timeout_s}s",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            return RunShellResult(
                allowed=True,
                exit_code=proc.returncode,
                stdout=out.decode("utf-8", errors="replace"),
                stderr=err.decode("utf-8", errors="replace"),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001
            return RunShellResult(
                allowed=True,
                exit_code=1,
                stdout="",
                stderr=f"shell runtime error: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

    def _resolve_cwd(self, raw: str) -> Path | None:
        candidate = Path(raw or ".")
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(self.workspace_root)
            return candidate
        except ValueError:
            return None

    @staticmethod
    def _blocked_reason(command: str, danger_ack: bool) -> str:
        text = (command or "").strip().lower()
        if not text:
            return "empty command"
        dangerous = (
            "rm -rf /",
            "shutdown",
            "reboot",
            "mkfs",
            "dd if=",
            "chmod -r 777 /",
            ":(){:|:&};:",
        )
        if any(token in text for token in dangerous) and not danger_ack:
            return "command requires danger_ack=true"
        return ""

    @staticmethod
    def _safe_env() -> dict[str, str]:
        keep = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM"}
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env
