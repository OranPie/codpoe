from __future__ import annotations

import html as html_lib
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx


class WebTools:
    def __init__(self, download_root: Path) -> None:
        self.download_root = download_root.resolve()
        self.download_root.mkdir(parents=True, exist_ok=True)

    async def get_web_raw(
        self,
        url: str,
        timeout_s: int = 20,
        max_chars: int = 200_000,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        body = resp.text[:max_chars]
        return {
            "url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": content_type,
            "body": body,
            "truncated": len(resp.text) > max_chars,
        }

    async def get_web(
        self,
        url: str,
        focus: str | None = None,
        timeout_s: int = 20,
        max_chars: int = 16_000,
    ) -> dict[str, object]:
        raw = await self.get_web_raw(url=url, timeout_s=timeout_s, max_chars=max_chars * 3)
        text = self._to_text(str(raw["body"]))
        title = self._extract_title(str(raw["body"]))

        if focus:
            needle = focus.lower()
            lines = [line for line in text.splitlines() if needle in line.lower()]
            compact = "\n".join(lines)
            text_out = compact[:max_chars] if compact else text[:max_chars]
        else:
            text_out = text[:max_chars]

        return {
            "url": raw["url"],
            "status_code": raw["status_code"],
            "title": title,
            "text": text_out,
            "truncated": len(text) > max_chars,
            "focus": focus,
        }

    async def get_web_file(
        self,
        url: str,
        save_as: str | None = None,
        folder: str = "downloads",
        overwrite: bool = False,
        timeout_s: int = 60,
        max_bytes: int = 20_000_000,
    ) -> dict[str, object]:
        out_dir = self._resolve_dir(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                filename = save_as or self._name_from_url(str(resp.url))
                target = out_dir / self._safe_name(filename)
                if target.exists() and not overwrite:
                    raise FileExistsError(f"destination exists: {target}")

                total = 0
                with target.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            fh.close()
                            target.unlink(missing_ok=True)
                            raise ValueError(f"download exceeds max_bytes={max_bytes}")
                        fh.write(chunk)

        return {
            "url": url,
            "saved_to": str(target),
            "bytes": total,
        }

    def _resolve_dir(self, folder: str) -> Path:
        path = Path(folder)
        if not path.is_absolute():
            path = (self.download_root / path).resolve()
        if not self._within_root(path):
            raise ValueError("folder outside download root")
        return path

    def _within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.download_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _name_from_url(url: str) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        return name if name else "download.bin"

    @staticmethod
    def _safe_name(name: str) -> str:
        name = name.replace("\x00", "")
        name = Path(name).name
        return name or "download.bin"

    @staticmethod
    def _extract_title(html_text: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return html_lib.unescape(" ".join(match.group(1).split()))

    @staticmethod
    def _to_text(html_text: str) -> str:
        no_script = re.sub(r"<script[^>]*>.*?</script>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
        no_style = re.sub(r"<style[^>]*>.*?</style>", " ", no_script, flags=re.IGNORECASE | re.DOTALL)
        no_tags = re.sub(r"<[^>]+>", " ", no_style)
        unescaped = html_lib.unescape(no_tags)
        normalized = re.sub(r"\s+", " ", unescaped).strip()
        return normalized
