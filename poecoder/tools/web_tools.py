from __future__ import annotations

import html as html_lib
import re
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from xml.etree import ElementTree as ET

import httpx

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore[assignment]


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
        selector: str | None = None,
        regex: str | None = None,
        max_matches: int = 60,
    ) -> dict[str, object]:
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        source = resp.text
        filter_meta: dict[str, object] = {"mode": "none", "parser": "none", "matches": 0}
        if selector:
            source, filter_meta = self._filter_with_selector(source, selector, max_matches=max_matches)
        elif regex:
            source, filter_meta = self._filter_with_regex(source, regex, max_matches=max_matches)
        body = source[:max_chars]
        hint = ""
        if not selector and not regex and len(resp.text) > max_chars * 2:
            hint = "Large page; prefer selector/regex filtering or use GetWebFile then read locally."
        return {
            "url": str(resp.url),
            "status_code": resp.status_code,
            "content_type": content_type,
            "title": self._extract_title(resp.text),
            "body": body,
            "truncated": len(source) > max_chars,
            "filter": filter_meta,
            "hint": hint,
        }

    async def search_web(
        self,
        query: str,
        limit: int = 8,
        timeout_s: int = 20,
        max_snippet_chars: int = 280,
    ) -> dict[str, object]:
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("SearchWeb requires query")
        limit = max(1, min(int(limit), 30))
        endpoint = f"https://duckduckgo.com/html/?q={quote_plus(clean_query)}"
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.get(endpoint, headers={"user-agent": "PoeCoder/0.1"})
            resp.raise_for_status()
        results = self._parse_duckduckgo_html(resp.text, limit=limit, max_snippet_chars=max_snippet_chars)
        return {
            "query": clean_query,
            "source": "duckduckgo-html",
            "url": endpoint,
            "count": len(results),
            "results": results,
        }

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 8,
        timeout_s: int = 20,
    ) -> dict[str, object]:
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("SearchArxiv requires query")
        max_results = max(1, min(int(max_results), 50))
        endpoint = (
            "https://export.arxiv.org/api/query"
            f"?search_query=all:{quote_plus(clean_query)}&start=0&max_results={max_results}"
        )
        async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
            resp = await client.get(endpoint, headers={"user-agent": "PoeCoder/0.1"})
            resp.raise_for_status()
        results = self._parse_arxiv_xml(resp.text, max_results=max_results)
        return {
            "query": clean_query,
            "source": "arxiv-api",
            "url": endpoint,
            "count": len(results),
            "results": results,
        }

    async def download_urls(
        self,
        urls: list[str],
        folder: str = "downloads",
        overwrite: bool = False,
        timeout_s: int = 60,
        max_bytes: int = 20_000_000,
        max_files: int = 8,
    ) -> dict[str, object]:
        results: list[dict[str, object]] = []
        selected = [url.strip() for url in urls if isinstance(url, str) and url.strip()][: max(1, min(max_files, 40))]
        for url in selected:
            try:
                saved = await self.get_web_file(
                    url=url,
                    folder=folder,
                    overwrite=overwrite,
                    timeout_s=timeout_s,
                    max_bytes=max_bytes,
                )
                results.append({"url": url, "ok": True, "result": saved})
            except Exception as exc:  # noqa: BLE001
                results.append({"url": url, "ok": False, "error": str(exc)})
        return {
            "count": len(results),
            "success": len([item for item in results if item.get("ok") is True]),
            "results": results,
        }

    async def get_web(
        self,
        url: str,
        focus: str | None = None,
        timeout_s: int = 20,
        max_chars: int = 16_000,
        selector: str | None = None,
        regex: str | None = None,
        max_matches: int = 60,
        download_if_large: bool = False,
        download_folder: str = "downloads",
    ) -> dict[str, object]:
        raw = await self.get_web_raw(
            url=url,
            timeout_s=timeout_s,
            max_chars=max_chars * 3,
            selector=selector,
            regex=regex,
            max_matches=max_matches,
        )
        text = self._to_text(str(raw["body"]))
        title = str(raw.get("title", "")) or self._extract_title(str(raw["body"]))

        if focus:
            needle = focus.lower()
            lines = [line for line in text.splitlines() if needle in line.lower()]
            compact = "\n".join(lines)
            text_out = compact[:max_chars] if compact else text[:max_chars]
        else:
            text_out = text[:max_chars]

        result = {
            "url": raw["url"],
            "status_code": raw["status_code"],
            "title": title,
            "text": text_out,
            "truncated": len(text) > max_chars,
            "focus": focus,
            "filter": raw.get("filter", {}),
            "hint": raw.get("hint", ""),
        }
        if download_if_large and bool(raw.get("truncated")):
            try:
                downloaded = await self.get_web_file(url=url, folder=download_folder, overwrite=False, timeout_s=max(30, timeout_s))
                result["downloaded_file"] = downloaded
            except Exception as exc:  # noqa: BLE001
                result["download_error"] = str(exc)
        return result

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

    def _filter_with_selector(self, html_text: str, selector: str, max_matches: int = 60) -> tuple[str, dict[str, object]]:
        clean_selector = selector.strip()
        if not clean_selector:
            return html_text, {"mode": "none", "parser": "none", "matches": 0}

        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_text, "html.parser")
            nodes = soup.select(clean_selector)
            chunks = [node.get_text(" ", strip=True) for node in nodes[:max_matches]]
            chunks = [chunk for chunk in chunks if chunk]
            return "\n".join(chunks), {
                "mode": "selector",
                "parser": "bs4",
                "selector": clean_selector,
                "matches": len(nodes),
                "returned_matches": len(chunks),
            }

        # regex fallback for simple selectors when bs4 isn't installed
        if clean_selector.startswith("."):
            cls = re.escape(clean_selector[1:])
            pattern = rf"<([a-zA-Z0-9:_-]+)\b[^>]*class=[\"'][^\"']*\b{cls}\b[^\"']*[\"'][^>]*>(.*?)</\1>"
        elif clean_selector.startswith("#"):
            ident = re.escape(clean_selector[1:])
            pattern = rf"<([a-zA-Z0-9:_-]+)\b[^>]*id=[\"']{ident}[\"'][^>]*>(.*?)</\1>"
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9:_-]*$", clean_selector):
            tag = re.escape(clean_selector)
            pattern = rf"<{tag}\b[^>]*>(.*?)</{tag}>"
        else:
            return "", {
                "mode": "selector",
                "parser": "regex-fallback",
                "selector": clean_selector,
                "matches": 0,
                "warning": "bs4 unavailable and selector is too complex; use regex or GetWebFile.",
            }

        matches = re.findall(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        snippets = matches[:max_matches]
        clean = [self._to_text(item[1] if isinstance(item, tuple) and len(item) > 1 else str(item)) for item in snippets]
        clean = [item for item in clean if item]
        return "\n".join(clean), {
            "mode": "selector",
            "parser": "regex-fallback",
            "selector": clean_selector,
            "matches": len(matches),
            "returned_matches": len(clean),
        }

    def _filter_with_regex(self, source: str, pattern: str, max_matches: int = 60) -> tuple[str, dict[str, object]]:
        compiled = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
        found = list(compiled.finditer(source))
        selected = found[:max_matches]
        chunks: list[str] = []
        for match in selected:
            if match.groups():
                groups = [item for item in match.groups() if item]
                chunks.extend(groups or [match.group(0)])
            else:
                chunks.append(match.group(0))
        compact = "\n".join(self._to_text(chunk) for chunk in chunks if chunk)
        return compact, {
            "mode": "regex",
            "parser": "regex",
            "regex": pattern,
            "matches": len(found),
            "returned_matches": len(chunks),
        }

    def _parse_duckduckgo_html(self, html_text: str, *, limit: int = 8, max_snippet_chars: int = 280) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_text, "html.parser")
            for node in soup.select("div.result")[:limit]:
                title_node = node.select_one("a.result__a")
                if title_node is None:
                    continue
                title = self._normalize_text(title_node.get_text(" ", strip=True))
                href = str(title_node.get("href") or "").strip()
                snippet_node = node.select_one(".result__snippet")
                snippet = self._normalize_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
                if len(snippet) > max_snippet_chars:
                    snippet = snippet[: max_snippet_chars - 3].rstrip() + "..."
                if title and href:
                    out.append({"title": title, "url": href, "snippet": snippet})
            return out

        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html_text):
            href = html_lib.unescape(match.group(1)).strip()
            title = self._normalize_text(self._to_text(match.group(2)))
            if title and href:
                out.append({"title": title, "url": href, "snippet": ""})
            if len(out) >= limit:
                break
        return out

    def _parse_arxiv_xml(self, xml_text: str, *, max_results: int = 8) -> list[dict[str, object]]:
        try:
            root = ET.fromstring(xml_text)
        except Exception:  # noqa: BLE001
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: list[dict[str, object]] = []
        for entry in root.findall("a:entry", ns)[:max_results]:
            title = self._normalize_text(entry.findtext("a:title", default="", namespaces=ns))
            summary = self._normalize_text(entry.findtext("a:summary", default="", namespaces=ns))
            published = self._normalize_text(entry.findtext("a:published", default="", namespaces=ns))
            entry_id = self._normalize_text(entry.findtext("a:id", default="", namespaces=ns))
            authors = [
                self._normalize_text(author.findtext("a:name", default="", namespaces=ns))
                for author in entry.findall("a:author", ns)
            ]
            authors = [name for name in authors if name]
            pdf_url = ""
            for link in entry.findall("a:link", ns):
                rel = str(link.attrib.get("rel", "")).strip().lower()
                href = str(link.attrib.get("href", "")).strip()
                typ = str(link.attrib.get("type", "")).strip().lower()
                if "pdf" in href.lower() or typ == "application/pdf" or rel == "related":
                    pdf_url = href
                    if "pdf" in href.lower():
                        break
            out.append(
                {
                    "title": title,
                    "summary": summary,
                    "published": published,
                    "id": entry_id,
                    "pdf_url": pdf_url,
                    "authors": authors,
                }
            )
        return out

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", html_lib.unescape(text or "")).strip()
