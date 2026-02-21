from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CodeTools:
    root: Path

    def read_raw(self, file: str, line: int = 1, end_line: int | None = None) -> dict[str, Any]:
        path = self._resolve(file)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        start_idx = max(line - 1, 0)
        if end_line is None:
            end_idx = min(start_idx + 40, len(lines))
        else:
            end_idx = min(end_line, len(lines))
        selected = lines[start_idx:end_idx]
        return {
            "file": str(path),
            "line_start": start_idx + 1,
            "line_end": end_idx,
            "content": "\n".join(selected),
        }

    def write_raw(self, file: str, line: int, content: str, append: bool = False) -> dict[str, Any]:
        path = self._resolve(file)
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []
        if append:
            lines.append(content)
        else:
            idx = max(line - 1, 0)
            while len(lines) < idx:
                lines.append("")
            lines.insert(idx, content)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"file": str(path), "line": line, "append": append}

    def write_replace(self, pattern: str, replacement: str, location: str = ".", max_changes: int = 0) -> dict[str, Any]:
        base = self._resolve(location)
        regex = re.compile(pattern, re.MULTILINE)
        changed_files = 0
        changes = 0
        for path in self._walk_files(base):
            text = path.read_text(encoding="utf-8")
            new_text, count = regex.subn(replacement, text, count=max_changes if max_changes > 0 else 0)
            if count > 0:
                path.write_text(new_text, encoding="utf-8")
                changed_files += 1
                changes += count
        return {"changed_files": changed_files, "changes": changes}

    def search(self, pattern: str, file_pattern: str = "*", boundary: int = 2, root: str = ".") -> list[dict[str, Any]]:
        base = self._resolve(root)
        regex = re.compile(pattern)
        hits: list[dict[str, Any]] = []
        for path in self._walk_files(base, file_pattern=file_pattern):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for idx, line in enumerate(lines):
                if not regex.search(line):
                    continue
                start = max(idx - boundary, 0)
                end = min(idx + boundary + 1, len(lines))
                hits.append(
                    {
                        "file": str(path),
                        "line": idx + 1,
                        "match": line,
                        "context": "\n".join(lines[start:end]),
                    }
                )
        return hits

    def read_struct(self, target: str, language: str, dependency_depth: int = 1) -> dict[str, Any]:
        path = self._resolve(target)
        text = path.read_text(encoding="utf-8", errors="replace")
        if language == "python":
            return self._read_struct_python(path, text, dependency_depth)
        return self._read_struct_js_ts(path, text, dependency_depth)

    def read_recursive(self, seed_files: list[str], boundary: int = 2) -> dict[str, Any]:
        queue = [self._resolve(item) for item in seed_files]
        visited: set[Path] = set()
        graph: dict[str, Any] = {}

        while queue:
            current = queue.pop(0)
            if current in visited or not current.exists():
                continue
            visited.add(current)
            text = current.read_text(encoding="utf-8", errors="replace")
            imports = self._extract_imports(current, text)
            graph[str(current)] = {
                "imports": imports,
                "preview": self.read_raw(str(current), 1, min(1 + boundary * 8, 80)).get("content", ""),
            }
            for imp in imports:
                maybe = self._resolve_relative_import(current.parent, imp)
                if maybe and maybe.exists() and maybe not in visited:
                    queue.append(maybe)

        return {"files": graph, "count": len(graph)}

    def _read_struct_python(self, path: Path, text: str, dependency_depth: int) -> dict[str, Any]:
        tree = ast.parse(text)
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({"name": node.name, "line": node.lineno, "args": [a.arg for a in node.args.args]})
            elif isinstance(node, ast.ClassDef):
                classes.append({"name": node.name, "line": node.lineno})
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        deps = self._collect_python_deps(path, imports, dependency_depth)
        return {
            "file": str(path),
            "language": "python",
            "functions": sorted(functions, key=lambda x: x["line"]),
            "classes": sorted(classes, key=lambda x: x["line"]),
            "imports": sorted(set(imports)),
            "dependencies": deps,
        }

    def _read_struct_js_ts(self, path: Path, text: str, dependency_depth: int) -> dict[str, Any]:
        import_re = re.compile(r"^\s*import\s+.*?from\s+[\"']([^\"']+)[\"']", re.MULTILINE)
        fn_re = re.compile(r"^\s*(?:export\s+)?function\s+([A-Za-z0-9_]+)", re.MULTILINE)
        class_re = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z0-9_]+)", re.MULTILINE)
        imports = import_re.findall(text)
        deps = self._collect_generic_deps(path, imports, dependency_depth)
        return {
            "file": str(path),
            "language": "javascript" if path.suffix == ".js" else "typescript",
            "functions": [{"name": name} for name in fn_re.findall(text)],
            "classes": [{"name": name} for name in class_re.findall(text)],
            "imports": sorted(set(imports)),
            "dependencies": deps,
        }

    def _collect_python_deps(self, base: Path, imports: list[str], depth: int) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        frontier = [base]
        visited: set[Path] = set()
        current_depth = 0

        while frontier and current_depth < max(depth, 1):
            nxt: list[Path] = []
            for file in frontier:
                if file in visited or not file.exists():
                    continue
                visited.add(file)
                text = file.read_text(encoding="utf-8", errors="replace")
                file_imports = self._extract_imports(file, text)
                out[str(file)] = file_imports
                for imp in file_imports:
                    maybe = self._resolve_relative_import(file.parent, imp)
                    if maybe and maybe not in visited:
                        nxt.append(maybe)
            frontier = nxt
            current_depth += 1

        if not out:
            out[str(base)] = imports
        return out

    def _collect_generic_deps(self, base: Path, imports: list[str], depth: int) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {str(base): imports}
        if depth <= 1:
            return out
        frontier = [base]
        visited = {base}
        for _ in range(depth - 1):
            nxt: list[Path] = []
            for file in frontier:
                content = file.read_text(encoding="utf-8", errors="replace")
                deps = self._extract_imports(file, content)
                out[str(file)] = deps
                for dep in deps:
                    maybe = self._resolve_relative_import(file.parent, dep)
                    if maybe and maybe.exists() and maybe not in visited:
                        visited.add(maybe)
                        nxt.append(maybe)
            frontier = nxt
            if not frontier:
                break
        return out

    def _extract_imports(self, path: Path, text: str) -> list[str]:
        imports: list[str] = []
        if path.suffix == ".py":
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("import "):
                    imports.extend(part.strip() for part in line.replace("import", "", 1).split(","))
                elif line.startswith("from ") and " import " in line:
                    imports.append(line.split(" import ", 1)[0].replace("from", "", 1).strip())
            return imports

        import_re = re.compile(r"import\s+.*?from\s+[\"']([^\"']+)[\"']")
        return import_re.findall(text)

    def _resolve_relative_import(self, base: Path, target: str) -> Path | None:
        if target.startswith("."):
            candidate = (base / target).resolve()
        else:
            candidate = (self.root / target.replace(".", "/")).resolve()

        options = [candidate, candidate.with_suffix(".py"), candidate.with_suffix(".js"), candidate.with_suffix(".ts")]
        for option in options:
            if option.exists() and option.is_file() and self._is_within_root(option):
                return option
        return None

    def _walk_files(self, base: Path, file_pattern: str = "*"):
        if base.is_file() and fnmatch.fnmatch(base.name, file_pattern):
            yield base
            return
        for path in base.rglob("*"):
            if path.is_file() and fnmatch.fnmatch(path.name, file_pattern):
                yield path

    def _resolve(self, file: str) -> Path:
        path = Path(file)
        if not path.is_absolute():
            path = (self.root / path).resolve()
        if not self._is_within_root(path):
            raise ValueError(f"path outside root: {path}")
        return path

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root.resolve())
            return True
        except ValueError:
            return False
