from __future__ import annotations

from poecoder.tools.code_tools import CodeTools


def test_read_struct_python(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(
        "import os\n"
        "from pkg import util\n"
        "\n"
        "class A:\n"
        "    pass\n"
        "\n"
        "def run(x):\n"
        "    return x\n",
        encoding="utf-8",
    )

    tools = CodeTools(root=tmp_path)
    data = tools.read_struct("sample.py", "python", dependency_depth=1)

    assert data["language"] == "python"
    assert any(item["name"] == "A" for item in data["classes"])
    assert any(item["name"] == "run" for item in data["functions"])
    assert "os" in data["imports"]


def test_read_struct_js(tmp_path):
    f = tmp_path / "sample.ts"
    f.write_text(
        "import { a } from './dep'\n"
        "export function run() { return a }\n"
        "export class C {}\n",
        encoding="utf-8",
    )

    tools = CodeTools(root=tmp_path)
    data = tools.read_struct("sample.ts", "typescript", dependency_depth=1)

    assert data["language"] == "typescript"
    assert any(item["name"] == "run" for item in data["functions"])
    assert any(item["name"] == "C" for item in data["classes"])


def test_search_and_replace(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("value = 1\nvalue = 2\n", encoding="utf-8")
    tools = CodeTools(root=tmp_path)

    hits = tools.search("value", "*.py", boundary=0)
    assert len(hits) == 2

    out = tools.write_replace(r"value", "item", location=".")
    assert out["changes"] == 2
    assert "item = 1" in f.read_text(encoding="utf-8")
