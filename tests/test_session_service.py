from __future__ import annotations

from poecoder.db import Database
from poecoder.models import SessionCreateRequest
from poecoder.services.session_service import SessionService


def test_default_context_selection_excludes_tool_results_and_model_output(tmp_path) -> None:
    db = Database(tmp_path / "db.sqlite3")
    sessions = SessionService(db=db, default_model="assistant")
    session = sessions.create(SessionCreateRequest(mode="coding"))

    sessions.put_context(session.id, "last_user_prompt", "check current directory", scope="pinned")
    sessions.put_context(session.id, "last_model_output", "the directory has 30 files", scope="pinned")
    sessions.put_context(session.id, "tool:ListFile", {"entries": ["a.py", "b.py"]}, scope="pinned")
    sessions.put_context(session.id, "repo_note", "project is in python", scope="pinned")

    selected, diagnostics = sessions.select_context_for_prompt(
        session_id=session.id,
        prompt="check current directory",
        max_items=20,
        max_value_chars=10000,
    )

    assert "last_user_prompt" in selected
    assert "repo_note" in selected
    assert "last_model_output" not in selected
    assert "tool:ListFile" not in selected
    assert int(diagnostics["candidate_items"]) == 2
