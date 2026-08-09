from __future__ import annotations

import os
from pathlib import Path

import pytest

from gaffa import _cli


def test_binary_path_honours_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = Path("/tmp/gaffa-search-test")
    monkeypatch.setenv("GAFFA_SEARCH_BINARY", str(binary))

    assert _cli._binary_candidates()[0] == binary


def test_main_executes_packaged_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    binary = Path("/tmp/gaffa/_bin/gaffa_search")
    monkeypatch.setattr(_cli, "_binary_path", lambda: binary)
    monkeypatch.setattr(Path, "is_file", lambda self: self == binary)
    monkeypatch.setattr(os, "access", lambda path, mode: path == binary)
    monkeypatch.setattr("sys.argv", ["gaffa_search", "--help"])

    invocation: tuple[Path, list[str]] | None = None

    def fake_execv(path: Path, arguments: list[str]) -> None:
        nonlocal invocation
        invocation = (path, arguments)

    monkeypatch.setattr(os, "execv", fake_execv)
    _cli.main()

    assert invocation == (binary, [str(binary), "--help"])


def test_main_rejects_missing_packaged_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    binary = Path("/tmp/gaffa/_bin/gaffa_search")
    monkeypatch.setattr(_cli, "_binary_path", lambda: binary)

    with pytest.raises(RuntimeError, match="executable is missing"):
        _cli.main()
