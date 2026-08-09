from __future__ import annotations

import os
import sys
from pathlib import Path


def _binary_path() -> Path:
    return Path(__file__).resolve().parent / "_bin" / "gaffa_search"


def main() -> None:
    binary = _binary_path()
    if not binary.is_file():
        raise RuntimeError(f"gaffa_search executable is missing from the package: {binary}")
    os.execv(binary, [str(binary), *sys.argv[1:]])
