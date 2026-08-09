from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def _binary_candidates() -> list[Path]:
    candidates: list[Path] = []

    override = os.environ.get("GAFFA_SEARCH_BINARY")
    if override:
        candidates.append(Path(override).expanduser())

    package_dir = Path(__file__).resolve().parent
    candidates.append(package_dir / "_bin" / "gaffa_search")

    # Editable installs execute this source file, while scikit-build-core
    # installs the executable into the environment's package directory.
    for install_dir in (sysconfig.get_path("purelib"),
                        sysconfig.get_path("platlib")):
        if install_dir:
            candidates.append(Path(install_dir) / "gaffa" / "_bin" /
                              "gaffa_search")

    # This fallback is only for a source checkout. It keeps `just install`
    # and direct development builds usable without copying binaries into git.
    source_root = package_dir.parents[1]
    if (source_root / "CMakeLists.txt").is_file():
        for build_dir in (
            "editable-loki-debug",
            "editable-debug",
            "loki-release",
            "release",
            "dev",
        ):
            candidates.append(source_root / "build" / build_dir /
                              "gaffa_search")

    return candidates


def _binary_path() -> Path:
    for candidate in _binary_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return _binary_candidates()[0]


def main() -> None:
    binary = _binary_path()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(
            "gaffa_search executable is missing from the package: "
            f"{binary}"
        )
    os.execv(binary, [str(binary), *sys.argv[1:]])
