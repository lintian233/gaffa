from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


_NEEDED_RE = re.compile(r"Shared library: \[([^]]+)\]")
_SYSTEM_PREFIXES = (
    "ld-linux",
    "libc.",
    "libdl.",
    "libgcc_s.",
    "libm.",
    "libpthread.",
    "librt.",
    "libstdc++.",
)
_CUDA_NAMES = {
    "libcuda.so.1",
    "libcudart.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
}


def needed_libraries(path: Path) -> list[str]:
    output = subprocess.run(
        ["readelf", "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return _NEEDED_RE.findall(output)


def is_external_runtime(name: str) -> bool:
    return name in _CUDA_NAMES or name.startswith(_SYSTEM_PREFIXES)


def find_runtime_library(runtime_dir: Path, name: str) -> Path:
    exact = runtime_dir / name
    if exact.is_file():
        return exact

    prefix = name.split(".so", maxsplit=1)[0]
    candidates = sorted(
        path
        for path in runtime_dir.glob(f"{prefix}-*")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"cannot resolve repaired runtime library {name!r}: {candidates}"
        )
    return candidates[0]


def copy_dependency_closure(
    source: Path,
    runtime_dir: Path,
    cli_lib_dir: Path,
    copied: set[Path],
) -> None:
    resolved_source = source.resolve()
    if resolved_source in copied:
        return
    copied.add(resolved_source)

    destination = cli_lib_dir / source.name
    if destination.resolve() != resolved_source:
        shutil.copy2(source, destination)

    for dependency in needed_libraries(source):
        if is_external_runtime(dependency):
            continue
        dependency_source = find_runtime_library(runtime_dir, dependency)
        copy_dependency_closure(
            dependency_source, runtime_dir, cli_lib_dir, copied
        )


def copy_cli_dependencies(raw_cli: Path, repaired_root: Path) -> None:
    runtime_dir = repaired_root / "gaffa.libs"
    cli_lib_dir = repaired_root / "gaffa" / ".libs"
    cli_lib_dir.mkdir(parents=True, exist_ok=True)

    # libloki already carries the repaired relative RPATH to gaffa.libs. The
    # CLI itself needs aliases for its original, pre-auditwheel NEEDED names.
    for dependency in needed_libraries(raw_cli):
        if is_external_runtime(dependency) or dependency == "libloki.so.0":
            continue
        source = find_runtime_library(runtime_dir, dependency)
        copy_dependency_closure(source, runtime_dir, cli_lib_dir, set())
        shutil.copy2(source, cli_lib_dir / dependency)


def file_record(path: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return path, "sha256=" + digest.rstrip(b"=").decode("ascii"), str(len(data))


def rebuild_record(entries: dict[str, bytes], record_path: str) -> bytes:
    rows = [file_record(name, data) for name, data in sorted(entries.items())
            if name != record_path]
    rows.append((record_path, "", ""))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def zip_mode(name: str) -> int:
    return 0o755 if name == "gaffa/_bin/gaffa_search" else 0o644


def patch_wheel(repaired_wheel: Path, raw_wheel: Path) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        repaired_root = root / "repaired"
        raw_root = root / "raw"
        repaired_root.mkdir()
        raw_root.mkdir()
        with zipfile.ZipFile(repaired_wheel) as archive:
            archive.extractall(repaired_root)
        with zipfile.ZipFile(raw_wheel) as archive:
            archive.extractall(raw_root)

        raw_cli = raw_root / "gaffa" / "_bin" / "gaffa_search"
        repaired_cli = repaired_root / "gaffa" / "_bin" / "gaffa_search"
        if not raw_cli.is_file() or not repaired_cli.is_file():
            raise RuntimeError("both wheels must contain gaffa/_bin/gaffa_search")

        copy_cli_dependencies(raw_cli, repaired_root)
        shutil.copy2(raw_cli, repaired_cli)

        record_path = next(repaired_root.glob("*.dist-info/RECORD"))
        entries: dict[str, bytes] = {}
        for path in repaired_root.rglob("*"):
            if path.is_file():
                entries[path.relative_to(repaired_root).as_posix()] = path.read_bytes()
        entries[record_path.relative_to(repaired_root).as_posix()] = b""
        record_name = record_path.relative_to(repaired_root).as_posix()
        entries[record_name] = rebuild_record(entries, record_name)

        temporary_wheel = repaired_wheel.with_suffix(".tmp.whl")
        with zipfile.ZipFile(
            temporary_wheel,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, data in sorted(entries.items()):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (0o100000 | zip_mode(name)) << 16
                archive.writestr(info, data)
        temporary_wheel.replace(repaired_wheel)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: restore_cli_binary.py REPAIRED_WHEEL RAW_WHEEL")
    patch_wheel(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())


if __name__ == "__main__":
    main()
