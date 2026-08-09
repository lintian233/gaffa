from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_elf(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def dynamic_symbols(path: Path, *, undefined: bool) -> set[str]:
    command = ["nm", "-D"]
    command.append("--undefined-only" if undefined else "--defined-only")
    command.append(str(path))
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    symbols = set()
    for line in output.splitlines():
        fields = line.split()
        if fields:
            symbols.add(fields[-1].split("@", maxsplit=1)[0])
    return symbols


def find_runtime_library(root: Path, soname: str) -> Path:
    candidates = [path for path in root.rglob(f"{soname}*") if path.is_file() and is_elf(path)]
    if not candidates:
        fail(f"CUDA runtime baseline is missing {soname} below {root}")
    return min(candidates, key=lambda path: (len(path.name), str(path)))


def verify_cuda_runtime_symbols(elf_paths: list[Path], runtime_root: Path) -> None:
    contracts = {
        "cuda": "libcudart.so.12",
        "cufft": "libcufft.so.11",
        "curand": "libcurand.so.10",
    }
    undefined: set[str] = set()
    for path in elf_paths:
        undefined.update(dynamic_symbols(path, undefined=True))
    for prefix, soname in contracts.items():
        required = {symbol for symbol in undefined if symbol.startswith(prefix)}
        runtime_library = find_runtime_library(runtime_root, soname)
        provided = dynamic_symbols(runtime_library, undefined=False)
        missing = required - provided
        if missing:
            fail(
                f"wheel requires symbols absent from the minimum-runtime {soname}: "
                f"{sorted(missing)}"
            )


def verify(wheel: Path, cuda_runtime_root: Path | None) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        required_patterns = {
            "core extension": r"^gaffa/_core.*\.so$",
            "Loki extension": r"^gaffa/_loki.*\.so$",
            "packaged CLI": r"^gaffa/_bin/gaffa_search$",
            "Loki runtime": r"^(gaffa/\.libs|gaffa\.libs)/.*libloki.*\.so(?:\.0)?$",
            "Loki license": r"^gaffa/licenses/LOKI_LICENSE$",
        }
        for description, pattern in required_patterns.items():
            if not any(re.match(pattern, name) for name in names):
                fail(f"wheel is missing {description}")

        forbidden_cuda = (
            "libcuda.so",
            "libcudart.so",
            "libcufft.so",
            "libcurand.so",
        )
        bundled_cuda = [
            name for name in names if any(token in Path(name).name for token in forbidden_cuda)
        ]
        if bundled_cuda:
            fail(f"wheel unexpectedly bundles CUDA libraries: {bundled_cuda}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive.extractall(root)
            cuda_dependencies: set[str] = set()
            elf_paths: list[Path] = []
            for path in root.rglob("*"):
                if not path.is_file() or not is_elf(path):
                    continue
                elf_paths.append(path)
                dynamic = subprocess.run(
                    ["readelf", "--dynamic", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                for line in dynamic.splitlines():
                    if ("RPATH" in line or "RUNPATH" in line) and (
                        "/home/" in line or "/opt/loki" in line
                    ):
                        fail(f"non-relocatable runtime path in {path.name}: {line.strip()}")
                    if "NEEDED" in line:
                        for dependency in (
                            "libcuda.so.1",
                            "libcudart.so.12",
                            "libcufft.so.11",
                            "libcurand.so.10",
                        ):
                            if dependency in line:
                                cuda_dependencies.add(dependency)

            expected_cuda = {
                "libcuda.so.1",
                "libcudart.so.12",
                "libcufft.so.11",
                "libcurand.so.10",
            }
            missing = expected_cuda - cuda_dependencies
            if missing:
                fail(f"wheel does not expose the expected CUDA runtime contract: {sorted(missing)}")
            if cuda_runtime_root is not None:
                verify_cuda_runtime_symbols(elf_paths, cuda_runtime_root)

    print(f"verified wheel: {wheel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--cuda-runtime-root",
        type=Path,
        help="check CUDA imports against libraries below this minimum-runtime root",
    )
    arguments = parser.parse_args()
    verify(
        arguments.wheel.resolve(),
        arguments.cuda_runtime_root.resolve() if arguments.cuda_runtime_root is not None else None,
    )
