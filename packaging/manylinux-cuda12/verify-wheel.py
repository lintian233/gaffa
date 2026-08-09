from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_elf(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def verify(wheel: Path) -> None:
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
            for path in root.rglob("*"):
                if not path.is_file() or not is_elf(path):
                    continue
                dynamic = subprocess.run(
                    ["readelf", "--dynamic", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                for line in dynamic.splitlines():
                    if "RPATH" in line or "RUNPATH" in line:
                        if "/home/" in line or "/opt/loki" in line:
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

    print(f"verified wheel: {wheel}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        fail("usage: verify-wheel.py WHEEL")
    verify(Path(sys.argv[1]).resolve())
