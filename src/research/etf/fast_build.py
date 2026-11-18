"""Linux/WSL M3 커널의 명시 빌드. import할 때 컴파일하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

SOURCE = Path(__file__).with_name("native") / "candidate.cpp"
LIBRARY = "libetf_candidate.so"
ABI_VERSION = 2


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_kernel(output_dir: Path, *, compiler="g++") -> Path:
    """새 폴더에 커널과 출처 기록을 작성한다. 기존 빌드는 덮어쓰지 않는다."""
    if platform.system() != "Linux":
        raise RuntimeError("M3 C++ 빌드는 Linux/WSL에서 실행하세요")
    executable = shutil.which(compiler)
    if executable is None:
        raise RuntimeError("C++17 컴파일러가 없습니다. g++와 libboost-dev를 준비하세요")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"기존 빌드 폴더는 덮어쓰지 않습니다: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="etf-m3-build-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        _compile(executable, staging)
        shutil.copytree(staging, output_dir)
    return output_dir


def _compile(compiler, staging):
    source = staging / SOURCE.name
    shutil.copyfile(SOURCE, source)
    command = [compiler, "-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared",
               "-Wall", "-Wextra", "-Werror", str(source), "-o", str(staging / LIBRARY)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode:
        raise RuntimeError(f"M3 C++ 빌드 실패({completed.returncode}):\n{completed.stderr}")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    manifest = {"abi": ABI_VERSION, "source_sha256": fingerprint(source),
                "library_sha256": fingerprint(staging / LIBRARY), "compiler": version,
                "machine": platform.machine(), "command": command}
    (staging / "build.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(build_kernel(arguments.output_dir))


if __name__ == "__main__":
    main()
