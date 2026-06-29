from __future__ import annotations
import hashlib
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify_registry(registry: Path, root: Path) -> list[str]:
    failures: list[str] = []
    for line in registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        candidate = root / relative.strip().lstrip("*")
        if not candidate.exists() or sha256_file(candidate) != expected:
            failures.append(relative.strip())
    return failures
