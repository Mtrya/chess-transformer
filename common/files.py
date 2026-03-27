from pathlib import Path


def ensure_runtime_dirs() -> None:
    for dirname in ("log", "ckpts"):
        Path(dirname).mkdir(exist_ok=True)
