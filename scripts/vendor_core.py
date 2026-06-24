from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "python" / "utopic" / "core"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=None if cwd is None else str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor a pinned Utopic core snapshot.")
    parser.add_argument("--repo", default="https://github.com/adavyas/utopic.git")
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    tmp = ROOT / ".vendor-core-tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    run(["git", "clone", args.repo, str(tmp)])
    run(["git", "checkout", args.ref], cwd=tmp)

    if CORE_DIR.exists():
        shutil.rmtree(CORE_DIR)
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(".gitignore", "__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(tmp / "native", CORE_DIR / "native", ignore=ignore)
    shutil.copytree(tmp / "python", CORE_DIR / "python", ignore=ignore)

    dist_chat = tmp / "chat" / "dist" / "utopic-chat.js"
    if dist_chat.exists():
        node_dir = CORE_DIR / "python" / "utopic_core" / "node"
        node_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dist_chat, node_dir / "utopic-chat.js")

    shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
