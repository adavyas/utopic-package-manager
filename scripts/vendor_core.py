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

    CORE_DIR.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("CMakeLists.txt", ".gitignore", "__pycache__", "*.pyc", "*.pyo")

    # The core native source is owned by this repository. Vendoring only refreshes
    # the Utopic Python control plane and built chat UI snapshot.
    python_dir = CORE_DIR / "python"
    if python_dir.exists():
        shutil.rmtree(python_dir)
    shutil.copytree(tmp / "python", python_dir, ignore=ignore)

    chat_dir = tmp / "chat"
    run(["npm", "ci"], cwd=chat_dir)
    run(["npm", "run", "build"], cwd=chat_dir)
    dist_chat = tmp / "chat" / "dist" / "utopic-chat.js"
    if not dist_chat.exists():
        raise FileNotFoundError(f"chat build did not produce {dist_chat}")
    node_dir = CORE_DIR / "python" / "utopic_core" / "node"
    node_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dist_chat, node_dir / "utopic-chat.js")

    shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
