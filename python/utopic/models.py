import argparse
import json
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from . import installer


PACKAGE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = PACKAGE_DIR / "models.json"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    family: str
    filename: str
    url: str
    size: str
    recommended: bool
    description: str

    @property
    def path(self) -> Path:
        return models_dir() / self.filename


def models_dir() -> Path:
    configured = os.environ.get("UTOPIC_MODELS_DIR")
    if configured:
        return Path(configured).expanduser()
    return installer.cache_root() / "models"


def catalog_path() -> Path:
    configured = os.environ.get("UTOPIC_MODELS_CATALOG")
    if configured:
        return Path(configured).expanduser()
    return CATALOG_PATH


def _load_catalog() -> list[ModelEntry]:
    data = json.loads(catalog_path().read_text(encoding="utf-8"))
    return [ModelEntry(**item) for item in data]


def list_models() -> list[ModelEntry]:
    return _load_catalog()


def get_model(model_id: str) -> Optional[ModelEntry]:
    for entry in list_models():
        if entry.id == model_id:
            return entry
    return None


def default_model() -> ModelEntry:
    catalog = list_models()
    for entry in catalog:
        if entry.recommended:
            return entry
    if not catalog:
        raise RuntimeError("Utopic model catalog is empty.")
    return catalog[0]


def _copy_stream_with_progress(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("content-length", "0") or "0")
        downloaded = 0
        with destination.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    print(f"\rDownloading {destination.name}: {percent:3d}%", end="", flush=True)
        if total:
            print()
            if downloaded != total:
                raise OSError(f"downloaded {downloaded} of {total} bytes")


def pull_model(model_id: str, *, force: bool = False) -> Path:
    entry = get_model(model_id)
    if entry is None:
        known = ", ".join(model.id for model in list_models())
        raise RuntimeError(f"Unknown Utopic model '{model_id}'. Known models: {known}")

    destination = entry.path
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    try:
        print(f"Pulling {entry.name} from Hugging Face")
        print(entry.url)
        _copy_stream_with_progress(entry.url, tmp)
        shutil.move(str(tmp), str(destination))
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to pull {entry.id} from {entry.url}: {exc}") from exc
    return destination


def resolve_model(value: Optional[str]) -> Path:
    if not value:
        entry = default_model()
        return pull_model(entry.id)

    possible_path = Path(value).expanduser()
    if (
        possible_path.exists()
        or possible_path.suffix.lower() == ".gguf"
        or "/" in value
        or "\\" in value
    ):
        return possible_path

    return pull_model(value)


def ensure_model(value: Optional[str] = None) -> Path:
    return resolve_model(value)


def _print_models() -> None:
    for entry in list_models():
        marker = "*" if entry.recommended else " "
        status = "downloaded" if entry.path.exists() and entry.path.stat().st_size > 0 else "not downloaded"
        print(f"{marker} {entry.id:24} {entry.size:14} {status}")
        print(f"  {entry.name}")
        print(f"  {entry.description}")


def _print_path(model_id: str) -> None:
    entry = get_model(model_id)
    if entry is None:
        raise RuntimeError(f"Unknown Utopic model '{model_id}'.")
    print(entry.path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="utopic models",
        description="List, download, and locate curated Utopic GGUF models.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List curated Utopic model aliases.")

    pull = subparsers.add_parser("pull", help="Download a curated model by alias.")
    pull.add_argument("model", nargs="?", help="Model alias. Defaults to the recommended model.")
    pull.add_argument("--force", action="store_true", help="Redownload even if the model exists locally.")

    path = subparsers.add_parser("path", help="Print the local path for a model alias.")
    path.add_argument("model", help="Model alias.")

    args = parser.parse_args(list(argv) if argv is not None else None)
    command = args.command or "list"

    try:
        if command == "list":
            _print_models()
            return 0
        if command == "pull":
            model_id = args.model or default_model().id
            print(pull_model(model_id, force=args.force))
            return 0
        if command == "path":
            _print_path(args.model)
            return 0
    except RuntimeError as exc:
        print(f"utopic models: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2
