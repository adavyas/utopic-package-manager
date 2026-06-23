import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from . import bridge
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
    bytes: Optional[int] = None
    modality: str = "text"
    engine: str = "native-gguf"
    runtime: str = "native"
    hardware: tuple[str, ...] = ("local",)
    endpoints: tuple[str, ...] = ("/v1/chat/completions",)
    outputs: tuple[str, ...] = ("text",)
    repo: Optional[str] = None

    @property
    def path(self) -> Path:
        if self.runtime == "bridge":
            return models_dir() / _safe_cache_name(self.id)
        return models_dir() / _safe_model_filename(self)


VALID_MODALITIES = {"text", "image", "tts", "music", "video"}
VALID_RUNTIMES = {"native", "bridge"}


def _safe_model_filename(entry: ModelEntry) -> str:
    filename = entry.filename
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise RuntimeError(f"unsafe model filename for '{entry.id}': {filename}")
    return filename


def _safe_cache_name(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise RuntimeError(f"unsafe model cache name: {value}")
    return value


def _validate_model_url(entry: ModelEntry) -> None:
    try:
        parsed = urllib.parse.urlsplit(entry.url)
    except ValueError as exc:
        raise RuntimeError(f"model URL for '{entry.id}' must be a URL") from exc
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"unsupported model URL protocol for '{entry.id}': {parsed.scheme or '<missing>'}")
    if not parsed.netloc:
        raise RuntimeError(f"model URL for '{entry.id}' must include a host")


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
    path = catalog_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read model catalog {path}: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Model catalog {path} must contain a JSON list")
    if not data:
        raise RuntimeError("Utopic model catalog is empty")
    return [_validate_catalog_entry(item, index) for index, item in enumerate(data)]


def _validate_catalog_entry(item: object, index: int) -> ModelEntry:
    if not isinstance(item, dict):
        raise RuntimeError(f"Invalid model catalog entry {index}: expected a JSON object")

    for field in ("id", "name", "family", "filename", "url", "size", "description"):
        if not isinstance(item.get(field), str):
            raise RuntimeError(f"Invalid model catalog entry {index}: {field} must be a string")
    if not isinstance(item.get("recommended"), bool):
        raise RuntimeError(f"Invalid model catalog entry {index}: recommended must be a boolean")
    if "bytes" in item and not (
        isinstance(item["bytes"], int) and not isinstance(item["bytes"], bool) and item["bytes"] > 0
    ):
        raise RuntimeError(f"Invalid model catalog entry {index}: bytes must be a positive integer")
    modality = _string_field(item, "modality", "text", index)
    engine = _string_field(item, "engine", "native-gguf", index)
    runtime = _string_field(item, "runtime", "native", index)
    hardware = _string_list_field(item, "hardware", ["local"], index)
    endpoints = _string_list_field(item, "endpoints", ["/v1/chat/completions"], index)
    outputs = _string_list_field(item, "outputs", ["text"], index)
    repo = item.get("repo")
    if repo is not None and not isinstance(repo, str):
        raise RuntimeError(f"Invalid model catalog entry {index}: repo must be a string")
    if modality not in VALID_MODALITIES:
        raise RuntimeError(f"Invalid model catalog entry {index}: modality must be one of {sorted(VALID_MODALITIES)}")
    if runtime not in VALID_RUNTIMES:
        raise RuntimeError(f"Invalid model catalog entry {index}: runtime must be one of {sorted(VALID_RUNTIMES)}")
    if runtime == "native" and not item["filename"].lower().endswith(".gguf"):
        raise RuntimeError(f"Invalid model catalog entry {index}: native models must use a GGUF filename")
    if runtime == "bridge" and engine not in bridge.ADAPTERS:
        raise RuntimeError(f"Invalid model catalog entry {index}: unknown bridge engine: {engine}")

    return ModelEntry(
        id=item["id"],
        name=item["name"],
        family=item["family"],
        filename=item["filename"],
        url=item["url"],
        size=item["size"],
        recommended=item["recommended"],
        description=item["description"],
        bytes=item.get("bytes"),
        modality=modality,
        engine=engine,
        runtime=runtime,
        hardware=tuple(hardware),
        endpoints=tuple(endpoints),
        outputs=tuple(outputs),
        repo=repo,
    )


def _string_field(item: dict[str, object], field: str, default: str, index: int) -> str:
    value = item.get(field, default)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid model catalog entry {index}: {field} must be a non-empty string")
    return value


def _string_list_field(
    item: dict[str, object],
    field: str,
    default: list[str],
    index: int,
) -> list[str]:
    value = item.get(field, default)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(part, str) or not part for part in value)
    ):
        raise RuntimeError(f"Invalid model catalog entry {index}: {field} must be a non-empty string list")
    return value


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
        content_length = response.headers.get("content-length", "0") or "0"
        try:
            total = int(content_length)
        except ValueError as exc:
            raise OSError(f"invalid content-length: {content_length}") from exc
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


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_model_downloaded(entry: ModelEntry) -> bool:
    if entry.runtime == "bridge":
        return (entry.path / "utopic-model.json").is_file()
    if not _is_nonempty_file(entry.path):
        return False
    if entry.bytes is None:
        return True
    return entry.path.stat().st_size == entry.bytes


def _is_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == 0


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _bridge_command_env_var(entry: ModelEntry) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in entry.engine.upper()).strip("_")
    return f"UTOPIC_BRIDGE_{normalized}_COMMAND"


def _bridge_input_key(entry: ModelEntry) -> str:
    return "input" if entry.modality == "tts" else "prompt"


def _bridge_model_metadata(entry: ModelEntry) -> dict[str, object]:
    adapter = bridge.ADAPTERS.get(entry.engine)
    return {
        "bridge": {
            "command": f"utopic-bridge {entry.engine}",
            "environment_variable": _bridge_command_env_var(entry),
            "input": _bridge_input_key(entry),
            "install_hint": adapter.install_hint if adapter is not None else "",
            "schema_version": bridge.SCHEMA_VERSION,
        },
        "endpoints": list(entry.endpoints),
        "engine": entry.engine,
        "hardware": list(entry.hardware),
        "id": entry.id,
        "modality": entry.modality,
        "name": entry.name,
        "outputs": list(entry.outputs),
        "repo": entry.repo,
        "runtime": entry.runtime,
        "url": entry.url,
    }


def pull_model(model_id: str, *, force: bool = False) -> Path:
    entry = get_model(model_id)
    if entry is None:
        known = ", ".join(model.id for model in list_models())
        raise RuntimeError(f"Unknown Utopic model '{model_id}'. Known models: {known}")

    destination = entry.path
    if entry.runtime == "bridge":
        if destination.exists() and is_model_downloaded(entry) and not force:
            return destination
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "utopic-model.json").write_text(
            json.dumps(_bridge_model_metadata(entry), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    if destination.exists() and is_model_downloaded(entry) and not force:
        return destination

    _validate_model_url(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    remove_empty_destination_on_failure = _is_empty_file(destination)
    if destination.exists() and not destination.is_file():
        _remove_path(destination)
    if tmp.exists():
        _remove_path(tmp)

    try:
        print(f"Pulling {entry.name} from Hugging Face")
        print(entry.url)
        _copy_stream_with_progress(entry.url, tmp)
        downloaded_size = tmp.stat().st_size
        if downloaded_size == 0:
            raise OSError("downloaded 0 bytes")
        if entry.bytes is not None and downloaded_size != entry.bytes:
            raise OSError(f"downloaded {downloaded_size} of {entry.bytes} bytes")
        tmp.replace(destination)
    except Exception as exc:
        if tmp.exists():
            _remove_path(tmp)
        if (
            remove_empty_destination_on_failure
            and destination.exists()
            and _is_empty_file(destination)
        ):
            _remove_path(destination)
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
        status = "downloaded" if is_model_downloaded(entry) else "not downloaded"
        print(f"{marker} {entry.id:28} {entry.size:14} {entry.modality:6} {entry.runtime:6} {status}")
        print(f"  {entry.name}")
        print(f"  {entry.description}")


def _print_path(model_id: str) -> None:
    entry = get_model(model_id)
    if entry is None:
        raise RuntimeError(f"Unknown Utopic model '{model_id}'.")
    print(entry.path)


def _native_model_check(entry: ModelEntry) -> dict[str, object]:
    path = entry.path
    present = _is_nonempty_file(path)
    size = path.stat().st_size if present else 0
    expected_size = entry.bytes
    ready = present and (expected_size is None or size == expected_size)
    return {
        "id": entry.id,
        "name": entry.name,
        "runtime": entry.runtime,
        "modality": entry.modality,
        "engine": entry.engine,
        "status": "ready" if ready else "missing_model_file",
        "ready": ready,
        "cache": {
            "path": str(path),
            "present": present,
            "size": size,
            "expected_size": expected_size,
        },
        "next_steps": [] if ready else [f"utopic models pull {entry.id}"],
    }


def _bridge_model_check(entry: ModelEntry) -> dict[str, object]:
    prepared = is_model_downloaded(entry)
    adapter = bridge.ADAPTERS.get(entry.engine)
    if adapter is None:
        bridge_check: dict[str, object] = {
            "schema_version": bridge.SCHEMA_VERSION,
            "engine": entry.engine,
            "status": "unknown_engine",
            "ready": False,
            "packages": [],
            "missing": [],
            "install_hint": "",
            "description": "",
        }
    else:
        bridge_check = bridge._check_adapter(adapter)
    bridge_ready = bool(bridge_check.get("ready"))
    ready = prepared and bridge_ready
    next_steps: list[str] = []
    if not prepared:
        next_steps.append(f"utopic models pull {entry.id}")
    install_hint = bridge_check.get("install_hint")
    if not bridge_ready and isinstance(install_hint, str) and install_hint:
        next_steps.append(install_hint)
    return {
        "id": entry.id,
        "name": entry.name,
        "runtime": entry.runtime,
        "modality": entry.modality,
        "engine": entry.engine,
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "cache": {
            "path": str(entry.path),
            "prepared": prepared,
            "metadata_path": str(entry.path / "utopic-model.json"),
        },
        "bridge": bridge_check,
        "next_steps": next_steps,
    }


def model_check(model_id: str) -> dict[str, object]:
    entry = get_model(model_id)
    if entry is None:
        raise RuntimeError(f"Unknown Utopic model '{model_id}'.")
    if entry.runtime == "bridge":
        return _bridge_model_check(entry)
    return _native_model_check(entry)


def _print_check(model_id: str) -> bool:
    payload = model_check(model_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return bool(payload.get("ready"))


def _all_model_checks() -> dict[str, object]:
    checks = [model_check(entry.id) for entry in list_models()]
    ready_count = sum(1 for item in checks if item.get("ready"))
    not_ready_count = len(checks) - ready_count
    return {
        "object": "utopic.model_check.list",
        "ready": not_ready_count == 0,
        "summary": {
            "ready": ready_count,
            "not_ready": not_ready_count,
            "total": len(checks),
        },
        "data": checks,
    }


def _print_check_all() -> bool:
    payload = _all_model_checks()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return bool(payload.get("ready"))


def _pull_all_models(*, force: bool = False) -> dict[str, object]:
    data = []
    for entry in list_models():
        with contextlib.redirect_stdout(io.StringIO()):
            path = pull_model(entry.id, force=force)
        data.append(
            {
                "id": entry.id,
                "path": str(path),
                "runtime": entry.runtime,
                "modality": entry.modality,
            }
        )
    return {"object": "utopic.model_pull.list", "data": data}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if any(arg == "--version" for arg in args_list):
        print(f"utopic models {__version__}")
        return 0

    parser = argparse.ArgumentParser(
        prog="utopic models",
        description="List, download, and locate curated Utopic GGUF models.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List curated Utopic model aliases.")

    pull = subparsers.add_parser("pull", help="Download a curated model by alias.")
    pull.add_argument("model", nargs="?", help="Model alias. Defaults to the recommended model.")
    pull.add_argument("--all", action="store_true", help="Pull or prepare every catalog model.")
    pull.add_argument("--force", action="store_true", help="Redownload even if the model exists locally.")

    path = subparsers.add_parser("path", help="Print the local path for a model alias.")
    path.add_argument("model", help="Model alias.")

    check = subparsers.add_parser("check", help="Print model cache and runtime readiness as JSON.")
    check.add_argument("model", nargs="?", help="Model alias.")
    check.add_argument("--all", action="store_true", help="Check every catalog model.")

    args = parser.parse_args(args_list)
    command = args.command or "list"

    try:
        if command == "list":
            _print_models()
            return 0
        if command == "pull":
            if args.all:
                if args.model:
                    raise RuntimeError("pull accepts either a model alias or --all, not both")
                print(json.dumps(_pull_all_models(force=args.force), indent=2, sort_keys=True))
                return 0
            model_id = args.model or default_model().id
            print(pull_model(model_id, force=args.force))
            return 0
        if command == "path":
            _print_path(args.model)
            return 0
        if command == "check":
            if args.all:
                return 0 if _print_check_all() else 1
            if not args.model:
                raise RuntimeError("check requires a model alias or --all")
            return 0 if _print_check(args.model) else 1
    except RuntimeError as exc:
        print(f"utopic models: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2
