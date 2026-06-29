import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
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
    requirements: Optional[dict[str, object]] = None
    runner: str = "utopic_runner"
    native_status: str = "ready"
    supported_backends: tuple[str, ...] = ("metal", "cuda", "cpu")
    artifact_filenames: tuple[str, ...] = ("output",)
    expected_vram_gib: Optional[float] = None
    expected_ram_gib: Optional[float] = None
    oom_policy: dict[str, object] = field(default_factory=dict)
    native_library: Optional[str] = None
    native_entrypoint: Optional[str] = None
    artifact_urls: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        if self.runtime == "bridge" or _native_uses_artifact_directory(self):
            return models_dir() / _safe_cache_name(self.id)
        return models_dir() / _safe_model_filename(self)


VALID_MODALITIES = {"text", "image", "tts", "music", "video", "misc"}
VALID_RUNTIMES = {"native", "bridge"}
VALID_NATIVE_STATUSES = {"ready", "planned", "experimental", "unsupported_on_device"}


def _safe_model_filename(entry: ModelEntry) -> str:
    filename = entry.filename
    _validate_safe_filename(filename, f"unsafe model filename for '{entry.id}'")
    return filename


def _validate_safe_filename(filename: str, message: str) -> None:
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise RuntimeError(f"{message}: {filename}")


def _safe_artifact_filename(entry: ModelEntry, filename: str) -> str:
    _validate_safe_filename(filename, f"unsafe artifact filename for '{entry.id}'")
    return filename


def _native_uses_artifact_directory(entry: ModelEntry) -> bool:
    return entry.runtime == "native" and (entry.native_library is not None or bool(entry.artifact_urls))


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
    runner = _string_field(item, "runner", "utopic_runner", index)
    native_status = _string_field(item, "native_status", "ready", index)
    supported_backends = _string_list_field(item, "supported_backends", ["metal", "cuda", "cpu"], index)
    artifact_filenames = _string_list_field(item, "artifact_filenames", [item["filename"]], index)
    expected_vram_gib = _optional_positive_number_field(item, "expected_vram_gib", index)
    expected_ram_gib = _optional_positive_number_field(item, "expected_ram_gib", index)
    oom_policy = item.get("oom_policy", {})
    native_library = item.get("native_library")
    native_entrypoint = item.get("native_entrypoint")
    artifact_urls = item.get("artifact_urls", {})
    repo = item.get("repo")
    if repo is not None and not isinstance(repo, str):
        raise RuntimeError(f"Invalid model catalog entry {index}: repo must be a string")
    requirements = item.get("requirements")
    if requirements is not None and not isinstance(requirements, dict):
        raise RuntimeError(f"Invalid model catalog entry {index}: requirements must be an object")
    if isinstance(requirements, dict):
        _validate_requirements(requirements, index)
    if modality not in VALID_MODALITIES:
        raise RuntimeError(f"Invalid model catalog entry {index}: modality must be one of {sorted(VALID_MODALITIES)}")
    if runtime not in VALID_RUNTIMES:
        raise RuntimeError(f"Invalid model catalog entry {index}: runtime must be one of {sorted(VALID_RUNTIMES)}")
    if native_status not in VALID_NATIVE_STATUSES:
        raise RuntimeError(
            f"Invalid model catalog entry {index}: native_status must be one of {sorted(VALID_NATIVE_STATUSES)}"
        )
    if not isinstance(oom_policy, dict):
        raise RuntimeError(f"Invalid model catalog entry {index}: oom_policy must be an object")
    if native_library is not None and not isinstance(native_library, str):
        raise RuntimeError(f"Invalid model catalog entry {index}: native_library must be a string")
    if native_entrypoint is not None and not isinstance(native_entrypoint, str):
        raise RuntimeError(f"Invalid model catalog entry {index}: native_entrypoint must be a string")
    if native_library is not None and not native_library:
        raise RuntimeError(f"Invalid model catalog entry {index}: native_library must be non-empty")
    if native_entrypoint is not None and not native_entrypoint:
        raise RuntimeError(f"Invalid model catalog entry {index}: native_entrypoint must be non-empty")
    if native_entrypoint is not None and native_library is None:
        raise RuntimeError(f"Invalid model catalog entry {index}: native_entrypoint requires native_library")
    if not isinstance(artifact_urls, dict):
        raise RuntimeError(f"Invalid model catalog entry {index}: artifact_urls must be an object")
    _validate_artifact_urls(artifact_urls, artifact_filenames, index)
    if runtime == "native" and native_library is None and not item["filename"].lower().endswith(".gguf"):
        raise RuntimeError(
            f"Invalid model catalog entry {index}: native GGUF models must use a GGUF filename"
        )
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
        requirements=dict(requirements) if isinstance(requirements, dict) else None,
        runner=runner,
        native_status=native_status,
        supported_backends=tuple(supported_backends),
        artifact_filenames=tuple(artifact_filenames),
        expected_vram_gib=expected_vram_gib,
        expected_ram_gib=expected_ram_gib,
        oom_policy=dict(oom_policy),
        native_library=native_library,
        native_entrypoint=native_entrypoint,
        artifact_urls=dict(artifact_urls),
    )


def _validate_artifact_urls(artifact_urls: dict[object, object], artifact_filenames: list[str], index: int) -> None:
    if not artifact_urls:
        return
    filenames = set(artifact_filenames)
    if set(artifact_urls) != filenames:
        raise RuntimeError(
            f"Invalid model catalog entry {index}: artifact_urls must include exactly artifact_filenames"
        )
    for filename, url in artifact_urls.items():
        if not isinstance(filename, str) or not filename:
            raise RuntimeError(f"Invalid model catalog entry {index}: artifact_urls keys must be non-empty strings")
        if not isinstance(url, str) or not url:
            raise RuntimeError(f"Invalid model catalog entry {index}: artifact_urls values must be non-empty strings")
        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError as exc:
            raise RuntimeError(f"Invalid model catalog entry {index}: artifact_urls.{filename} must be a URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(f"Invalid model catalog entry {index}: artifact_urls.{filename} must be an HTTP URL")


def _validate_requirements(requirements: dict[str, object], index: int) -> None:
    for key in ("min_gpu_memory_gib", "preferred_gpu_memory_gib"):
        value = requirements.get(key)
        if value is not None and not (
            isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        ):
            raise RuntimeError(f"Invalid model catalog entry {index}: requirements.{key} must be a positive number")
    allow_cpu = requirements.get("allow_cpu")
    if allow_cpu is not None and not isinstance(allow_cpu, bool):
        raise RuntimeError(f"Invalid model catalog entry {index}: requirements.allow_cpu must be a boolean")


def _string_field(item: dict[str, object], field: str, default: str, index: int) -> str:
    value = item.get(field, default)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Invalid model catalog entry {index}: {field} must be a non-empty string")
    return value


def _optional_positive_number_field(item: dict[str, object], field: str, index: int) -> Optional[float]:
    if field not in item:
        return None
    value = item[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"Invalid model catalog entry {index}: {field} must be a positive number")
    return float(value)


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


def _is_nonempty_dir(path: Path) -> bool:
    return path.is_dir() and any(child.is_file() for child in path.rglob("*"))


def is_model_downloaded(entry: ModelEntry) -> bool:
    if entry.runtime == "bridge":
        return (entry.path / "utopic-model.json").is_file()
    if _native_uses_artifact_directory(entry):
        return all(_native_artifact_present(entry, filename) for filename in entry.artifact_filenames)
    if not _is_nonempty_file(entry.path):
        return False
    if entry.bytes is None:
        return True
    return entry.path.stat().st_size == entry.bytes


def _native_artifact_cache_path(entry: ModelEntry, filename: str) -> Path:
    return entry.path / _safe_artifact_filename(entry, filename)


def _native_artifact_present(entry: ModelEntry, filename: str) -> bool:
    path = _native_artifact_cache_path(entry, filename)
    if _artifact_url_is_hf_tree(entry.artifact_urls.get(filename, "")):
        return _is_nonempty_dir(path)
    return _is_nonempty_file(path)


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
    if entry.modality == "misc":
        return "artifact"
    return "input" if entry.modality == "tts" else "prompt"


def _bridge_model_metadata(entry: ModelEntry) -> dict[str, object]:
    adapter = bridge.ADAPTERS.get(entry.engine)
    payload: dict[str, object] = {
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
        "runner": entry.runner,
        "native_status": entry.native_status,
        "supported_backends": list(entry.supported_backends),
        "artifact_filenames": list(entry.artifact_filenames),
        "expected_vram_gib": entry.expected_vram_gib,
        "expected_ram_gib": entry.expected_ram_gib,
        "oom_policy": entry.oom_policy,
        "url": entry.url,
    }
    if entry.native_library:
        payload["native_library"] = entry.native_library
    if entry.native_entrypoint:
        payload["native_entrypoint"] = entry.native_entrypoint
    if entry.requirements:
        payload["requirements"] = entry.requirements
    return payload


def readiness_metadata(entry: ModelEntry) -> dict[str, object]:
    payload: dict[str, object] = {
        "native_status": entry.native_status,
        "runner": entry.runner,
        "supported_backends": list(entry.supported_backends),
        "artifact_filenames": list(entry.artifact_filenames),
        "expected_vram_gib": entry.expected_vram_gib,
        "expected_ram_gib": entry.expected_ram_gib,
        "oom_policy": entry.oom_policy,
    }
    if entry.native_library:
        payload["native_library"] = entry.native_library
    if entry.native_entrypoint:
        payload["native_entrypoint"] = entry.native_entrypoint
    if entry.artifact_urls:
        payload["artifact_urls"] = entry.artifact_urls
    return payload


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

    if _native_uses_artifact_directory(entry):
        return _pull_native_artifact_model(entry, force=force)

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


def _pull_native_artifact_model(entry: ModelEntry, *, force: bool = False) -> Path:
    destination = entry.path
    if destination.exists() and is_model_downloaded(entry) and not force:
        return destination
    if destination.exists() and not destination.is_dir():
        _remove_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in entry.artifact_filenames:
        if filename not in entry.artifact_urls:
            raise RuntimeError(f"Native model '{entry.id}' is missing artifact URL for {filename}")
        target = _native_artifact_cache_path(entry, filename)
        artifact_url = entry.artifact_urls[filename]
        if target.exists() and _native_artifact_present(entry, filename) and not force:
            continue
        if _artifact_url_is_hf_tree(artifact_url):
            _pull_native_hf_tree_artifact(entry, filename, artifact_url, target, force=force)
            continue
        tmp = target.with_name(target.name + ".partial")
        if tmp.exists():
            _remove_path(tmp)
        try:
            print(f"Pulling {entry.name} artifact {filename} from Hugging Face")
            print(artifact_url)
            _copy_stream_with_progress(artifact_url, tmp)
            if tmp.stat().st_size == 0:
                raise OSError("downloaded 0 bytes")
            tmp.replace(target)
        except Exception as exc:
            if tmp.exists():
                _remove_path(tmp)
            raise RuntimeError(f"Failed to pull {entry.id} artifact {filename}: {exc}") from exc
    return destination


def _artifact_url_is_hf_tree(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    return parsed.netloc == "huggingface.co" and "/tree/" in parsed.path and parsed.path.startswith("/api/models/")


def _pull_native_hf_tree_artifact(
    entry: ModelEntry,
    filename: str,
    url: str,
    target: Path,
    *,
    force: bool,
) -> None:
    if target.exists() and force:
        _remove_path(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Pulling {entry.name} artifact tree {filename} from Hugging Face")
        print(url)
        for item in _fetch_hf_tree_listing(url):
            if not isinstance(item, dict) or item.get("type") != "file" or not isinstance(item.get("path"), str):
                continue
            remote_path = item["path"]
            local_path = _hf_tree_local_path(target, filename, remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if local_path.exists() and _is_nonempty_file(local_path) and not force:
                continue
            tmp = local_path.with_name(local_path.name + ".partial")
            if tmp.exists():
                _remove_path(tmp)
            _copy_stream_with_progress(_hf_tree_resolve_url(url, remote_path), tmp)
            if tmp.stat().st_size == 0:
                raise OSError(f"downloaded 0 bytes for {remote_path}")
            tmp.replace(local_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to pull {entry.id} artifact tree {filename}: {exc}") from exc


def _fetch_hf_tree_listing(url: str) -> list[object]:
    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise OSError("Hugging Face tree API did not return a JSON list")
    return payload


def _hf_tree_parts(url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    try:
        tree_index = parts.index("tree")
    except ValueError as exc:
        raise OSError(f"invalid Hugging Face tree URL: {url}") from exc
    if len(parts) < tree_index + 2 or parts[:2] != ["api", "models"]:
        raise OSError(f"invalid Hugging Face tree URL: {url}")
    repo = "/".join(parts[2:tree_index])
    revision = parts[tree_index + 1]
    if not repo or not revision:
        raise OSError(f"invalid Hugging Face tree URL: {url}")
    return repo, revision


def _hf_tree_resolve_url(tree_url: str, remote_path: str) -> str:
    repo, revision = _hf_tree_parts(tree_url)
    quoted_path = urllib.parse.quote(remote_path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{quoted_path}"


def _hf_tree_local_path(target: Path, artifact_name: str, remote_path: str) -> Path:
    normalized_remote = Path(remote_path)
    if normalized_remote.is_absolute() or any(part == ".." for part in normalized_remote.parts):
        raise OSError(f"unsafe Hugging Face tree path: {remote_path}")
    parts = normalized_remote.parts
    if not parts or parts[0] != artifact_name:
        raise OSError(f"Hugging Face tree path {remote_path} is outside artifact {artifact_name}")
    relative_parts = parts[1:]
    if not relative_parts:
        raise OSError(f"Hugging Face tree path {remote_path} does not name a file")
    return target.joinpath(*relative_parts)


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
    artifact_cache = _native_artifact_cache(entry)
    present = _is_nonempty_file(path) if not artifact_cache else all(
        bool(value["present"]) for value in artifact_cache.values()
    )
    size = path.stat().st_size if path.is_file() else sum(
        int(value["size"]) for value in artifact_cache.values()
    )
    expected_size = entry.bytes
    ready = present and (expected_size is None or size == expected_size)
    return {
        "id": entry.id,
        "name": entry.name,
        "runtime": entry.runtime,
        "modality": entry.modality,
        "engine": entry.engine,
        **readiness_metadata(entry),
        "status": "ready" if ready else "missing_model_file",
        "ready": ready,
        "requirements": entry.requirements or {},
        "cache": {
            "path": str(path),
            "present": present,
            "size": size,
            "expected_size": expected_size,
            "artifacts": artifact_cache,
        },
        "next_steps": [] if ready else [f"utopic models pull {entry.id}"],
    }


def _native_artifact_cache(entry: ModelEntry) -> dict[str, dict[str, object]]:
    if not _native_uses_artifact_directory(entry):
        return {}
    cache: dict[str, dict[str, object]] = {}
    for filename in entry.artifact_filenames:
        path = _native_artifact_cache_path(entry, filename)
        present = _is_nonempty_file(path)
        cache[filename] = {
            "path": str(path),
            "present": present,
            "size": path.stat().st_size if present else 0,
            "url": entry.artifact_urls.get(filename),
        }
    return cache


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
        **readiness_metadata(entry),
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "requirements": entry.requirements or {},
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
