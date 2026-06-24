import json
import os
import subprocess
from pathlib import Path

import pytest

from utopic import _native


def _runner_binary(name: str = "utopic_runner") -> Path:
    value = os.environ.get("UTOPIC_RUNNER_BINARY")
    if value and name == "utopic_runner":
        path = Path(value).expanduser()
    else:
        try:
            path = _native.binary_path(name)
        except RuntimeError:
            pytest.skip(f"run utopic setup or set UTOPIC_RUNNER_BINARY=/path/to/utopic_runner")
    if not path.is_file():
        pytest.fail(f"{name} does not exist: {path}")
    return path


def _run_runner(runner: Path, request_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(runner), "--json-request", str(request_path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        assert isinstance(payload, dict)
        return payload
    raise AssertionError("native runner wrote no JSON response")


def test_native_runner_rejects_malformed_json(tmp_path):
    request_path = tmp_path / "bad-request.json"
    request_path.write_text("{not json", encoding="utf-8")

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "runner_failed"
    assert "invalid JSON request" in payload["error"]["message"]


def test_native_runner_accepts_contract_without_schema_version(tmp_path):
    request_path = tmp_path / "without-schema-version.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "missing_model"
    assert payload["error"]["message"] == "options.model_path is required for native chat"


@pytest.mark.parametrize(
    ("runner_request", "field", "message"),
    [
        (
            {
                "schema_version": "utopic-runner/v0",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
            },
            "schema_version",
            "unsupported schema_version",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
            },
            "task",
            "task is required",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
            },
            "model",
            "model is required",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "model": "unit-text",
                "input": "hello",
                "options": {},
                "output_dir": ".",
            },
            "input",
            "input must be an object",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": [],
                "output_dir": ".",
            },
            "options",
            "options must be an object",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
            },
            "output_dir",
            "output_dir is required",
        ),
    ],
)
def test_native_runner_rejects_incomplete_contract(tmp_path, runner_request, field, message):
    request_path = tmp_path / "incomplete-request.json"
    request_path.write_text(json.dumps(runner_request), encoding="utf-8")

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == message
    assert payload["error"]["detail"]["field"] == field
    assert payload["error"]["detail"]["schema_version"] == "utopic-runner/v1"


def test_native_runner_reports_missing_model_path_for_chat(tmp_path):
    request_path = tmp_path / "missing-model-path.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "missing_model"
    assert payload["error"]["message"] == "options.model_path is required for native chat"


def test_native_runner_rejects_unknown_task(tmp_path):
    request_path = tmp_path / "unknown-task.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "not-a-task",
                "model": "unit-model",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "task must be chat, image, tts, music, video, or misc"
    assert payload["error"]["detail"]["field"] == "task"


def test_native_runner_reports_planned_non_text_task_readiness(tmp_path):
    request_path = tmp_path / "planned-image.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "image",
                "model": "unit-image",
                "input": {"prompt": "a red cube"},
                "options": {
                    "modality": "image",
                    "engine": "diffusers",
                    "runtime": "planned_native",
                    "runner": "image_runner",
                    "native_status": "planned",
                    "supported_backends": ["metal", "cuda"],
                    "expected_vram_gib": 8.0,
                    "expected_ram_gib": 16.0,
                },
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["message"] == "native C++ runner task is not implemented yet"
    assert payload["error"]["detail"]["task"] == "image"
    assert payload["error"]["detail"]["model"] == "unit-image"
    assert payload["error"]["detail"]["modality"] == "image"
    assert payload["error"]["detail"]["engine"] == "diffusers"
    assert payload["error"]["detail"]["runtime"] == "planned_native"
    assert payload["error"]["detail"]["runner"] == "image_runner"
    assert payload["error"]["detail"]["native_status"] == "planned"
    assert payload["error"]["detail"]["supported_backends"] == ["metal", "cuda"]
    assert payload["error"]["detail"]["expected_vram_gib"] == 8.0
    assert payload["error"]["detail"]["expected_ram_gib"] == 16.0


def test_modality_runner_entrypoint_reports_planned_readiness(tmp_path):
    request_path = tmp_path / "planned-image.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "image",
                "model": "unit-image",
                "input": {"prompt": "a red cube"},
                "options": {
                    "modality": "image",
                    "engine": "diffusers",
                    "runtime": "planned_native",
                    "runner": "image_runner",
                    "native_status": "planned",
                    "supported_backends": ["metal", "cuda"],
                },
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary("image_runner"), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["detail"]["runner"] == "image_runner"
    assert payload["error"]["detail"]["task"] == "image"


def test_modality_runner_entrypoint_reports_its_own_name_without_runner_option(tmp_path):
    request_path = tmp_path / "planned-image-no-runner-option.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "image",
                "model": "unit-image",
                "input": {"prompt": "a red cube"},
                "options": {
                    "modality": "image",
                    "engine": "diffusers",
                    "runtime": "planned_native",
                    "native_status": "planned",
                    "supported_backends": ["metal", "cuda"],
                },
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary("image_runner"), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["detail"]["runner"] == "image_runner"
    assert payload["error"]["detail"]["task"] == "image"


def test_native_runner_reports_unloadable_model_cleanly(tmp_path):
    missing_model = tmp_path / "missing.gguf"
    request_path = tmp_path / "missing-model.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {"model_path": str(missing_model)},
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "missing_model"
    assert payload["error"]["message"] == "failed to load model"
    assert payload["error"]["detail"]["model"] == "unit-text"
    assert payload["error"]["detail"]["model_path"] == str(missing_model)


def test_native_runner_reports_unsupported_modality_with_readiness_detail(tmp_path):
    request_path = tmp_path / "image-request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "utopic-runner/v1",
                "task": "image",
                "model": "unit-image",
                "input": {"prompt": "a red cube"},
                "options": {
                    "modality": "image",
                    "engine": "diffusers",
                    "runtime": "planned_native",
                    "runner": "image_runner",
                    "native_status": "planned",
                    "supported_backends": ["metal", "cuda"],
                    "expected_vram_gib": 8.0,
                },
                "output_dir": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    detail = payload["error"]["detail"]
    assert detail["task"] == "image"
    assert detail["model"] == "unit-image"
    assert detail["runtime"] == "planned_native"
    assert detail["runner"] == "image_runner"
    assert detail["native_status"] == "planned"
    assert detail["supported_backends"] == ["metal", "cuda"]
    assert detail["expected_vram_gib"] == 8.0
