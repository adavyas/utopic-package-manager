import json
import os
import subprocess
from pathlib import Path

import pytest

from utopic import _native


def _runner_binary(name: str = "utopic-runner") -> Path:
    value = os.environ.get("UTOPIC_RUNNER_BINARY")
    if value and name == "utopic-runner":
        path = Path(value).expanduser()
    else:
        try:
            path = _native.binary_path(name)
        except RuntimeError:
            pytest.skip(f"run utopic setup or set UTOPIC_RUNNER_BINARY=/path/to/utopic-runner")
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


def _run_runner_with_env(
    runner: Path,
    request_path: Path,
    extra_env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.run(
        [str(runner), "--json-request", str(request_path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env=env,
    )


def test_native_runner_help_describes_task_contract():
    completed = subprocess.run(
        [str(_runner_binary()), "--help"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    text = completed.stdout + completed.stderr

    assert completed.returncode == 0
    assert "--json-request request.json" in text
    assert "schema_version=utopic-runner/v1" in text
    assert "chat: native GGUF text generation" in text
    assert "image, tts, music, video, misc: planned native tasks" in text
    assert "structured unsupported_model readiness errors" in text


def _last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        assert isinstance(payload, dict)
        return payload
    raise AssertionError("native runner wrote no JSON response")


def _contract_request(tmp_path: Path, payload: dict[str, object]) -> dict[str, object]:
    request: dict[str, object] = {
        "schema_version": "utopic-runner/v1",
        "run_id": "run_unit",
        "input": {},
        "options": {},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }
    request.update(payload)
    return request


def _progress_events(progress_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]


def test_native_runner_rejects_malformed_json(tmp_path):
    request_path = tmp_path / "bad-request.json"
    request_path.write_text("{not json", encoding="utf-8")

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "runner_failed"
    assert "invalid JSON request" in payload["error"]["message"]


def test_native_runner_rejects_contract_without_schema_version(tmp_path):
    request_path = tmp_path / "without-schema-version.json"
    request_path.write_text(
        json.dumps(
            {
                "run_id": "run_unit",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": str(tmp_path),
                "progress_path": str(tmp_path / "progress.jsonl"),
            }
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "runner_failed"
    assert payload["error"]["message"] == "schema_version is required"
    assert payload["error"]["detail"]["field"] == "schema_version"


@pytest.mark.parametrize(
    ("runner_request", "field", "message"),
    [
        (
            {
                "schema_version": "utopic-runner/v0",
                "run_id": "run_unit",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
                "progress_path": "./progress.jsonl",
            },
            "schema_version",
            "unsupported schema_version",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "run_id": "run_unit",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
                "progress_path": "./progress.jsonl",
            },
            "task",
            "task is required",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "run_id": "run_unit",
                "task": "chat",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
                "progress_path": "./progress.jsonl",
            },
            "model",
            "model is required",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
                "progress_path": "./progress.jsonl",
            },
            "run_id",
            "run_id is required",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "run_id": "run_unit",
                "task": "chat",
                "model": "unit-text",
                "input": "hello",
                "options": {},
                "output_dir": ".",
                "progress_path": "./progress.jsonl",
            },
            "input",
            "input must be an object",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "run_id": "run_unit",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": [],
                "output_dir": ".",
                "progress_path": "./progress.jsonl",
            },
            "options",
            "options must be an object",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "run_id": "run_unit",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "progress_path": "./progress.jsonl",
            },
            "output_dir",
            "output_dir is required",
        ),
        (
            {
                "schema_version": "utopic-runner/v1",
                "run_id": "run_unit",
                "task": "chat",
                "model": "unit-text",
                "input": {"prompt": "hello"},
                "options": {},
                "output_dir": ".",
            },
            "progress_path",
            "progress_path is required",
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
    assert payload["error"]["code"] == "runner_failed"
    assert payload["error"]["message"] == message
    assert payload["error"]["detail"]["field"] == field
    assert payload["error"]["detail"]["schema_version"] == "utopic-runner/v1"


def test_native_runner_reports_missing_model_path_for_chat(tmp_path):
    request_path = tmp_path / "missing-model-path.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "task": "chat",
                    "model": "unit-text",
                    "input": {"prompt": "hello"},
                    "options": {},
                },
            )
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
            _contract_request(
                tmp_path,
                {
                    "task": "not-a-task",
                    "model": "unit-model",
                    "input": {"prompt": "hello"},
                    "options": {},
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "runner_failed"
    assert payload["error"]["message"] == "task must be chat, image, tts, music, video, or misc"
    assert payload["error"]["detail"]["field"] == "task"


def test_native_runner_reports_planned_non_text_task_readiness(tmp_path):
    request_path = tmp_path / "planned-image.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "task": "image",
                    "model": "unit-image",
                    "input": {"prompt": "a red cube"},
                    "options": {
                        "modality": "image",
                        "engine": "diffusers",
                        "runtime": "planned_native",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                        "supported_backends": ["metal", "cuda"],
                        "expected_vram_gib": 8.0,
                        "expected_ram_gib": 16.0,
                        "requirements": {
                            "min_ram_gib": 16.0,
                            "allow_cpu": False,
                        },
                        "oom_policy": {
                            "action": "fail_before_runner",
                            "min_gpu_memory_gib": 8.0,
                            "min_ram_gib": 16.0,
                            "allow_cpu": False,
                        },
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["message"] == "native task is not implemented behind utopic-runner yet"
    assert payload["error"]["detail"]["task"] == "image"
    assert payload["error"]["detail"]["model"] == "unit-image"
    assert payload["error"]["detail"]["modality"] == "image"
    assert payload["error"]["detail"]["engine"] == "diffusers"
    assert payload["error"]["detail"]["runtime"] == "planned_native"
    assert payload["error"]["detail"]["runner"] == "utopic-runner"
    assert payload["error"]["detail"]["native_status"] == "planned"
    assert payload["error"]["detail"]["supported_backends"] == ["metal", "cuda"]
    assert payload["error"]["detail"]["expected_vram_gib"] == 8.0
    assert payload["error"]["detail"]["expected_ram_gib"] == 16.0
    assert payload["error"]["detail"]["requirements"] == {
        "min_ram_gib": 16.0,
        "allow_cpu": False,
    }
    assert payload["error"]["detail"]["oom_policy"] == {
        "action": "fail_before_runner",
        "min_gpu_memory_gib": 8.0,
        "min_ram_gib": 16.0,
        "allow_cpu": False,
    }


def test_native_runner_writes_progress_events_for_planned_task(tmp_path):
    progress_path = tmp_path / "run" / "progress.jsonl"
    request_path = tmp_path / "planned-image-with-progress.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "run_id": "run_progress",
                    "task": "image",
                    "model": "unit-image",
                    "input": {"prompt": "a red cube"},
                    "progress_path": str(progress_path),
                    "options": {
                        "modality": "image",
                        "engine": "diffusers",
                        "runtime": "planned_native",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["error"]["code"] == "unsupported_model"
    events = _progress_events(progress_path)
    assert [event["event"] for event in events] == ["started", "failed"]
    assert events[0]["schema_version"] == "utopic-runner/v1"
    assert events[0]["run_id"] == "run_progress"
    assert events[0]["task"] == "image"
    assert events[0]["model"] == "unit-image"
    assert events[0]["runner"] == "utopic-runner"
    assert isinstance(events[0]["time_ms"], int)
    assert events[1]["error"]["code"] == "unsupported_model"


def test_native_runner_response_includes_run_metadata_for_planned_task(tmp_path):
    output_dir = tmp_path / "run" / "outputs"
    progress_path = tmp_path / "run" / "progress.jsonl"
    request_path = tmp_path / "planned-image-with-run-metadata.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "run_id": "run_response_metadata",
                    "task": "image",
                    "model": "unit-image",
                    "input": {"prompt": "a red cube"},
                    "output_dir": str(output_dir),
                    "progress_path": str(progress_path),
                    "options": {
                        "modality": "image",
                        "engine": "diffusers",
                        "runtime": "planned_native",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary(), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["run_id"] == "run_response_metadata"
    assert payload["output_dir"] == str(output_dir)
    assert payload["progress_path"] == str(progress_path)
    assert payload["progress_url"] == "/v1/utopic/runs/run_response_metadata/events"


def test_modality_runner_entrypoint_reports_planned_readiness(tmp_path):
    request_path = tmp_path / "planned-image.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "task": "image",
                    "model": "unit-image",
                    "input": {"prompt": "a red cube"},
                    "options": {
                        "modality": "image",
                        "engine": "diffusers",
                        "runtime": "planned_native",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                        "supported_backends": ["metal", "cuda"],
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary("utopic-runner"), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["detail"]["runner"] == "utopic-runner"
    assert payload["error"]["detail"]["task"] == "image"


def test_modality_runner_readiness_includes_detected_runtime(tmp_path):
    request_path = tmp_path / "planned-image-detected-runtime.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
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
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner_with_env(
        _runner_binary("utopic-runner"),
        request_path,
        {
            "UTOPIC_RUNTIME_BACKEND": "cuda",
            "UTOPIC_RUNTIME_DEVICE": "unit-test-gpu",
        },
    )
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["detail"]["runner"] == "utopic-runner"
    assert payload["error"]["detail"]["detected"] == {
        "backend": "cuda",
        "device": "unit-test-gpu",
    }


def test_modality_runner_entrypoint_reports_its_own_name_without_runner_option(tmp_path):
    request_path = tmp_path / "planned-image-no-runner-option.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
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
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner(_runner_binary("utopic-runner"), request_path)
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["detail"]["runner"] == "utopic-runner"
    assert payload["error"]["detail"]["task"] == "image"


def test_native_runner_reports_oom_preflight_before_planned_readiness(tmp_path):
    request_path = tmp_path / "too-large-image.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "task": "image",
                    "model": "cosmos3-super",
                    "input": {"prompt": "a red cube"},
                    "options": {
                        "modality": "image",
                        "engine": "cosmos",
                        "runtime": "planned_native",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                        "supported_backends": ["cuda"],
                        "expected_vram_gib": 96.0,
                        "requirements": {
                            "min_gpu_memory_gib": 96,
                            "allow_cpu": False,
                        },
                        "oom_policy": {
                            "action": "fail_before_runner",
                            "min_gpu_memory_gib": 96,
                            "allow_cpu": False,
                        },
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner_with_env(
        _runner_binary("utopic-runner"),
        request_path,
        {
            "UTOPIC_GPU_MEMORY_GIB": "40",
            "UTOPIC_RUNTIME_BACKEND": "cuda",
            "UTOPIC_RUNTIME_DEVICE": "unit-test-gpu",
        },
    )
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "oom"
    assert "requires at least 96 GiB GPU memory" in payload["error"]["message"]
    detail = payload["error"]["detail"]
    assert detail["model"] == "cosmos3-super"
    assert detail["runner"] == "utopic-runner"
    assert detail["required_gpu_memory_gib"] == 96
    assert detail["oom_policy"] == {
        "action": "fail_before_runner",
        "min_gpu_memory_gib": 96,
        "allow_cpu": False,
    }
    assert detail["detected"]["gpu_memory_gib"] == 40
    assert detail["detected"]["backend"] == "cuda"
    assert detail["detected"]["device"] == "unit-test-gpu"


def test_native_runner_reports_unloadable_model_cleanly(tmp_path):
    missing_model = tmp_path / "missing.gguf"
    request_path = tmp_path / "missing-model.json"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "task": "chat",
                    "model": "unit-text",
                    "input": {"prompt": "hello"},
                    "options": {"model_path": str(missing_model)},
                },
            )
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
            _contract_request(
                tmp_path,
                {
                    "task": "image",
                    "model": "unit-image",
                    "input": {"prompt": "a red cube"},
                    "options": {
                        "modality": "image",
                        "engine": "diffusers",
                        "runtime": "planned_native",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                        "supported_backends": ["metal", "cuda"],
                        "expected_vram_gib": 8.0,
                    },
                },
            )
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
    assert detail["runner"] == "utopic-runner"
    assert detail["native_status"] == "planned"
    assert detail["supported_backends"] == ["metal", "cuda"]
    assert detail["expected_vram_gib"] == 8.0


def test_native_runner_reports_backend_unavailable_before_planned_modality(tmp_path):
    request_path = tmp_path / "cuda-image-request.json"
    progress_path = tmp_path / "progress.jsonl"
    request_path.write_text(
        json.dumps(
            _contract_request(
                tmp_path,
                {
                    "task": "image",
                    "model": "cuda-only-image",
                    "input": {"prompt": "a native backend gate"},
                    "progress_path": str(progress_path),
                    "options": {
                        "modality": "image",
                        "runner": "utopic-runner",
                        "native_status": "planned",
                        "supported_backends": ["cuda"],
                        "expected_vram_gib": 8.0,
                        "expected_ram_gib": 16.0,
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    completed = _run_runner_with_env(
        _runner_binary(),
        request_path,
        {
            "UTOPIC_RUNTIME_BACKEND": "metal",
            "UTOPIC_RUNTIME_DEVICE": "Apple test GPU",
            "UTOPIC_GPU_MEMORY_GIB": "48",
        },
    )
    payload = _last_json(completed.stdout)

    assert completed.returncode != 0
    assert payload["ok"] is False
    assert payload["error"]["code"] == "backend_unavailable"
    assert payload["run_id"] == "run_unit"
    assert payload["progress_url"] == "/v1/utopic/runs/run_unit/events"
    assert payload["progress_path"] == str(progress_path)
    assert payload["output_dir"] == str(tmp_path / "outputs")
    assert payload["error"]["detail"]["model"] == "cuda-only-image"
    assert payload["error"]["detail"]["supported_backends"] == ["cuda"]
    assert payload["error"]["detail"]["detected"]["backend"] == "metal"

    events = _progress_events(progress_path)
    assert [event["event"] for event in events] == ["started", "failed"]
    assert events[-1]["error"]["code"] == "backend_unavailable"
