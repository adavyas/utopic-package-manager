import json
import os
import subprocess
from pathlib import Path

import pytest

from utopic import _native


def _runner_binary() -> Path:
    value = os.environ.get("UTOPIC_RUNNER_BINARY")
    if value:
        path = Path(value).expanduser()
    else:
        try:
            path = _native.binary_path("utopic_runner")
        except RuntimeError:
            pytest.skip("run utopic setup or set UTOPIC_RUNNER_BINARY=/path/to/utopic_runner")
    if not path.is_file():
        pytest.fail(f"utopic_runner does not exist: {path}")
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
