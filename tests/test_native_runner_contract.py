import json
import subprocess
from pathlib import Path

from utopic import installer
from utopic.core_loader import load_core_module


native_runner = load_core_module("native_runner", installer_api=installer)
models = load_core_module("models", installer_api=installer)
CONTRACT = Path(native_runner.__file__).parent / "runner_contract" / "v1"


def _load_fixture(name: str) -> dict[str, object]:
    payload = json.loads((CONTRACT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _normalize_runner_request(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    normalized["run_id"] = "$RUN_ID"
    normalized["output_dir"] = "$OUTPUT_DIR"
    normalized["progress_path"] = "$PROGRESS_PATH"
    options = dict(normalized.get("options") or {})
    if "model_cache_path" in options:
        options["model_cache_path"] = "$MODEL_PATH"
    if "model_path" in options:
        options["model_path"] = "$MODEL_PATH"
    normalized["options"] = options
    return normalized


def test_runner_contract_schema_and_fixtures_are_packaged():
    schema = _load_fixture("schema.json")
    response_schema = _load_fixture("response_schema.json")
    chat = _load_fixture("chat_request.json")

    assert schema["properties"]["schema_version"]["const"] == native_runner.SCHEMA_VERSION
    assert set(schema["required"]) == set(chat)
    assert chat["schema_version"] == native_runner.SCHEMA_VERSION
    assert chat["task"] in schema["properties"]["task"]["enum"]
    option_schema = schema["properties"]["options"]
    option_properties = option_schema["properties"]
    assert option_schema["additionalProperties"] is True
    assert option_properties["requirements"]["type"] == "object"
    assert option_properties["requirements"]["properties"]["min_gpu_memory_gib"]["type"] == "number"
    assert option_properties["requirements"]["properties"]["allow_cpu"]["type"] == "boolean"
    assert option_properties["oom_policy"]["type"] == "object"
    assert option_properties["oom_policy"]["properties"]["action"]["const"] == "fail_before_runner"
    assert option_properties["oom_policy"]["properties"]["min_gpu_memory_gib"]["type"] == ["number", "null"]
    assert option_properties["oom_policy"]["properties"]["allow_cpu"]["type"] == ["boolean", "null"]
    assert chat["options"]["requirements"]["allow_cpu"] is True
    assert chat["options"]["oom_policy"]["action"] == "fail_before_runner"
    success_schema, error_schema = response_schema["oneOf"]
    assert success_schema["required"] == ["ok", "type", "artifacts", "metrics", "backend"]
    assert success_schema["properties"]["ok"]["const"] is True
    assert success_schema["properties"]["type"]["enum"] == ["text", "image", "audio", "video", "artifact"]
    assert success_schema["properties"]["backend"]["enum"] == ["metal", "cuda", "cpu"]
    assert success_schema["properties"]["artifacts"]["items"]["properties"]["mime_type"]["type"] == "string"
    assert error_schema["required"] == ["ok", "error"]
    assert error_schema["properties"]["ok"]["const"] is False
    error_properties = error_schema["properties"]["error"]["properties"]
    assert error_properties["code"]["enum"] == [
        "missing_model",
        "oom",
        "backend_unavailable",
        "unsupported_model",
        "runner_failed",
    ]
    assert error_properties["detail"]["type"] == "object"


def test_runner_request_emits_stable_contract_for_chat(tmp_path):
    entry = models.ModelEntry(
        id="unit-text",
        name="Unit Text",
        family="unit",
        filename="unit-text.gguf",
        url="https://example.invalid/unit-text.gguf",
        size="1 GiB",
        recommended=True,
        description="unit",
        modality="text",
        engine="native-text",
        runtime="native",
        runner="utopic-runner",
        native_status="ready",
    )

    payload = native_runner._runner_request(
        entry,
        "chat",
        {"messages": [{"role": "user", "content": "hello"}]},
        {"max_tokens": 16},
    )

    assert set(payload) == {
        "schema_version",
        "run_id",
        "task",
        "model",
        "input",
        "options",
        "output_dir",
        "progress_path",
    }
    assert payload["schema_version"] == "utopic-runner/v1"
    assert payload["task"] == "chat"
    assert payload["model"] == "unit-text"
    assert payload["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert isinstance(payload["options"], dict)
    assert payload["options"]["max_tokens"] == 16
    assert isinstance(payload["output_dir"], str)
    assert payload["output_dir"]


def test_runner_request_matches_versioned_chat_fixture():
    entry = models.ModelEntry(
        id="unit-text",
        name="Unit Text",
        family="unit",
        filename="unit-text.gguf",
        url="https://example.invalid/unit-text.gguf",
        size="1 GiB",
        recommended=True,
        description="unit",
        modality="text",
        engine="native-text",
        runtime="native",
        runner="utopic-runner",
        native_status="ready",
    )

    payload = native_runner._runner_request(
        entry,
        "chat",
        {"messages": [{"role": "user", "content": "hello"}]},
        {"max_tokens": 16},
    )

    assert _normalize_runner_request(payload) == _load_fixture("chat_request.json")


def test_runner_rejects_invalid_success_response(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake", encoding="utf-8")
    runner_path = tmp_path / "utopic-runner"
    runner_path.write_text("runner", encoding="utf-8")
    entry = models.ModelEntry(
        id="unit-text",
        name="Unit Text",
        family="unit",
        filename="model.gguf",
        url="https://example.invalid/model.gguf",
        size="1 MiB",
        recommended=True,
        description="unit",
        modality="text",
        engine="native-text",
        runtime="native",
        runner="utopic-runner",
        native_status="ready",
    )
    monkeypatch.setattr(type(entry), "path", property(lambda self: model_path))
    monkeypatch.setattr(native_runner._native, "binary_path", lambda _name: runner_path)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "type": "text"}) + "\n", stderr="")

    monkeypatch.setattr(native_runner.subprocess, "run", fake_run)

    payload = native_runner.chat_completion(entry, {"messages": [{"role": "user", "content": "hello"}]})

    assert payload["ok"] is False
    assert payload["error"]["code"] == "runner_failed"
    assert payload["error"]["message"] == "native runner response did not match utopic-runner/v1"
    assert payload["error"]["detail"]["missing"] == ["artifacts", "metrics", "backend"]


def test_runner_request_emits_stable_contract_for_artifact_generation():
    entry = models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="native",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
        supported_backends=("metal", "cuda"),
        runner="utopic-runner",
        native_status="ready",
        expected_vram_gib=8.0,
        requirements={"min_gpu_memory_gib": 96, "allow_cpu": False},
    )

    payload = native_runner._runner_request(
        entry,
        "image",
        {"prompt": "a native image"},
        {"size": "1024x1024"},
        endpoint="/v1/images/generations",
    )

    assert set(payload) == {
        "schema_version",
        "run_id",
        "task",
        "model",
        "input",
        "options",
        "output_dir",
        "progress_path",
    }
    assert payload["schema_version"] == "utopic-runner/v1"
    assert payload["task"] == "image"
    assert payload["model"] == "unit-image"
    assert payload["input"] == {"prompt": "a native image"}
    assert isinstance(payload["options"], dict)
    assert payload["options"]["endpoint"] == "/v1/images/generations"
    assert payload["options"]["runner"] == "utopic-runner"
    assert payload["options"]["requirements"] == {"min_gpu_memory_gib": 96, "allow_cpu": False}
    assert payload["options"]["oom_policy"]["min_gpu_memory_gib"] == 96
    assert payload["options"]["oom_policy"]["action"] == "fail_before_runner"
    assert payload["options"]["size"] == "1024x1024"
    assert isinstance(payload["output_dir"], str)
    assert payload["output_dir"]


def test_runner_request_allocates_unique_output_dir_per_invocation(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    entry = models.ModelEntry(
        id="unit-text",
        name="Unit Text",
        family="unit",
        filename="unit-text.gguf",
        url="https://example.invalid/unit-text.gguf",
        size="1 GiB",
        recommended=True,
        description="unit",
        modality="text",
        engine="native-text",
        runtime="native",
        runner="utopic-runner",
        native_status="ready",
    )
    monkeypatch.setenv("UTOPIC_RUNS_DIR", str(runs_dir))

    first = native_runner._runner_request(entry, "chat", {"messages": []}, {})
    second = native_runner._runner_request(entry, "chat", {"messages": []}, {})

    first_output = Path(first["output_dir"])
    second_output = Path(second["output_dir"])
    assert first_output != second_output
    assert first_output.name == "outputs"
    assert second_output.name == "outputs"
    assert first_output.parent.parent == runs_dir
    assert second_output.parent.parent == runs_dir
    assert first_output.is_dir()
    assert second_output.is_dir()


def test_runner_request_allocates_versioned_run_layout(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    entry = models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="native",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
        supported_backends=("metal", "cuda"),
        runner="utopic-runner",
        native_status="ready",
    )
    monkeypatch.setenv("UTOPIC_RUNS_DIR", str(runs_dir))

    payload = native_runner._runner_request(entry, "image", {"prompt": "a native image"}, {})

    assert isinstance(payload["run_id"], str)
    assert payload["run_id"]
    assert payload["output_dir"] == str(runs_dir / payload["run_id"] / "outputs")
    assert payload["progress_path"] == str(runs_dir / payload["run_id"] / "progress.jsonl")
    assert Path(payload["output_dir"]).is_dir()
    assert Path(payload["progress_path"]).parent.is_dir()


def test_generation_uses_catalog_runner_binary(monkeypatch, tmp_path):
    runner_path = tmp_path / "utopic-runner"
    runner_path.write_text("runner", encoding="utf-8")
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(tmp_path / "models"))
    entry = models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="native",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
        supported_backends=("metal", "cuda"),
        runner="utopic-runner",
        native_status="ready",
        expected_vram_gib=8.0,
    )
    captured = {}
    monkeypatch.setattr(native_runner._native, "binary_path", lambda name: captured.setdefault("binary", name) or runner_path)

    def fake_run(command, **kwargs):
        request_path = Path(command[2])
        captured["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "unsupported_model",
                        "message": "native runner task is not implemented yet",
                        "detail": {"task": "image", "model": entry.id},
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(native_runner.subprocess, "run", fake_run)

    payload = native_runner.generation(
        entry,
        "/v1/images/generations",
        {"model": entry.id, "prompt": "a native image", "size": "1024x1024"},
    )

    assert captured["binary"] == "utopic-runner"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert captured["request"]["task"] == "image"
    assert captured["request"]["options"]["runner"] == "utopic-runner"
    assert captured["request"]["options"]["model_path"].endswith("unit-image")
    assert captured["request"]["options"]["size"] == "1024x1024"


def test_generation_passes_installed_runtime_env(monkeypatch, tmp_path):
    runner_path = tmp_path / "utopic-runner"
    runner_path.write_text("runner", encoding="utf-8")
    entry = models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="planned_native",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
        supported_backends=("metal", "cuda"),
        runner="utopic-runner",
        native_status="planned",
    )
    captured = {}
    monkeypatch.setattr(native_runner._native, "binary_path", lambda _name: runner_path)
    monkeypatch.setattr(
        installer,
        "runner_environment",
        lambda: {
            "UTOPIC_RUNTIME_BACKEND": "metal",
            "UTOPIC_RUNTIME_DEVICE": "Apple M4 Pro",
        },
    )
    monkeypatch.setenv("UTOPIC_RUNTIME_BACKEND", "cuda")

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "unsupported_model",
                        "message": "native runner task is not implemented yet",
                        "detail": {"task": "image", "model": entry.id},
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(native_runner.subprocess, "run", fake_run)

    payload = native_runner.generation(
        entry,
        "/v1/images/generations",
        {"model": entry.id, "prompt": "a native image"},
    )

    assert payload["ok"] is False
    assert captured["env"]["UTOPIC_RUNTIME_BACKEND"] == "cuda"
    assert captured["env"]["UTOPIC_RUNTIME_DEVICE"] == "Apple M4 Pro"


def test_runner_error_payload_exposes_run_progress_metadata(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runner_path = tmp_path / "utopic-runner"
    runner_path.write_text("runner", encoding="utf-8")
    entry = models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="planned_native",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
        supported_backends=("metal", "cuda"),
        runner="utopic-runner",
        native_status="planned",
    )
    captured = {}
    monkeypatch.setenv("UTOPIC_RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(native_runner._native, "binary_path", lambda _name: runner_path)

    def fake_run(command, **kwargs):
        request_path = Path(command[2])
        captured["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "unsupported_model",
                        "message": "native runner task is not implemented yet",
                        "detail": {"task": "image", "model": entry.id},
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(native_runner.subprocess, "run", fake_run)

    payload = native_runner.generation(
        entry,
        "/v1/images/generations",
        {"model": entry.id, "prompt": "a native image"},
    )

    assert payload["ok"] is False
    assert payload["run_id"] == captured["request"]["run_id"]
    assert payload["output_dir"] == captured["request"]["output_dir"]
    assert payload["progress_path"] == captured["request"]["progress_path"]
    assert payload["progress_url"] == f"/v1/utopic/runs/{captured['request']['run_id']}/events"
    assert payload["error"]["detail"]["run_id"] == captured["request"]["run_id"]
    assert payload["error"]["detail"]["progress_url"] == payload["progress_url"]


def test_generation_reports_missing_catalog_runner_binary_as_setup_error(monkeypatch):
    entry = models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="planned_native",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
        supported_backends=("metal", "cuda"),
        runner="utopic-runner",
        native_status="planned",
        expected_vram_gib=8.0,
    )

    def missing_binary(name):
        raise RuntimeError(f"Utopic native binary is not installed: /cache/bin/{name}. Run `utopic setup`.")

    monkeypatch.setattr(native_runner._native, "binary_path", missing_binary)
    monkeypatch.setattr(native_runner.subprocess, "run", lambda *_args, **_kwargs: None)

    payload = native_runner.generation(
        entry,
        "/v1/images/generations",
        {"model": entry.id, "prompt": "a native image"},
    )

    assert payload["ok"] is False
    assert payload["error"]["code"] == "backend_unavailable"
    assert payload["error"]["model"] == entry.id
    assert payload["error"]["modality"] == "image"
    assert payload["error"]["runner"] == "utopic-runner"
    assert payload["error"]["native_status"] == "planned"
    assert payload["error"]["detail"]["binary"] == "utopic-runner"
    assert payload["error"]["detail"]["setup_command"] == "utopic setup"
