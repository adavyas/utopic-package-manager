import json
import subprocess
from pathlib import Path

from utopic import installer
from utopic.core_loader import load_core_module


native_runner = load_core_module("native_runner", installer_api=installer)
models = load_core_module("models", installer_api=installer)


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

    assert set(payload) == {"schema_version", "task", "model", "input", "options", "output_dir"}
    assert payload["schema_version"] == "utopic-runner/v1"
    assert payload["task"] == "chat"
    assert payload["model"] == "unit-text"
    assert payload["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert isinstance(payload["options"], dict)
    assert payload["options"]["max_tokens"] == 16
    assert isinstance(payload["output_dir"], str)
    assert payload["output_dir"]


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
        runner="image_runner",
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

    assert set(payload) == {"schema_version", "task", "model", "input", "options", "output_dir"}
    assert payload["schema_version"] == "utopic-runner/v1"
    assert payload["task"] == "image"
    assert payload["model"] == "unit-image"
    assert payload["input"] == {"prompt": "a native image"}
    assert isinstance(payload["options"], dict)
    assert payload["options"]["endpoint"] == "/v1/images/generations"
    assert payload["options"]["runner"] == "image_runner"
    assert payload["options"]["requirements"] == {"min_gpu_memory_gib": 96, "allow_cpu": False}
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
    assert first_output.parent == runs_dir
    assert second_output.parent == runs_dir
    assert first_output.is_dir()
    assert second_output.is_dir()


def test_generation_uses_catalog_runner_binary(monkeypatch, tmp_path):
    runner_path = tmp_path / "image_runner"
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
        runner="image_runner",
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

    assert captured["binary"] == "image_runner"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert captured["request"]["task"] == "image"
    assert captured["request"]["options"]["runner"] == "image_runner"
    assert captured["request"]["options"]["model_path"].endswith("unit-image")
    assert captured["request"]["options"]["size"] == "1024x1024"


def test_generation_passes_installed_runtime_env(monkeypatch, tmp_path):
    runner_path = tmp_path / "image_runner"
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
        runner="image_runner",
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
        runner="image_runner",
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
    assert payload["error"]["runner"] == "image_runner"
    assert payload["error"]["native_status"] == "planned"
    assert payload["error"]["detail"]["binary"] == "image_runner"
    assert payload["error"]["detail"]["setup_command"] == "utopic setup"
