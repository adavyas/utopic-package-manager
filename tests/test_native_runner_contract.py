import json
import subprocess
from pathlib import Path

from utopic.core_loader import load_core_module


native_runner = load_core_module("native_runner")
models = load_core_module("models")


def test_generation_uses_catalog_runner_binary(monkeypatch, tmp_path):
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
