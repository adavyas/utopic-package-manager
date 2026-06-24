import json

from utopic import bridge


def test_bridge_check_reports_retired_adapter_without_dependency_imports(monkeypatch, capsys):
    def fail_if_imported(_packages):
        raise AssertionError("retired bridge shim must not inspect heavyweight packages")

    monkeypatch.setattr(bridge, "_missing_packages", fail_if_imported, raising=False)

    assert bridge.main(["diffusers", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "schema_version": "utopic-bridge/v1",
        "engine": "diffusers",
        "status": "retired",
        "ready": False,
        "packages": [],
        "missing": [],
        "install_hint": "",
        "description": "Compatibility shim only; production generation uses the native runner.",
        "message": (
            "The packaged Python bridge adapter has been retired. "
            "Use utopic setup plus the local native runner, gateway, or MCP surfaces."
        ),
    }


def test_bridge_generation_returns_retired_error_even_when_experimental_flag_is_set(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("UTOPIC_EXPERIMENTAL_BRIDGE", "1")
    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "native_runner_required"
    assert "native runner" in payload["error"]["message"]
    assert payload["error"]["install_hint"] == "utopic setup"
    assert payload["metadata"]["schema_version"] == "utopic-bridge/v1"
    assert payload["metadata"]["engine"] == "diffusers"
    assert not (tmp_path / "outputs").exists()


def test_bridge_rejects_invalid_contract_before_retired_error(capsys):
    assert bridge.main(["kokoro"], stdin=json.dumps({"model": "kokoro-82m"})) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "bridge_invalid_request"
    assert "schema_version must be utopic-bridge/v1" in payload["error"]["message"]
    assert payload["error"]["install_hint"] == ""


def test_bridge_unknown_engine_still_returns_structured_error(capsys):
    assert bridge.main(["missing-engine", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "bridge_engine_unknown"
    assert "unknown bridge engine: missing-engine" in payload["error"]["message"]


def test_retired_bridge_keeps_compatibility_engine_names():
    assert sorted(bridge.ADAPTERS) == [
        "ace-step",
        "artifact",
        "chatterbox",
        "cosmos",
        "dia",
        "diffusers",
        "kokoro",
        "ltx",
        "wan",
    ]
    assert all(adapter.packages == () for adapter in bridge.ADAPTERS.values())
