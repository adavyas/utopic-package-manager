from pathlib import Path

from utopic import gateway, models


def test_catalog_has_diffusiongemma_default():
    default = models.default_model()

    assert default.id.startswith("diffusiongemma")
    assert default.runtime == "native"
    assert default.modality == "text"


def test_catalog_includes_non_text_modalities():
    modalities = {entry.modality for entry in models.list_models()}

    assert {"image", "tts", "music", "video", "misc"}.issubset(modalities)


def test_catalog_entries_expose_native_readiness_metadata():
    for entry in models.list_models():
        assert entry.runner
        assert entry.native_status in models.VALID_NATIVE_STATUSES
        assert set(entry.supported_backends).issubset(models.VALID_BACKENDS)
        assert entry.expected_vram_gib is not None
        assert entry.expected_ram_gib is not None
        if entry.native_status == "ready":
            assert entry.runtime == "native"
            assert entry.runner == "utopic_runner"
        else:
            assert entry.runner.endswith("_runner")


def test_model_payload_exposes_native_readiness_fields():
    payload = gateway._model_payload(models.default_model())

    assert payload["runner"] == "utopic_runner"
    assert payload["native_status"] == "ready"
    assert "metal" in payload["supported_backends"]
    assert payload["expected_vram_gib"] > 0
    assert payload["expected_ram_gib"] > 0


def test_bridge_model_payload_hides_experimental_bridge_metadata_by_default(monkeypatch):
    monkeypatch.delenv("UTOPIC_EXPERIMENTAL_BRIDGE", raising=False)
    entry = next(entry for entry in models.list_models() if entry.runtime == "bridge")

    payload = gateway._model_payload(entry)

    assert payload["runner"] == entry.runner
    assert payload["native_status"] == entry.native_status
    assert "bridge" not in payload
    assert "experimental_bridge" not in payload


def test_bridge_model_payload_shows_experimental_bridge_metadata_when_enabled(monkeypatch):
    monkeypatch.setenv("UTOPIC_EXPERIMENTAL_BRIDGE", "1")
    entry = next(entry for entry in models.list_models() if entry.runtime == "bridge")

    payload = gateway._model_payload(entry)

    assert "bridge" not in payload
    assert payload["experimental_bridge"]["engine"] == entry.engine
    assert payload["experimental_bridge"]["environment_variable"].startswith("UTOPIC_BRIDGE_")


def test_bridge_model_cache_metadata_hides_bridge_command_by_default(monkeypatch):
    monkeypatch.delenv("UTOPIC_EXPERIMENTAL_BRIDGE", raising=False)
    entry = next(entry for entry in models.list_models() if entry.runtime == "bridge")

    payload = models._bridge_model_metadata(entry)

    assert payload["runner"] == entry.runner
    assert payload["native_status"] == entry.native_status
    assert "bridge" not in payload
    assert "experimental_bridge" not in payload


def test_bridge_model_check_defaults_to_native_runner_readiness(monkeypatch, tmp_path):
    monkeypatch.delenv("UTOPIC_EXPERIMENTAL_BRIDGE", raising=False)
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(tmp_path))
    entry = next(entry for entry in models.list_models() if entry.runtime == "bridge")

    payload = models.model_check(entry.id)

    assert payload["ready"] is False
    assert payload["status"] == "native_runner_not_ready"
    assert "bridge" not in payload
    assert payload["runner"] == entry.runner


def test_core_runtime_does_not_own_package_manager_cmake():
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "native" / "CMakeLists.txt").exists()


def test_native_runner_reports_package_selected_backend_when_defined():
    repo_root = Path(__file__).resolve().parents[1]
    runner_source = (
        repo_root / "python" / "utopic" / "core" / "native" / "runner.cpp"
    ).read_text(encoding="utf-8")

    assert "UTOPIC_BACKEND_NAME" in runner_source
