import json
import os
from pathlib import Path
import types

from utopic import bridge


def test_bridge_check_reports_missing_dependencies(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda packages: ["diffusers"] if packages else [])

    assert bridge.main(["diffusers", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "schema_version": "utopic-bridge/v1",
        "engine": "diffusers",
        "status": "missing_dependencies",
        "ready": False,
        "packages": ["diffusers", "torch", "torchvision"],
        "missing": ["diffusers"],
        "install_hint": 'pip install "utopic[image]"',
        "description": "Qwen-Image, FLUX, and Krea image generation through Diffusers.",
    }


def test_bridge_check_reports_ready_dependencies(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(bridge, "_engine_api_error", lambda _adapter: None)

    assert bridge.main(["kokoro", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "schema_version": "utopic-bridge/v1",
        "engine": "kokoro",
        "status": "ready",
        "ready": True,
        "packages": ["kokoro", "soundfile", "en_core_web_sm"],
        "missing": [],
        "install_hint": (
            'pip install "utopic[tts]" && python -m pip install '
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        ),
        "description": "Kokoro text-to-speech bridge.",
    }


def test_kokoro_check_reports_missing_spacy_model(monkeypatch, capsys):
    monkeypatch.setattr(
        bridge,
        "_missing_packages",
        lambda packages: ["en_core_web_sm"] if "en_core_web_sm" in packages else [],
    )

    assert bridge.main(["kokoro", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["engine"] == "kokoro"
    assert payload["ready"] is False
    assert payload["missing"] == ["en_core_web_sm"]
    assert payload["install_hint"] == (
        'pip install "utopic[tts]" && python -m pip install '
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    )


def test_chatterbox_check_reports_isolated_install_hint(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: ["chatterbox"])

    assert bridge.main(["chatterbox", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["engine"] == "chatterbox"
    assert payload["ready"] is False
    assert payload["install_hint"] == 'pip install "utopic[chatterbox]"'


def test_chatterbox_check_reports_missing_perth_watermarker(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setitem(
        __import__("sys").modules,
        "chatterbox.tts",
        types.SimpleNamespace(ChatterboxTTS=object),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "perth",
        types.SimpleNamespace(PerthImplicitWatermarker=None),
    )

    assert bridge.main(["chatterbox", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "api_mismatch"
    assert payload["ready"] is False
    assert "PerthImplicitWatermarker" in payload["message"]
    assert "setuptools<81" in payload["message"]


def test_torchvision_sensitive_bridge_adapters_declare_torchvision_dependency():
    for engine in ["diffusers", "dia", "wan", "ltx"]:
        assert "torchvision" in bridge.ADAPTERS[engine].packages


def test_dia_bridge_declares_soundfile_dependency():
    assert "soundfile" in bridge.ADAPTERS["dia"].packages


def test_acestep_bridge_declares_torchcodec_dependency():
    assert "torchcodec" in bridge.ADAPTERS["ace-step"].packages


def test_artifact_bridge_copies_input_artifact(tmp_path, capsys):
    source = tmp_path / "source.eeg"
    source.write_bytes(b"zuna-signal")
    output_dir = tmp_path / "outputs"
    progress_path = tmp_path / "progress.jsonl"
    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "zuna",
        "engine": "artifact",
        "modality": "misc",
        "input": {"artifact": str(source)},
        "parameters": {"artifact_type": "application/octet-stream"},
        "metadata": {"outputs": ["application/octet-stream"]},
        "output_dir": str(output_dir),
        "progress_path": str(progress_path),
    }

    assert bridge.ADAPTERS["artifact"].packages == ()
    assert bridge.main(["artifact"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "utopic-bridge/v1"
    assert payload["engine"] == "artifact"
    assert payload["artifacts"][0]["type"] == "application/octet-stream"
    output_path = Path(payload["artifacts"][0]["path"])
    assert output_path.parent == output_dir
    assert output_path.read_bytes() == b"zuna-signal"
    assert payload["artifacts"][0]["metadata"]["source"] == str(source)
    progress = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in progress] == ["loading", "generating", "completed"]


def test_safe_run_preserves_import_error_message_without_module_name(tmp_path):
    def runner(_request):
        raise ImportError("Please install `soundfile` to save audio files.")

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "dia-1.6b",
        "engine": "dia",
        "modality": "tts",
        "input": {"input": "[S1] Hello."},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    payload = bridge._safe_run(bridge.ADAPTERS["dia"], request, runner)

    assert payload["error"]["code"] == "bridge_dependency_missing"
    assert "Please install `soundfile` to save audio files." in payload["error"]["message"]


def test_audio_array_squeezes_channel_first_mono_tensor():
    class FakeArray:
        shape = (1, 3)

        def __getitem__(self, index):
            assert index == 0
            return ["sample-a", "sample-b", "sample-c"]

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return FakeArray()

    assert bridge._audio_array(FakeTensor()) == ["sample-a", "sample-b", "sample-c"]


def test_bridge_check_reports_api_mismatch(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(
        bridge,
        "_engine_api_error",
        lambda _adapter: "Could not import module 'PreTrainedModel'",
    )

    assert bridge.main(["diffusers", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "utopic-bridge/v1"
    assert payload["engine"] == "diffusers"
    assert payload["status"] == "api_mismatch"
    assert payload["ready"] is False
    assert payload["missing"] == []
    assert payload["install_hint"] == 'pip install "utopic[image]"'
    assert "PreTrainedModel" in payload["message"]


def test_bridge_check_explains_torchvision_nms_stack_mismatch(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(
        bridge,
        "_engine_api_error",
        lambda _adapter: (
            "Failed to import diffusers.pipelines.pipeline_utils because of the following error: "
            "Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly? "
            "Root cause: RuntimeError: operator torchvision::nms does not exist"
        ),
    )

    assert bridge.main(["diffusers", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "api_mismatch"
    assert payload["ready"] is False
    assert "torch/torchvision versions are incompatible" in payload["message"]
    assert "operator torchvision::nms does not exist" in payload["message"]


def test_bridge_exception_message_preserves_root_cause_chain():
    try:
        try:
            raise RuntimeError("operator torchvision::nms does not exist")
        except RuntimeError as exc:
            raise ModuleNotFoundError(
                "Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?"
            ) from exc
    except ModuleNotFoundError as exc:
        message = bridge._exception_message_with_causes(exc)

    assert "PreTrainedModel" in message
    assert "operator torchvision::nms does not exist" in message


def test_bridge_rejects_invalid_contract_before_dependency_check(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: ["diffusers"])

    assert bridge.main(["diffusers"], stdin=json.dumps({"model": "qwen-image"})) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "bridge_invalid_request"
    assert "schema_version must be utopic-bridge/v1" in payload["error"]["message"]
    assert payload["error"]["install_hint"] == ""
    assert payload["metadata"]["schema_version"] == "utopic-bridge/v1"
    assert payload["metadata"]["model"] == "qwen-image"


def test_bridge_rejects_engine_mismatch_before_dependency_check(monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: ["diffusers"])
    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "kokoro",
        "modality": "image",
        "input": {"prompt": "a teapot"},
        "model_cache_path": "/tmp/model-cache",
        "output_dir": "/tmp/outputs",
        "progress_path": "/tmp/progress.jsonl",
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "bridge_invalid_request"
    assert "engine must match adapter diffusers" in payload["error"]["message"]
    assert payload["error"]["install_hint"] == ""
    assert payload["metadata"]["model"] == "qwen-image"
    assert payload["metadata"]["modality"] == "image"


def test_diffusers_bridge_saves_image_artifact(tmp_path, monkeypatch, capsys):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"png")

    class FakePipeline:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def to(self, device):
            self.seen["device"] = device
            return self

        def __call__(self, prompt, **kwargs):
            self.seen["prompt"] = prompt
            self.seen["call_kwargs"] = kwargs
            return types.SimpleNamespace(images=[FakeImage()])

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"size": "512x512", "num_inference_steps": 2, "seed": 7},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "image/png"
    assert payload["artifacts"][0]["path"].endswith("image.png")
    assert payload["artifacts"][0]["metadata"]["model"] == "qwen-image"
    assert FakePipeline.seen["repo"] == str(tmp_path / "model-cache")
    assert FakePipeline.seen["prompt"] == "a glass teapot"
    assert FakePipeline.seen["call_kwargs"]["num_inference_steps"] == 2
    assert "size" not in FakePipeline.seen["call_kwargs"]
    assert "seed" not in FakePipeline.seen["call_kwargs"]

    progress = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert progress[0]["event"] == "loading"
    assert progress[-1] == {
        "event": "completed",
        "progress": 1.0,
        "message": "image saved",
    }


def test_qwen_image_bridge_uses_safe_true_cfg_default(tmp_path, monkeypatch, capsys):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"png")

    class FakePipeline:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            return cls()

        def to(self, device):
            return self

        def __call__(self, prompt, **kwargs):
            self.seen["call_kwargs"] = kwargs
            return types.SimpleNamespace(images=[FakeImage()])

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"size": "512x512", "num_inference_steps": 2},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "image/png"
    assert FakePipeline.seen["call_kwargs"]["true_cfg_scale"] == 4.0
    assert FakePipeline.seen["call_kwargs"]["negative_prompt"] == " "


def test_qwen_image_bridge_rejects_invalid_true_cfg_scale(tmp_path, monkeypatch, capsys):
    class FakePipeline:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            raise AssertionError("invalid Qwen parameters should be rejected before model load")

        def to(self, device):
            return self

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"true_cfg_scale": 1.0},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "bridge_adapter_failed"
    assert "true_cfg_scale must be greater than 1.0" in payload["error"]["message"]


def test_qwen_image_bridge_rejects_one_step_generation(tmp_path, monkeypatch, capsys):
    class FakePipeline:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            raise AssertionError("invalid Qwen parameters should be rejected before model load")

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"num_inference_steps": 1},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["error"]["code"] == "bridge_adapter_failed"
    assert "num_inference_steps must be at least 2" in payload["error"]["message"]


def test_flux_bridge_uses_schnell_defaults_and_cpu_offload(tmp_path, monkeypatch, capsys):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"png")

    class FakePipeline:
        seen = {"to_called": False, "offload_called": False}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def enable_model_cpu_offload(self):
            self.seen["offload_called"] = True

        def to(self, device):
            self.seen["to_called"] = True
            self.seen["device"] = device
            return self

        def __call__(self, prompt, **kwargs):
            self.seen["prompt"] = prompt
            self.seen["call_kwargs"] = kwargs
            return types.SimpleNamespace(images=[FakeImage()])

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "flux-1-schnell",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"size": "512x512"},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "image/png"
    assert FakePipeline.seen["repo"] == "black-forest-labs/FLUX.1-schnell"
    assert FakePipeline.seen["offload_called"] is True
    assert FakePipeline.seen["to_called"] is False
    assert FakePipeline.seen["call_kwargs"]["guidance_scale"] == 0.0
    assert FakePipeline.seen["call_kwargs"]["num_inference_steps"] == 4
    assert FakePipeline.seen["call_kwargs"]["max_sequence_length"] == 256


def test_flux_bridge_can_disable_cpu_offload(tmp_path, monkeypatch, capsys):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"png")

    class FakePipeline:
        seen = {"to_called": False, "offload_called": False}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            return cls()

        def enable_model_cpu_offload(self):
            self.seen["offload_called"] = True

        def to(self, device):
            self.seen["to_called"] = True
            self.seen["device"] = device
            return self

        def __call__(self, prompt, **kwargs):
            return types.SimpleNamespace(images=[FakeImage()])

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "flux-1-schnell",
        "repo": "black-forest-labs/FLUX.1-schnell",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"enable_model_cpu_offload": False},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0

    assert FakePipeline.seen["offload_called"] is False
    assert FakePipeline.seen["to_called"] is True
    assert FakePipeline.seen["device"] == "cuda"


def test_diffusers_bridge_can_disable_pipeline_safety_checker(tmp_path, monkeypatch, capsys):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"png")

    class FakePipeline:
        latest = None

        def __init__(self):
            self.safety_checker = object()
            self.requires_safety_checker = True
            self.registered_config = {}
            FakePipeline.latest = self

        @classmethod
        def from_pretrained(cls, _repo, **_kwargs):
            return cls()

        def register_to_config(self, **kwargs):
            self.registered_config.update(kwargs)

        def to(self, _device):
            return self

        def __call__(self, _prompt, **kwargs):
            self.call_kwargs = kwargs
            return types.SimpleNamespace(images=[FakeImage()])

    fake_diffusers = types.SimpleNamespace(DiffusionPipeline=FakePipeline)
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "tiny-diffusers-smoke",
        "repo": "hf-internal-testing/tiny-stable-diffusion-pipe",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {"disable_safety_checker": True, "num_inference_steps": 1},
        "model_cache_path": str(tmp_path / "model-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["path"].endswith("image.png")
    assert FakePipeline.latest.safety_checker is None
    assert FakePipeline.latest.requires_safety_checker is False
    assert FakePipeline.latest.registered_config == {"requires_safety_checker": False}
    assert "disable_safety_checker" not in FakePipeline.latest.call_kwargs


def test_diffusers_bridge_uses_repo_with_cache_dir_when_cache_has_only_utopic_metadata(
    tmp_path, monkeypatch, capsys
):
    class FakeImage:
        def save(self, path):
            path.write_bytes(b"png")

    class FakePipeline:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def to(self, device):
            return self

        def __call__(self, prompt, **kwargs):
            return types.SimpleNamespace(images=[FakeImage()])

    fake_diffusers = types.SimpleNamespace(
        DiffusionPipeline=FakePipeline,
        QwenImagePipeline=FakePipeline,
        FluxPipeline=FakePipeline,
    )
    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    cache_dir = tmp_path / "qwen-image"
    cache_dir.mkdir()
    (cache_dir / "utopic-model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "repo": "Qwen/Qwen-Image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "a glass teapot"},
        "parameters": {},
        "model_cache_path": str(cache_dir),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "image/png"
    assert FakePipeline.seen["repo"] == "Qwen/Qwen-Image"
    assert FakePipeline.seen["kwargs"]["cache_dir"] == str(cache_dir)


def test_model_source_uses_repo_with_cache_dir_when_cache_contains_hf_snapshot_metadata(tmp_path):
    cache_dir = tmp_path / "qwen-image"
    (cache_dir / "models--Qwen--Qwen-Image" / "snapshots" / "abc123").mkdir(parents=True)
    (cache_dir / "utopic-model.json").write_text("{}", encoding="utf-8")

    source, kwargs = bridge._model_source_and_kwargs(
        {
            "repo": "Qwen/Qwen-Image",
            "model_cache_path": str(cache_dir),
        },
        default_source="qwen-image",
    )

    assert source == "Qwen/Qwen-Image"
    assert kwargs == {"cache_dir": str(cache_dir)}


def test_kokoro_bridge_saves_audio_artifact(tmp_path, monkeypatch, capsys):
    class FakeKPipeline:
        seen = {}

        def __init__(self, lang_code):
            self.seen["lang_code"] = lang_code

        def __call__(self, text, voice, speed):
            self.seen["text"] = text
            self.seen["voice"] = voice
            self.seen["speed"] = speed
            yield ("graphemes", "phonemes", [0.0, 0.1, -0.1])

    writes = {}

    def fake_write(path, audio, sample_rate):
        writes["path"] = str(path)
        writes["audio"] = audio
        writes["sample_rate"] = sample_rate
        path.write_bytes(b"wav")

    monkeypatch.setitem(
        __import__("sys").modules,
        "kokoro",
        types.SimpleNamespace(KPipeline=FakeKPipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "kokoro-82m",
        "engine": "kokoro",
        "modality": "tts",
        "input": {"input": "hello"},
        "parameters": {"voice": "af_heart", "speed": 1.25, "lang_code": "a"},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["kokoro"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert payload["artifacts"][0]["path"].endswith("speech.wav")
    assert payload["artifacts"][0]["metadata"]["voice"] == "af_heart"
    assert FakeKPipeline.seen == {
        "lang_code": "a",
        "text": "hello",
        "voice": "af_heart",
        "speed": 1.25,
    }
    assert writes["sample_rate"] == 24000
    assert writes["audio"] == [0.0, 0.1, -0.1]


def test_bridge_resets_stale_progress_file_for_new_run(tmp_path, monkeypatch, capsys):
    class FakeKPipeline:
        def __init__(self, lang_code):
            self.lang_code = lang_code

        def __call__(self, _text, voice, speed):
            yield ("graphemes", "phonemes", [0.0])

    def fake_write(path, _audio, _sample_rate):
        path.write_bytes(b"wav")

    monkeypatch.setitem(
        __import__("sys").modules,
        "kokoro",
        types.SimpleNamespace(KPipeline=FakeKPipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        json.dumps({"event": "stale", "progress": 0.0, "message": "old"}) + "\n",
        encoding="utf-8",
    )
    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "kokoro-82m",
        "engine": "kokoro",
        "modality": "tts",
        "input": {"input": "hello"},
        "parameters": {},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(progress_path),
    }

    assert bridge.main(["kokoro"], stdin=json.dumps(request)) == 0
    json.loads(capsys.readouterr().out)

    progress = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["event"] for item in progress] == ["loading", "generating", "completed"]


def test_kokoro_bridge_accepts_current_result_object_shape(tmp_path, monkeypatch, capsys):
    class FakeOutput:
        audio = [0.0, 0.25, -0.25]

    class FakeResult:
        output = FakeOutput()

    class FakeKPipeline:
        def __init__(self, lang_code):
            self.lang_code = lang_code

        def __call__(self, _text, voice, speed):
            yield FakeResult()

    writes = {}

    def fake_write(path, audio, sample_rate):
        writes["audio"] = audio
        writes["sample_rate"] = sample_rate
        path.write_bytes(b"wav")

    monkeypatch.setitem(
        __import__("sys").modules,
        "kokoro",
        types.SimpleNamespace(KPipeline=FakeKPipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "kokoro-82m",
        "engine": "kokoro",
        "modality": "tts",
        "input": {"input": "hello"},
        "parameters": {"voice": "af_heart"},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["kokoro"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert writes == {"audio": [0.0, 0.25, -0.25], "sample_rate": 24000}


def test_bridge_redirects_adapter_stdout_noise_to_stderr(tmp_path, monkeypatch, capsys):
    class FakeOutput:
        audio = [0.0]

    class FakeResult:
        output = FakeOutput()

    class NoisyKPipeline:
        def __init__(self, lang_code):
            print("external setup log")

        def __call__(self, _text, voice, speed):
            print("external generation log")
            yield FakeResult()

    def fake_write(path, _audio, _sample_rate):
        path.write_bytes(b"wav")

    monkeypatch.setitem(
        __import__("sys").modules,
        "kokoro",
        types.SimpleNamespace(KPipeline=NoisyKPipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "kokoro-82m",
        "engine": "kokoro",
        "modality": "tts",
        "input": {"input": "hello"},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["kokoro"], stdin=json.dumps(request)) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert "external setup log" not in captured.out
    assert "external generation log" not in captured.out
    assert "external setup log" in captured.err
    assert "external generation log" in captured.err


def test_bridge_redirects_adapter_fd_stdout_noise_to_stderr(tmp_path, monkeypatch, capfd):
    class FakeOutput:
        audio = [0.0]

    class FakeResult:
        output = FakeOutput()

    class FdNoisyKPipeline:
        def __init__(self, lang_code):
            os.write(1, b"external fd setup log\n")

        def __call__(self, _text, voice, speed):
            os.write(1, b"external fd generation log\n")
            yield FakeResult()

    def fake_write(path, _audio, _sample_rate):
        path.write_bytes(b"wav")

    monkeypatch.setitem(
        __import__("sys").modules,
        "kokoro",
        types.SimpleNamespace(KPipeline=FdNoisyKPipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "kokoro-82m",
        "engine": "kokoro",
        "modality": "tts",
        "input": {"input": "hello"},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["kokoro"], stdin=json.dumps(request)) == 0
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert "external fd setup log" not in captured.out
    assert "external fd generation log" not in captured.out
    assert "external fd setup log" in captured.err
    assert "external fd generation log" in captured.err


def test_bridge_reports_optional_engine_import_mismatch_with_install_hint():
    adapter = bridge.ADAPTERS["diffusers"]

    def broken_runner(_request):
        raise RuntimeError(
            "Failed to import diffusers.pipelines.pipeline_utils because of the following error: "
            "Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?"
        )

    payload = bridge._safe_run(
        adapter,
        {"schema_version": "utopic-bridge/v1", "model": "qwen-image"},
        broken_runner,
    )

    assert payload["error"]["code"] == "bridge_adapter_api_mismatch"
    assert payload["error"]["engine"] == "diffusers"
    assert payload["error"]["install_hint"] == adapter.install_hint
    assert "PreTrainedModel" in payload["error"]["message"]


def test_bridge_reports_huggingface_gated_repo_errors_with_auth_hint():
    adapter = bridge.ADAPTERS["diffusers"]

    def gated_runner(_request):
        raise RuntimeError(
            "401 Client Error. Cannot access gated repo for url "
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/model_index.json. "
            "Access to model black-forest-labs/FLUX.1-schnell is restricted. Please log in."
        )

    payload = bridge._safe_run(
        adapter,
        {"schema_version": "utopic-bridge/v1", "model": "flux-1-schnell"},
        gated_runner,
    )

    assert payload["error"]["code"] == "bridge_auth_required"
    assert payload["error"]["engine"] == "diffusers"
    assert "requires Hugging Face access" in payload["error"]["message"]
    assert "HF_TOKEN" in payload["error"]["install_hint"]


def test_bridge_reports_non_runtime_huggingface_auth_errors_with_auth_hint():
    adapter = bridge.ADAPTERS["diffusers"]

    def gated_runner(_request):
        raise OSError(
            "401 Client Error. Cannot access gated repo for url "
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/model_index.json. "
            "Access to model black-forest-labs/FLUX.1-schnell is restricted. Please log in."
        )

    payload = bridge._safe_run(
        adapter,
        {"schema_version": "utopic-bridge/v1", "model": "flux-1-schnell"},
        gated_runner,
    )

    assert payload["error"]["code"] == "bridge_auth_required"
    assert "requires Hugging Face access" in payload["error"]["message"]
    assert "HF_TOKEN" in payload["error"]["install_hint"]


def test_chatterbox_bridge_saves_audio_artifact(tmp_path, monkeypatch, capsys):
    class FakeChatterboxTTS:
        seen = {}
        sr = 24000

        @classmethod
        def from_pretrained(cls, device):
            cls.seen["device"] = device
            return cls()

        def generate(self, text, **kwargs):
            self.seen["text"] = text
            self.seen["kwargs"] = kwargs
            return [0.2, 0.1, 0.0]

    writes = {}

    def fake_write(path, audio, sample_rate):
        writes["path"] = str(path)
        writes["audio"] = audio
        writes["sample_rate"] = sample_rate
        path.write_bytes(b"wav")

    monkeypatch.setitem(__import__("sys").modules, "chatterbox", types.SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "chatterbox.tts",
        types.SimpleNamespace(ChatterboxTTS=FakeChatterboxTTS),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(bridge, "_device_for_python_model", lambda: "cpu")

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "chatterbox",
        "engine": "chatterbox",
        "modality": "tts",
        "input": {"input": "hello from chatterbox"},
        "parameters": {"temperature": 0.7, "voice_prompt_path": "/tmp/voice.wav"},
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["chatterbox"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert payload["artifacts"][0]["path"].endswith("speech.wav")
    assert payload["artifacts"][0]["metadata"]["engine"] == "chatterbox"
    assert FakeChatterboxTTS.seen == {
        "device": "cpu",
        "text": "hello from chatterbox",
        "kwargs": {"temperature": 0.7, "voice_prompt_path": "/tmp/voice.wav"},
    }
    assert writes == {
        "path": str(tmp_path / "outputs" / "speech.wav"),
        "audio": [0.2, 0.1, 0.0],
        "sample_rate": 24000,
    }


def test_dia_bridge_uses_transformers_and_saves_audio(tmp_path, monkeypatch, capsys):
    class FakeInputs(dict):
        def to(self, device):
            self["device"] = device
            return self

    class FakeProcessor:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo):
            cls.seen["repo"] = repo
            return cls()

        def __call__(self, **kwargs):
            self.seen["call"] = kwargs
            return FakeInputs({"input_ids": [1]})

        def batch_decode(self, outputs):
            self.seen["outputs"] = outputs
            return ["decoded-audio"]

        def save_audio(self, outputs, path):
            self.seen["save_audio"] = {"outputs": outputs, "path": path}
            __import__("pathlib").Path(path).write_bytes(b"wav")

    class FakeDiaModel:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo):
            cls.seen["repo"] = repo
            return cls()

        def to(self, device):
            self.seen["device"] = device
            return self

        def generate(self, **kwargs):
            self.seen["generate"] = kwargs
            return ["tokens"]

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            DiaForConditionalGeneration=FakeDiaModel,
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        ),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "dia-1.6b",
        "engine": "dia",
        "modality": "tts",
        "input": {"input": "[S1] Hello. [S2] Hi."},
        "parameters": {"max_new_tokens": 128, "guidance_scale": 3.0},
        "model_cache_path": "nari-labs/Dia-1.6B-0626",
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["dia"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert payload["artifacts"][0]["path"].endswith("speech.wav")
    assert FakeProcessor.seen["repo"] == "nari-labs/Dia-1.6B-0626"
    assert FakeProcessor.seen["call"]["text"] == ["[S1] Hello. [S2] Hi."]
    assert FakeDiaModel.seen["generate"]["max_new_tokens"] == 128
    assert FakeDiaModel.seen["generate"]["guidance_scale"] == 3.0


def test_dia_bridge_uses_repo_with_cache_dir_for_metadata_only_cache(
    tmp_path, monkeypatch, capsys
):
    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def __call__(self, **kwargs):
            return FakeInputs({"input_ids": [1]})

        def batch_decode(self, outputs):
            return ["decoded-audio"]

        def save_audio(self, outputs, path):
            __import__("pathlib").Path(path).write_bytes(b"wav")

    class FakeDiaModel:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def to(self, device):
            return self

        def generate(self, **kwargs):
            return ["tokens"]

    cache_dir = tmp_path / "dia-1.6b"
    cache_dir.mkdir()
    (cache_dir / "utopic-model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            DiaForConditionalGeneration=FakeDiaModel,
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        ),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "dia-1.6b",
        "repo": "nari-labs/Dia-1.6B-0626",
        "engine": "dia",
        "modality": "tts",
        "input": {"input": "[S1] Hello."},
        "parameters": {},
        "model_cache_path": str(cache_dir),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["dia"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert FakeProcessor.seen == {
        "repo": "nari-labs/Dia-1.6B-0626",
        "kwargs": {"cache_dir": str(cache_dir)},
    }
    assert FakeDiaModel.seen == {
        "repo": "nari-labs/Dia-1.6B-0626",
        "kwargs": {"cache_dir": str(cache_dir)},
    }


def test_wan_bridge_exports_video_artifact(tmp_path, monkeypatch, capsys):
    class FakeWanPipeline:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def to(self, device):
            self.seen["device"] = device
            return self

        def __call__(self, **kwargs):
            self.seen["call_kwargs"] = kwargs
            return types.SimpleNamespace(frames=[["frame-1", "frame-2"]])

    exports = {}

    def fake_export_to_video(frames, path, fps):
        exports["frames"] = frames
        exports["path"] = str(path)
        exports["fps"] = fps
        path.write_bytes(b"mp4")

    fake_diffusers = types.SimpleNamespace(
        WanPipeline=FakeWanPipeline,
        DiffusionPipeline=FakeWanPipeline,
        utils=types.SimpleNamespace(export_to_video=fake_export_to_video),
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers.utils",
        types.SimpleNamespace(export_to_video=fake_export_to_video),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "wan2.1-t2v-1.3b",
        "engine": "wan",
        "modality": "video",
        "input": {"prompt": "a camera moves through a forest"},
        "parameters": {"num_frames": 9, "fps": 12, "guidance_scale": 4.0},
        "model_cache_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["wan"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "video/mp4"
    assert payload["artifacts"][0]["path"].endswith("video.mp4")
    assert FakeWanPipeline.seen["repo"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert FakeWanPipeline.seen["call_kwargs"]["prompt"] == "a camera moves through a forest"
    assert FakeWanPipeline.seen["call_kwargs"]["num_frames"] == 9
    assert exports == {
        "frames": ["frame-1", "frame-2"],
        "path": str(tmp_path / "outputs" / "video.mp4"),
        "fps": 12,
    }


def test_wan_bridge_translates_openai_size_to_width_and_height(tmp_path, monkeypatch, capsys):
    class FakeWanPipeline:
        seen = {}

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return cls()

        def to(self, device):
            return self

        def __call__(self, prompt, **kwargs):
            self.seen["kwargs"] = kwargs
            return types.SimpleNamespace(frames=[["frame-1"]])

    def fake_export_to_video(frames, path, fps):
        path.write_bytes(b"mp4")

    fake_diffusers = types.SimpleNamespace(WanPipeline=FakeWanPipeline, DiffusionPipeline=FakeWanPipeline)
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers.utils",
        types.SimpleNamespace(export_to_video=fake_export_to_video),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "wan2.1-t2v-1.3b",
        "engine": "wan",
        "modality": "video",
        "input": {"prompt": "waves"},
        "parameters": {"size": "320x240", "num_frames": 5},
        "model_cache_path": str(tmp_path / "wan-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["wan"], stdin=json.dumps(request)) == 0

    assert FakeWanPipeline.seen["kwargs"]["width"] == 320
    assert FakeWanPipeline.seen["kwargs"]["height"] == 240
    assert "size" not in FakeWanPipeline.seen["kwargs"]


def test_video_frame_extractor_accepts_array_like_batched_frames():
    class FakeFrames:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return "first-batch-video-frames"

    result = types.SimpleNamespace(frames=FakeFrames())

    assert bridge._first_video_frames(result) == "first-batch-video-frames"


def test_ltx_bridge_exports_video_artifact(tmp_path, monkeypatch, capsys):
    class FakeLTXPipeline:
        seen = {}

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            cls.seen["repo"] = repo
            cls.seen["kwargs"] = kwargs
            return cls()

        def to(self, device):
            self.seen["device"] = device
            return self

        def __call__(self, **kwargs):
            self.seen["call_kwargs"] = kwargs
            return types.SimpleNamespace(frames=[["ltx-frame-1", "ltx-frame-2"]])

    exports = {}

    def fake_export_to_video(frames, path, fps):
        exports["frames"] = frames
        exports["path"] = str(path)
        exports["fps"] = fps
        path.write_bytes(b"mp4")

    fake_diffusers = types.SimpleNamespace(
        LTXPipeline=FakeLTXPipeline,
        DiffusionPipeline=FakeLTXPipeline,
        utils=types.SimpleNamespace(export_to_video=fake_export_to_video),
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers.utils",
        types.SimpleNamespace(export_to_video=fake_export_to_video),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "ltx-video",
        "repo": "Lightricks/LTX-Video",
        "engine": "ltx",
        "modality": "video",
        "input": {"prompt": "a clean product turntable shot"},
        "parameters": {"num_frames": 8, "fps": 8, "guidance_scale": 3.0},
        "model_cache_path": "Lightricks/LTX-Video",
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["ltx"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "video/mp4"
    assert payload["artifacts"][0]["path"].endswith("video.mp4")
    assert payload["artifacts"][0]["metadata"]["engine"] == "ltx"
    assert FakeLTXPipeline.seen["repo"] == "Lightricks/LTX-Video"
    assert FakeLTXPipeline.seen["call_kwargs"]["prompt"] == "a clean product turntable shot"
    assert FakeLTXPipeline.seen["call_kwargs"]["num_frames"] == 8
    assert exports == {
        "frames": ["ltx-frame-1", "ltx-frame-2"],
        "path": str(tmp_path / "outputs" / "video.mp4"),
        "fps": 8,
    }


def test_acestep_check_uses_official_pipeline_module(monkeypatch, capsys):
    class FakePipeline:
        pass

    monkeypatch.setitem(__import__("sys").modules, "acestep", types.SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "acestep.pipeline_ace_step",
        types.SimpleNamespace(ACEStepPipeline=FakePipeline),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])

    assert bridge.main(["ace-step", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["engine"] == "ace-step"
    assert payload["ready"] is True
    assert payload["status"] == "ready"


def test_acestep_bridge_uses_official_pipeline_and_saves_music(tmp_path, monkeypatch, capsys):
    class FakePipeline:
        seen = {}

        def __init__(self, checkpoint_dir="", dtype="bfloat16", **kwargs):
            self.seen["checkpoint_dir"] = checkpoint_dir
            self.seen["dtype"] = dtype
            self.seen["init_kwargs"] = kwargs

        def __call__(self, **kwargs):
            self.seen["call"] = kwargs
            return {"audio": [0.0, 0.3], "sample_rate": 44100}

    writes = {}

    def fake_write(path, audio, sample_rate):
        writes["path"] = str(path)
        writes["audio"] = audio
        writes["sample_rate"] = sample_rate
        path.write_bytes(b"wav")

    monkeypatch.setitem(__import__("sys").modules, "acestep", types.SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "acestep.pipeline_ace_step",
        types.SimpleNamespace(ACEStepPipeline=FakePipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=fake_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(bridge, "_device_for_python_model", lambda: "cpu")

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "ace-step-3.5b",
        "engine": "ace-step",
        "modality": "music",
        "input": {"prompt": "bright synthwave"},
        "parameters": {"lyrics": "", "duration": 8},
        "model_cache_path": str(tmp_path / "ace-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["ace-step"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert payload["artifacts"][0]["path"].endswith("music.wav")
    assert FakePipeline.seen["checkpoint_dir"] == str(tmp_path / "ace-cache")
    assert FakePipeline.seen["call"] == {
        "prompt": "bright synthwave",
        "lyrics": "",
        "audio_duration": 8,
        "save_path": str(tmp_path / "outputs" / "music.wav"),
    }
    assert writes["sample_rate"] == 44100


def test_acestep_bridge_returns_pipeline_saved_music_file(tmp_path, monkeypatch, capsys):
    class FakePipeline:
        seen = {}

        def __init__(self, checkpoint_dir="", dtype="bfloat16", **kwargs):
            self.seen["checkpoint_dir"] = checkpoint_dir
            self.seen["dtype"] = dtype
            self.seen["init_kwargs"] = kwargs

        def __call__(self, **kwargs):
            self.seen["call"] = kwargs
            Path(kwargs["save_path"]).write_bytes(b"wav")
            return [kwargs["save_path"]]

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("ACE-Step save_path artifacts should not be re-saved")

    monkeypatch.setitem(__import__("sys").modules, "acestep", types.SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "acestep.pipeline_ace_step",
        types.SimpleNamespace(ACEStepPipeline=FakePipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=unexpected_write),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(bridge, "_device_for_python_model", lambda: "cpu")

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "ace-step-3.5b",
        "engine": "ace-step",
        "modality": "music",
        "input": {"prompt": "bright synthwave"},
        "parameters": {"lyrics": "", "duration": 8},
        "model_cache_path": str(tmp_path / "ace-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["ace-step"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["artifacts"][0]["type"] == "audio/wav"
    assert payload["artifacts"][0]["path"] == str(tmp_path / "outputs" / "music.wav")
    assert FakePipeline.seen["call"]["save_path"] == str(tmp_path / "outputs" / "music.wav")


def test_acestep_bridge_keeps_sample_rate_as_metadata_only(tmp_path, monkeypatch, capsys):
    class FakePipeline:
        seen = {}

        def __init__(self, checkpoint_dir="", dtype="bfloat16", **kwargs):
            self.seen["checkpoint_dir"] = checkpoint_dir
            self.seen["dtype"] = dtype
            self.seen["init_kwargs"] = kwargs

        def __call__(self, **kwargs):
            self.seen["call"] = kwargs
            Path(kwargs["save_path"]).write_bytes(b"wav")
            return [kwargs["save_path"]]

    monkeypatch.setitem(__import__("sys").modules, "acestep", types.SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "acestep.pipeline_ace_step",
        types.SimpleNamespace(ACEStepPipeline=FakePipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "soundfile",
        types.SimpleNamespace(write=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(bridge, "_missing_packages", lambda _packages: [])
    monkeypatch.setattr(bridge, "_device_for_python_model", lambda: "cuda")

    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "ace-step-3.5b",
        "engine": "ace-step",
        "modality": "music",
        "input": {"prompt": "bright synthwave"},
        "parameters": {"duration": 8, "sample_rate": 24000},
        "model_cache_path": str(tmp_path / "ace-cache"),
        "output_dir": str(tmp_path / "outputs"),
        "progress_path": str(tmp_path / "progress.jsonl"),
    }

    assert bridge.main(["ace-step"], stdin=json.dumps(request)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert "sample_rate" not in FakePipeline.seen["call"]
    assert FakePipeline.seen["call"]["audio_duration"] == 8
    assert payload["artifacts"][0]["metadata"]["sample_rate"] == 24000
