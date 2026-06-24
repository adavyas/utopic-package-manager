import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any, Optional


SCHEMA_VERSION = "utopic-bridge/v1"


@dataclass(frozen=True)
class BridgeAdapter:
    engine: str
    packages: tuple[str, ...]
    install_hint: str
    description: str


ADAPTERS = {
    "diffusers": BridgeAdapter(
        engine="diffusers",
        packages=("diffusers", "torch", "torchvision"),
        install_hint='pip install "utopic[image]"',
        description="Qwen-Image, FLUX, and Krea image generation through Diffusers.",
    ),
    "cosmos": BridgeAdapter(
        engine="cosmos",
        packages=("cosmos", "torch"),
        install_hint=(
            "Install the NVIDIA Cosmos runtime for this Python environment and "
            "set UTOPIC_BRIDGE_COSMOS_COMMAND to a compatible local bridge command."
        ),
        description="NVIDIA Cosmos3 Super image generation bridge.",
    ),
    "kokoro": BridgeAdapter(
        engine="kokoro",
        packages=("kokoro", "soundfile", "en_core_web_sm"),
        install_hint=(
            'pip install "utopic[tts]" && python -m pip install '
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        ),
        description="Kokoro text-to-speech bridge.",
    ),
    "chatterbox": BridgeAdapter(
        engine="chatterbox",
        packages=("chatterbox", "soundfile"),
        install_hint='pip install "utopic[chatterbox]"',
        description="Chatterbox text-to-speech bridge.",
    ),
    "dia": BridgeAdapter(
        engine="dia",
        packages=("transformers", "torch", "torchvision", "soundfile"),
        install_hint='pip install "utopic[tts]"',
        description="Dia expressive dialogue TTS bridge through Transformers.",
    ),
    "ace-step": BridgeAdapter(
        engine="ace-step",
        packages=("acestep", "soundfile", "torchcodec"),
        install_hint='pip install "utopic[music]" && pip install git+https://github.com/ace-step/ACE-Step.git',
        description="ACE-Step text-to-music bridge.",
    ),
    "wan": BridgeAdapter(
        engine="wan",
        packages=("diffusers", "torch", "torchvision"),
        install_hint='pip install "utopic[video]"',
        description="Wan text-to-video bridge.",
    ),
    "ltx": BridgeAdapter(
        engine="ltx",
        packages=("diffusers", "torch", "torchvision"),
        install_hint='pip install "utopic[video]"',
        description="LTX-Video bridge through Diffusers.",
    ),
    "artifact": BridgeAdapter(
        engine="artifact",
        packages=(),
        install_hint="",
        description="Generic local file-in/file-out bridge for misc artifacts.",
    ),
}


HELP = """usage: utopic-bridge ENGINE

Run a packaged Utopic bridge adapter. The adapter reads one utopic-bridge/v1
JSON request from stdin and writes one JSON response to stdout.

Generation is experimental and disabled unless UTOPIC_EXPERIMENTAL_BRIDGE=1
is set. Use --check to inspect adapter dependencies without enabling runs.

Known engines:
  diffusers   Qwen-Image, FLUX, and Krea image generation
  cosmos      Cosmos3 Super image generation
  kokoro      Kokoro TTS
  chatterbox  Chatterbox TTS
  dia         Dia TTS
  ace-step    ACE-Step music generation
  wan         Wan video generation
  ltx         LTX-Video generation
  artifact    Generic misc artifact passthrough
"""


def main(argv: Optional[list[str]] = None, *, stdin: Optional[str] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or any(arg in ("-h", "--help") for arg in args):
        print(HELP)
        return 0
    engine = args[0]
    adapter = ADAPTERS.get(engine)
    if adapter is None:
        print(json.dumps(_error(engine, f"unknown bridge engine: {engine}", "bridge_engine_unknown", "")))
        return 0
    if "--check" in args:
        print(json.dumps(_check_adapter(adapter)))
        return 0
    raw_request = sys.stdin.read() if stdin is None else stdin
    try:
        request = json.loads(raw_request or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps(_error(adapter.engine, f"invalid bridge request JSON: {exc}", "bridge_invalid_request", "")))
        return 0
    validation_error = _validate_bridge_request(adapter, request)
    if validation_error is not None:
        print(json.dumps(_error(adapter.engine, validation_error, "bridge_invalid_request", "", request=request)))
        return 0
    if not _experimental_bridge_enabled():
        print(
            json.dumps(
                _error(
                    adapter.engine,
                    (
                        "utopic-bridge generation is experimental and disabled by default. "
                        "Set UTOPIC_EXPERIMENTAL_BRIDGE=1 to run this local Python adapter, "
                        "or use the native runner/OpenAI/MCP surfaces for production paths."
                    ),
                    "bridge_experimental_disabled",
                    "export UTOPIC_EXPERIMENTAL_BRIDGE=1",
                    request=request,
                )
            )
        )
        return 0
    missing = _missing_packages(adapter.packages)
    if missing:
        print(
            json.dumps(
                _error(
                    adapter.engine,
                    f"{adapter.engine} bridge dependencies are not installed: {', '.join(missing)}",
                    "bridge_dependency_missing",
                    adapter.install_hint,
                    request=request,
                )
            )
        )
        return 0
    if adapter.engine == "diffusers":
        _print_run_result(adapter, request, _run_diffusers)
        return 0
    if adapter.engine == "cosmos":
        _print_run_result(adapter, request, _run_cosmos)
        return 0
    if adapter.engine == "kokoro":
        _print_run_result(adapter, request, _run_kokoro)
        return 0
    if adapter.engine == "chatterbox":
        _print_run_result(adapter, request, _run_chatterbox)
        return 0
    if adapter.engine == "dia":
        _print_run_result(adapter, request, _run_dia)
        return 0
    if adapter.engine == "ace-step":
        _print_run_result(adapter, request, _run_acestep)
        return 0
    if adapter.engine == "wan":
        _print_run_result(adapter, request, _run_wan)
        return 0
    if adapter.engine == "ltx":
        _print_run_result(adapter, request, _run_ltx)
        return 0
    if adapter.engine == "artifact":
        _print_run_result(adapter, request, _run_artifact)
        return 0
    print(
        json.dumps(
            _error(
                adapter.engine,
                f"{adapter.engine} bridge adapter is installed but generation is not implemented in this wheel yet",
                "bridge_adapter_not_implemented",
                "",
                request=request,
            )
        )
    )
    return 0


def _experimental_bridge_enabled() -> bool:
    value = os.environ.get("UTOPIC_EXPERIMENTAL_BRIDGE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _print_run_result(adapter: BridgeAdapter, request: dict[str, Any], runner: object) -> None:
    with _redirect_adapter_stdout_to_stderr():
        result = _safe_run(adapter, request, runner)
    print(json.dumps(result))


@contextlib.contextmanager
def _redirect_adapter_stdout_to_stderr():
    saved_stdout_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        if saved_stdout_fd is not None:
            os.close(saved_stdout_fd)
            saved_stdout_fd = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)


def _check_adapter(adapter: BridgeAdapter) -> dict[str, object]:
    missing = _missing_packages(adapter.packages)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "engine": adapter.engine,
        "status": "ready",
        "ready": True,
        "packages": list(adapter.packages),
        "missing": missing,
        "install_hint": adapter.install_hint,
        "description": adapter.description,
    }
    if missing:
        payload["status"] = "missing_dependencies"
        payload["ready"] = False
        return payload
    api_error = _engine_api_error(adapter)
    if api_error is not None:
        payload["status"] = "api_mismatch"
        payload["ready"] = False
        payload["message"] = _bridge_api_error_message(api_error)
    return payload


def _bridge_api_error_message(message: str) -> str:
    if "operator torchvision::nms does not exist" in message:
        return (
            "torch/torchvision versions are incompatible: "
            "operator torchvision::nms does not exist. "
            "Install a matching torch and torchvision pair in the same Python environment. "
            f"Original error: {message}"
        )
    return message


def _validate_bridge_request(adapter: BridgeAdapter, request: object) -> Optional[str]:
    if not isinstance(request, dict):
        return "bridge request must be a JSON object"
    if request.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version must be {SCHEMA_VERSION}"
    engine = request.get("engine")
    if engine != adapter.engine:
        return f"engine must match adapter {adapter.engine}"
    for field in ("model", "modality", "output_dir", "progress_path"):
        value = request.get(field)
        if not isinstance(value, str) or not value:
            return f"{field} must be a non-empty string"
    input_value = request.get("input")
    if not isinstance(input_value, dict) or not input_value:
        return "input must be a non-empty object"
    if request["modality"] == "tts":
        if not isinstance(input_value.get("input"), str) or not input_value.get("input"):
            return "input.input must be a non-empty string"
    elif request["modality"] == "misc":
        if not isinstance(input_value.get("artifact"), str) or not input_value.get("artifact"):
            return "input.artifact must be a non-empty string"
    elif not isinstance(input_value.get("prompt"), str) or not input_value.get("prompt"):
        return "input.prompt must be a non-empty string"
    parameters = request.get("parameters", {})
    if parameters is not None and not isinstance(parameters, dict):
        return "parameters must be an object"
    metadata = request.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        return "metadata must be an object"
    model_cache_path = request.get("model_cache_path")
    if model_cache_path is not None and not isinstance(model_cache_path, str):
        return "model_cache_path must be a string"
    return None


def _engine_api_error(adapter: BridgeAdapter) -> Optional[str]:
    try:
        if adapter.engine in {"diffusers", "wan", "ltx"}:
            import diffusers  # type: ignore

            if adapter.engine == "wan" and not (
                hasattr(diffusers, "WanPipeline") or hasattr(diffusers, "DiffusionPipeline")
            ):
                return "diffusers does not expose WanPipeline or DiffusionPipeline"
            if adapter.engine == "ltx" and not (
                hasattr(diffusers, "LTXPipeline") or hasattr(diffusers, "DiffusionPipeline")
            ):
                return "diffusers does not expose LTXPipeline or DiffusionPipeline"
            if adapter.engine == "diffusers" and not hasattr(diffusers, "DiffusionPipeline"):
                return "diffusers does not expose DiffusionPipeline"
        elif adapter.engine == "kokoro":
            from kokoro import KPipeline  # type: ignore

            if KPipeline is None:
                return "kokoro does not expose KPipeline"
        elif adapter.engine == "chatterbox":
            from chatterbox.tts import ChatterboxTTS  # type: ignore

            if ChatterboxTTS is None:
                return "chatterbox.tts does not expose ChatterboxTTS"
            import perth  # type: ignore

            if getattr(perth, "PerthImplicitWatermarker", None) is None:
                return (
                    "resemble-perth does not expose PerthImplicitWatermarker. "
                    "Install setuptools<81 in the Chatterbox environment; "
                    "newer setuptools releases can remove pkg_resources, which "
                    "resemble-perth still imports."
                )
        elif adapter.engine == "dia":
            from transformers import AutoProcessor, DiaForConditionalGeneration  # type: ignore

            if AutoProcessor is None or DiaForConditionalGeneration is None:
                return "transformers does not expose DiaForConditionalGeneration"
        elif adapter.engine == "ace-step":
            if _acestep_pipeline_class() is None:
                return "acestep.pipeline_ace_step does not expose ACEStepPipeline"
    except Exception as exc:
        return _exception_message_with_causes(exc)
    return None


def _exception_message_with_causes(exc: BaseException) -> str:
    parts: list[str] = []
    current: Optional[BaseException] = exc
    while current is not None:
        message = str(current)
        if message and message not in parts:
            parts.append(message)
        current = current.__cause__ or current.__context__
    return " Root cause: ".join(parts) if parts else exc.__class__.__name__


def _safe_run(adapter: BridgeAdapter, request: dict[str, Any], runner: object) -> dict[str, object]:
    try:
        _reset_progress_file(request)
        return runner(request)  # type: ignore[misc]
    except ImportError as exc:
        missing_dependency = exc.name or str(exc) or exc.__class__.__name__
        return _error(
            adapter.engine,
            f"{adapter.engine} bridge dependencies are not installed: {missing_dependency}",
            "bridge_dependency_missing",
            adapter.install_hint,
            request=request,
        )
    except (AttributeError, TypeError) as exc:
        return _error(
            adapter.engine,
            f"{adapter.engine} bridge could not use the installed package API: {exc}",
            "bridge_adapter_api_mismatch",
            adapter.install_hint,
            request=request,
        )
    except RuntimeError as exc:
        if _looks_like_huggingface_auth_error(str(exc)):
            return _error(
                adapter.engine,
                f"{adapter.engine} bridge requires Hugging Face access for this model: {exc}",
                "bridge_auth_required",
                "Accept the model license on Hugging Face and set HF_TOKEN in this environment.",
                request=request,
            )
        if _looks_like_optional_import_mismatch(str(exc)):
            return _error(
                adapter.engine,
                f"{adapter.engine} bridge could not import a compatible optional engine API: {exc}",
                "bridge_adapter_api_mismatch",
                adapter.install_hint,
                request=request,
            )
        return _error(
            adapter.engine,
            f"{adapter.engine} bridge generation failed: {exc}",
            "bridge_adapter_failed",
            "",
            request=request,
        )
    except Exception as exc:
        if _looks_like_huggingface_auth_error(str(exc)):
            return _error(
                adapter.engine,
                f"{adapter.engine} bridge requires Hugging Face access for this model: {exc}",
                "bridge_auth_required",
                "Accept the model license on Hugging Face and set HF_TOKEN in this environment.",
                request=request,
            )
        return _error(
            adapter.engine,
            f"{adapter.engine} bridge generation failed: {exc}",
            "bridge_adapter_failed",
            "",
            request=request,
        )


def _looks_like_optional_import_mismatch(message: str) -> bool:
    return (
        "Failed to import" in message
        or "Could not import module" in message
        or "requirements defined correctly" in message
    )


def _looks_like_huggingface_auth_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "huggingface.co" in lowered
        and (
            "401 client error" in lowered
            or "cannot access gated repo" in lowered
            or "access to model" in lowered and "restricted" in lowered
            or "please log in" in lowered
        )
    )


def _run_artifact(request: dict[str, Any]) -> dict[str, object]:
    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(_artifact_input_from_request(request)).expanduser()
    if not source.is_file():
        raise RuntimeError(f"artifact input does not exist: {source}")
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    metadata = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    _write_progress(progress_path, "loading", 0.1, "loading input artifact")
    destination = output_dir / _safe_output_artifact_name(source.name)
    _write_progress(progress_path, "generating", 0.5, "copying artifact")
    destination.write_bytes(source.read_bytes())
    _write_progress(progress_path, "completed", 1.0, "artifact saved")
    artifact_type = parameters.get("artifact_type") if isinstance(parameters.get("artifact_type"), str) else ""
    if not artifact_type:
        outputs = metadata.get("outputs")
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], str):
            artifact_type = outputs[0]
    if not artifact_type:
        artifact_type = "application/octet-stream"
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "artifact",
        "artifacts": [
            {
                "type": artifact_type,
                "path": str(destination),
                "metadata": {
                    "engine": "artifact",
                    "source": str(source),
                },
            }
        ],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": "artifact",
            "source": str(source),
        },
    }


def _run_diffusers(request: dict[str, Any]) -> dict[str, object]:
    try:
        import diffusers  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        return _error(
            "diffusers",
            f"diffusers bridge dependencies are not installed: {exc.name}",
            "bridge_dependency_missing",
            ADAPTERS["diffusers"].install_hint,
            request=request,
        )
    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = str(request.get("model") or "")
    source, source_kwargs = _model_source_and_kwargs(request, default_source=model_id)
    prompt = _prompt_from_request(request)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    call_kwargs = _diffusers_call_kwargs(model_id, parameters)
    _write_progress(progress_path, "loading", 0.1, "loading diffusers pipeline")
    pipeline_cls = _diffusers_pipeline_class(diffusers, model_id)
    pipeline = pipeline_cls.from_pretrained(source, torch_dtype=_torch_dtype(torch), **source_kwargs)
    if _disable_diffusers_safety_checker(parameters):
        _disable_pipeline_safety_checker(pipeline)
    device = _torch_device(torch)
    if _enable_model_cpu_offload(model_id, parameters, pipeline):
        pipeline.enable_model_cpu_offload()
    elif device:
        pipeline = pipeline.to(device)
    _write_progress(progress_path, "generating", 0.5, "generating image")
    result = pipeline(prompt, **call_kwargs)
    image = _first_image(result)
    output_path = output_dir / "image.png"
    image.save(output_path)
    _write_progress(progress_path, "completed", 1.0, "image saved")
    return {
        "artifacts": [
            {
                "type": "image/png",
                "path": str(output_path),
                "metadata": {
                    "model": model_id,
                    "engine": "diffusers",
                },
            }
        ],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": "diffusers",
            "device": device or "cpu",
        },
    }


def _run_cosmos(request: dict[str, Any]) -> dict[str, object]:
    return _error(
        "cosmos",
        (
            "cosmos bridge dependencies are installed, but direct Cosmos3 generation "
            "is not implemented in this wheel yet. Use an external Cosmos/SGLang "
            "bridge command through UTOPIC_BRIDGE_COSMOS_COMMAND."
        ),
        "bridge_adapter_not_implemented",
        ADAPTERS["cosmos"].install_hint,
        request=request,
    )


def _run_kokoro(request: dict[str, Any]) -> dict[str, object]:
    try:
        from kokoro import KPipeline  # type: ignore
        import soundfile as sf  # type: ignore
    except ImportError as exc:
        return _error(
            "kokoro",
            f"kokoro bridge dependencies are not installed: {exc.name}",
            "bridge_dependency_missing",
            ADAPTERS["kokoro"].install_hint,
            request=request,
        )
    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    text = _text_input_from_request(request)
    voice = str(parameters.get("voice") or "af_heart")
    lang_code = str(parameters.get("lang_code") or "a")
    speed = float(parameters.get("speed") or 1.0)
    sample_rate = int(parameters.get("sample_rate") or 24000)
    _write_progress(progress_path, "loading", 0.1, "loading kokoro pipeline")
    pipeline = KPipeline(lang_code=lang_code)
    _write_progress(progress_path, "generating", 0.5, "generating speech")
    generated = pipeline(text, voice=voice, speed=speed)
    audio = _first_kokoro_audio(generated)
    output_path = output_dir / "speech.wav"
    sf.write(output_path, audio, sample_rate)
    _write_progress(progress_path, "completed", 1.0, "speech saved")
    return {
        "artifacts": [
            {
                "type": "audio/wav",
                "path": str(output_path),
                "metadata": {
                    "model": str(request.get("model") or ""),
                    "engine": "kokoro",
                    "voice": voice,
                    "sample_rate": sample_rate,
                },
            }
        ],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": "kokoro",
        },
    }


def _run_chatterbox(request: dict[str, Any]) -> dict[str, object]:
    from chatterbox.tts import ChatterboxTTS  # type: ignore
    import soundfile as sf  # type: ignore

    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    text = _text_input_from_request(request)
    device = _device_for_python_model()
    _write_progress(progress_path, "loading", 0.1, "loading chatterbox pipeline")
    model = ChatterboxTTS.from_pretrained(device=device)
    _write_progress(progress_path, "generating", 0.5, "generating speech")
    audio = model.generate(text, **_clean_audio_parameters(parameters))
    sample_rate = int(parameters.get("sample_rate") or getattr(model, "sr", 24000))
    output_path = output_dir / "speech.wav"
    sf.write(output_path, _audio_array(audio), sample_rate)
    _write_progress(progress_path, "completed", 1.0, "speech saved")
    return _audio_response(
        request,
        "chatterbox",
        output_path,
        "audio/wav",
        {"sample_rate": sample_rate},
    )


def _run_dia(request: dict[str, Any]) -> dict[str, object]:
    import torch  # type: ignore
    from transformers import AutoProcessor, DiaForConditionalGeneration  # type: ignore

    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    text = _dia_text(_text_input_from_request(request))
    source, source_kwargs = _model_source_and_kwargs(request, default_source="nari-labs/Dia-1.6B-0626")
    device = _torch_device(torch)
    _write_progress(progress_path, "loading", 0.1, "loading dia processor and model")
    processor = AutoProcessor.from_pretrained(source, **source_kwargs)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    model = DiaForConditionalGeneration.from_pretrained(source, **source_kwargs).to(device)
    _write_progress(progress_path, "generating", 0.5, "generating speech")
    outputs = model.generate(**dict(inputs), **_dia_generate_kwargs(parameters))
    decoded = processor.batch_decode(outputs)
    output_path = output_dir / "speech.wav"
    processor.save_audio(decoded, str(output_path))
    _write_progress(progress_path, "completed", 1.0, "speech saved")
    return _audio_response(request, "dia", output_path, "audio/wav", {"device": device})


def _run_acestep(request: dict[str, Any]) -> dict[str, object]:
    import soundfile as sf  # type: ignore

    pipeline_cls = _acestep_pipeline_class()
    if pipeline_cls is None:
        raise AttributeError("expected acestep.pipeline_ace_step.ACEStepPipeline")
    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    prompt = _prompt_from_request(request)
    device = _device_for_python_model()
    checkpoint_path = str(request.get("model_cache_path") or "")
    _write_progress(progress_path, "loading", 0.1, "loading ace-step pipeline")
    pipeline = pipeline_cls(
        checkpoint_dir=checkpoint_path,
        dtype=str(parameters.get("dtype") or "bfloat16"),
        torch_compile=bool(parameters.get("torch_compile") or False),
        cpu_offload=bool(parameters.get("cpu_offload") or False),
        overlapped_decode=bool(parameters.get("overlapped_decode") or False),
    )
    output_path = output_dir / "music.wav"
    call_kwargs = _acestep_call_kwargs(parameters, prompt)
    call_kwargs.setdefault("save_path", str(output_path))
    saved_path = Path(str(call_kwargs["save_path"])).expanduser()
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    _write_progress(progress_path, "generating", 0.5, "generating music")
    result = pipeline(**call_kwargs)
    if saved_path.exists():
        _write_progress(progress_path, "completed", 1.0, "music saved")
        return _audio_response(
            request,
            "ace-step",
            saved_path,
            "audio/wav",
            {"sample_rate": int(parameters.get("sample_rate") or 44100), "device": device},
        )
    audio, sample_rate = _audio_and_sample_rate(result, default_sample_rate=44100)
    sf.write(output_path, _audio_array(audio), sample_rate)
    _write_progress(progress_path, "completed", 1.0, "music saved")
    return _audio_response(
        request,
        "ace-step",
        output_path,
        "audio/wav",
        {"sample_rate": sample_rate, "device": device},
    )


def _run_wan(request: dict[str, Any]) -> dict[str, object]:
    import diffusers  # type: ignore
    import torch  # type: ignore
    from diffusers.utils import export_to_video  # type: ignore

    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    source, source_kwargs = _model_source_and_kwargs(
        request,
        default_source=_wan_repo_for_model(str(request.get("model") or "")),
    )
    device = _torch_device(torch)
    _write_progress(progress_path, "loading", 0.1, "loading wan pipeline")
    pipeline_cls = getattr(diffusers, "WanPipeline", getattr(diffusers, "DiffusionPipeline"))
    pipeline = pipeline_cls.from_pretrained(source, torch_dtype=_torch_dtype(torch), **source_kwargs)
    if device:
        pipeline = pipeline.to(device)
    _write_progress(progress_path, "generating", 0.5, "generating video")
    result = pipeline(prompt=_prompt_from_request(request), **_wan_call_kwargs(parameters))
    frames = _first_video_frames(result)
    output_path = output_dir / "video.mp4"
    export_to_video(frames, output_path, fps=int(parameters.get("fps") or 16))
    _write_progress(progress_path, "completed", 1.0, "video saved")
    return {
        "artifacts": [
            {
                "type": "video/mp4",
                "path": str(output_path),
                "metadata": {
                    "model": str(request.get("model") or ""),
                    "engine": "wan",
                    "device": device,
                },
            }
        ],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": "wan",
            "device": device,
        },
    }


def _run_ltx(request: dict[str, Any]) -> dict[str, object]:
    import diffusers  # type: ignore
    import torch  # type: ignore
    from diffusers.utils import export_to_video  # type: ignore

    output_dir = Path(str(request.get("output_dir") or ".")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(request.get("progress_path") or output_dir / "progress.jsonl")).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = request.get("parameters") if isinstance(request.get("parameters"), dict) else {}
    source, source_kwargs = _model_source_and_kwargs(request, default_source="Lightricks/LTX-Video")
    device = _torch_device(torch)
    _write_progress(progress_path, "loading", 0.1, "loading ltx pipeline")
    pipeline_cls = getattr(diffusers, "LTXPipeline", getattr(diffusers, "DiffusionPipeline"))
    pipeline = pipeline_cls.from_pretrained(source, torch_dtype=_torch_dtype(torch), **source_kwargs)
    if device:
        pipeline = pipeline.to(device)
    _write_progress(progress_path, "generating", 0.5, "generating video")
    result = pipeline(prompt=_prompt_from_request(request), **_video_call_kwargs(parameters))
    frames = _first_video_frames(result)
    output_path = output_dir / "video.mp4"
    export_to_video(frames, output_path, fps=int(parameters.get("fps") or 16))
    _write_progress(progress_path, "completed", 1.0, "video saved")
    return {
        "artifacts": [
            {
                "type": "video/mp4",
                "path": str(output_path),
                "metadata": {
                    "model": str(request.get("model") or ""),
                    "engine": "ltx",
                    "device": device,
                },
            }
        ],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": "ltx",
            "device": device,
        },
    }


def _audio_response(
    request: dict[str, Any],
    engine: str,
    output_path: Path,
    content_type: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "artifacts": [
            {
                "type": content_type,
                "path": str(output_path),
                "metadata": {
                    "model": str(request.get("model") or ""),
                    "engine": engine,
                    **metadata,
                },
            }
        ],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": engine,
        },
    }


def _acestep_pipeline_class() -> object:
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore
    except ImportError:
        return None
    return ACEStepPipeline


def _first_kokoro_audio(generated: object) -> object:
    for item in generated:
        if isinstance(item, tuple) and len(item) >= 3:
            return item[2]
        if isinstance(item, dict) and "audio" in item:
            return item["audio"]
        audio = getattr(item, "audio", None)
        if audio is not None:
            return audio
        output = getattr(item, "output", None)
        output_audio = getattr(output, "audio", None)
        if output_audio is not None:
            return output_audio
    raise RuntimeError("kokoro pipeline returned no audio")


def _audio_array(audio: object) -> object:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    shape = getattr(audio, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == 1:
        return audio[0]  # type: ignore[index]
    return audio


def _audio_and_sample_rate(result: object, *, default_sample_rate: int) -> tuple[object, int]:
    if isinstance(result, dict):
        audio = result.get("audio") or result.get("wav") or result.get("samples")
        sample_rate = int(result.get("sample_rate") or result.get("sr") or default_sample_rate)
        if audio is not None:
            return audio, sample_rate
    if isinstance(result, tuple) and result:
        audio = result[0]
        sample_rate = int(result[1]) if len(result) > 1 and isinstance(result[1], int) else default_sample_rate
        return audio, sample_rate
    return result, default_sample_rate


def _model_source_and_kwargs(
    request: dict[str, Any],
    *,
    default_source: str,
) -> tuple[str, dict[str, object]]:
    cache_path_value = request.get("model_cache_path")
    repo_value = request.get("repo")
    cache_path = (
        Path(str(cache_path_value)).expanduser()
        if isinstance(cache_path_value, str) and cache_path_value
        else None
    )
    has_repo = isinstance(repo_value, str) and bool(repo_value)
    repo = str(repo_value) if has_repo else default_source
    if not has_repo and isinstance(cache_path_value, str) and cache_path_value:
        return cache_path_value, {}
    if cache_path is not None and _looks_like_model_directory(cache_path):
        return str(cache_path), {}
    if repo:
        kwargs: dict[str, object] = {}
        if cache_path is not None:
            kwargs["cache_dir"] = str(cache_path)
        return repo, kwargs
    if cache_path is not None:
        return str(cache_path), {}
    return default_source, {}


def _looks_like_model_directory(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    model_markers = {
        "model_index.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    }
    for child in path.iterdir():
        if child.name in model_markers:
            return True
        if child.suffix in {".safetensors", ".bin", ".pt", ".pth", ".gguf"}:
            return True
    return False


def _diffusers_pipeline_class(diffusers_module: object, model_id: str) -> object:
    if model_id.startswith("qwen-image") and hasattr(diffusers_module, "QwenImagePipeline"):
        return getattr(diffusers_module, "QwenImagePipeline")
    if model_id.startswith("flux") and hasattr(diffusers_module, "FluxPipeline"):
        return getattr(diffusers_module, "FluxPipeline")
    if model_id.startswith("krea-2") and hasattr(diffusers_module, "Krea2Pipeline"):
        return getattr(diffusers_module, "Krea2Pipeline")
    return getattr(diffusers_module, "DiffusionPipeline")


def _torch_dtype(torch_module: object) -> object:
    return getattr(torch_module, "bfloat16", getattr(torch_module, "float16", None))


def _torch_device(torch_module: object) -> str:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda"
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"
    return "cpu"


def _device_for_python_model() -> str:
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    return _torch_device(torch)


def _diffusers_call_kwargs(model_id: str, parameters: dict[str, object]) -> dict[str, object]:
    excluded = {"size", "seed", "response_format", "disable_safety_checker", "enable_model_cpu_offload"}
    kwargs = {key: value for key, value in parameters.items() if key not in excluded}
    if model_id.startswith("qwen-image"):
        true_cfg_scale = kwargs.get("true_cfg_scale")
        if true_cfg_scale is None:
            kwargs["true_cfg_scale"] = 4.0
        elif isinstance(true_cfg_scale, (int, float)) and true_cfg_scale <= 1.0:
            raise RuntimeError(
                "qwen-image true_cfg_scale must be greater than 1.0; "
                "omit it for the default 4.0"
            )
        num_steps = kwargs.get("num_inference_steps")
        if isinstance(num_steps, int) and num_steps < 2:
            raise RuntimeError(
                "qwen-image num_inference_steps must be at least 2; "
                "omit it for the Diffusers default"
            )
        if "negative_prompt" not in kwargs:
            kwargs["negative_prompt"] = " "
    if model_id.startswith("flux"):
        kwargs.setdefault("guidance_scale", 0.0)
        kwargs.setdefault("num_inference_steps", 4)
        kwargs.setdefault("max_sequence_length", 256)
    size = parameters.get("size")
    if isinstance(size, str) and "x" in size:
        width, height = size.lower().split("x", 1)
        if width.isdigit() and height.isdigit():
            kwargs["width"] = int(width)
            kwargs["height"] = int(height)
    return kwargs


def _enable_model_cpu_offload(model_id: str, parameters: dict[str, object], pipeline: object) -> bool:
    requested = parameters.get("enable_model_cpu_offload")
    should_enable = requested is True or (requested is None and model_id.startswith("flux"))
    if not should_enable:
        return False
    if not callable(getattr(pipeline, "enable_model_cpu_offload", None)):
        if requested is True:
            raise RuntimeError("pipeline does not support enable_model_cpu_offload")
        return False
    return True


def _disable_diffusers_safety_checker(parameters: dict[str, object]) -> bool:
    return parameters.get("disable_safety_checker") is True


def _disable_pipeline_safety_checker(pipeline: object) -> None:
    if hasattr(pipeline, "safety_checker"):
        setattr(pipeline, "safety_checker", None)
    if hasattr(pipeline, "requires_safety_checker"):
        setattr(pipeline, "requires_safety_checker", False)
    register = getattr(pipeline, "register_to_config", None)
    if callable(register):
        register(requires_safety_checker=False)


def _clean_audio_parameters(parameters: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in parameters.items() if key not in {"sample_rate"}}


def _acestep_call_kwargs(parameters: dict[str, object], prompt: str) -> dict[str, object]:
    excluded = {
        "dtype",
        "torch_compile",
        "cpu_offload",
        "overlapped_decode",
        "duration",
        "sample_rate",
    }
    kwargs = {key: value for key, value in parameters.items() if key not in excluded}
    kwargs["prompt"] = prompt
    if "audio_duration" not in kwargs and "duration" in parameters:
        kwargs["audio_duration"] = parameters["duration"]
    return kwargs


def _dia_generate_kwargs(parameters: dict[str, object]) -> dict[str, object]:
    defaults = {
        "max_new_tokens": 3072,
        "guidance_scale": 3.0,
        "temperature": 1.8,
        "top_p": 0.90,
        "top_k": 45,
    }
    return {**defaults, **parameters}


def _dia_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[S1]") or stripped.startswith("[S2]"):
        return stripped
    return f"[S1] {stripped}"


def _wan_repo_for_model(model_id: str) -> str:
    if model_id.endswith("14b"):
        return "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    return "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _wan_call_kwargs(parameters: dict[str, object]) -> dict[str, object]:
    return _video_call_kwargs(parameters)


def _video_call_kwargs(parameters: dict[str, object]) -> dict[str, object]:
    excluded = {"fps", "response_format", "size"}
    kwargs = {key: value for key, value in parameters.items() if key not in excluded}
    _add_size_kwargs(parameters, kwargs)
    return kwargs


def _add_size_kwargs(parameters: dict[str, object], kwargs: dict[str, object]) -> None:
    size = parameters.get("size")
    if isinstance(size, str) and "x" in size:
        width, height = size.lower().split("x", 1)
        if width.isdigit() and height.isdigit():
            kwargs["width"] = int(width)
            kwargs["height"] = int(height)


def _first_video_frames(result: object) -> object:
    frames = getattr(result, "frames", None)
    first = _first_batched_item(frames)
    if first is not None:
        return first
    if isinstance(result, dict):
        frames = result.get("frames")
        first = _first_batched_item(frames)
        if first is not None:
            return first
    raise RuntimeError("video pipeline returned no frames")


def _first_batched_item(value: object) -> object:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        if len(value) <= 0:  # type: ignore[arg-type]
            return None
        return value[0]  # type: ignore[index]
    except (AssertionError, TypeError, KeyError, IndexError):
        return None


def _first_image(result: object) -> object:
    images = getattr(result, "images", None)
    if isinstance(images, list) and images:
        return images[0]
    if isinstance(result, dict):
        images = result.get("images")
        if isinstance(images, list) and images:
            return images[0]
    raise RuntimeError("diffusers pipeline returned no images")


def _prompt_from_request(request: dict[str, Any]) -> str:
    input_value = request.get("input")
    if isinstance(input_value, dict):
        prompt = input_value.get("prompt") or input_value.get("input")
        if isinstance(prompt, str):
            return prompt
    if isinstance(input_value, str):
        return input_value
    return ""


def _text_input_from_request(request: dict[str, Any]) -> str:
    input_value = request.get("input")
    if isinstance(input_value, dict):
        value = input_value.get("input") or input_value.get("prompt")
        if isinstance(value, str):
            return value
    if isinstance(input_value, str):
        return input_value
    return ""


def _artifact_input_from_request(request: dict[str, Any]) -> str:
    input_value = request.get("input")
    if isinstance(input_value, dict):
        value = input_value.get("artifact") or input_value.get("input_file") or input_value.get("path")
        if isinstance(value, str):
            return value
    if isinstance(input_value, str):
        return input_value
    return ""


def _safe_output_artifact_name(name: str) -> str:
    candidate = Path(name).name
    if not candidate or candidate in {".", ".."}:
        return "artifact.bin"
    return candidate


def _write_progress(path: Path, event: str, progress: float, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "progress": progress, "message": message}) + "\n")


def _reset_progress_file(request: dict[str, Any]) -> None:
    progress_path = request.get("progress_path")
    if not isinstance(progress_path, str) or not progress_path:
        return
    path = Path(progress_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _missing_packages(packages: tuple[str, ...]) -> list[str]:
    return [package for package in packages if importlib.util.find_spec(package) is None]


def _error(
    engine: str,
    message: str,
    code: str,
    install_hint: str,
    *,
    request: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    model = request.get("model") if isinstance(request, dict) else None
    modality = request.get("modality") if isinstance(request, dict) else None
    return {
        "error": {
            "code": code,
            "message": message,
            "engine": engine,
            "install_hint": install_hint,
        },
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "modality": modality,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
