import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utopic import __version__, bridge, cli, gateway

REPO_ROOT = Path(__file__).resolve().parents[1]


def decode(response):
    status, headers, body = response
    assert headers["content-type"] == "application/json"
    return status, json.loads(body.decode("utf-8"))


def test_gateway_module_entrypoint_prints_help():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "python")}
    result = subprocess.run(
        [sys.executable, "-m", "utopic.gateway", "--help"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "utopic-runtime" in result.stdout
    assert "--port" in result.stdout


def test_gateway_models_endpoint_exposes_multimodal_runtime_metadata():
    status, payload = decode(gateway.handle_openai_request("GET", "/v1/models", None))

    assert status == 200
    by_id = {item["id"]: item for item in payload["data"]}
    assert by_id["diffusiongemma-26b-a4b-q4"]["modality"] == "text"
    assert by_id["diffusiongemma-26b-a4b-q4"]["runtime"] == "native"
    assert by_id["qwen-image"]["modality"] == "image"
    assert by_id["qwen-image"]["runtime"] == "bridge"
    assert by_id["qwen-image"]["repo"] == "Qwen/Qwen-Image"
    assert by_id["qwen-image"]["url"] == "https://huggingface.co/Qwen/Qwen-Image"
    assert "/v1/images/generations" in by_id["qwen-image"]["endpoints"]
    assert by_id["qwen-image"]["bridge"] == {
        "schema_version": "utopic-bridge/v1",
        "engine": "diffusers",
        "command": "utopic-bridge diffusers",
        "environment_variable": "UTOPIC_BRIDGE_DIFFUSERS_COMMAND",
        "install_hint": 'pip install "utopic[image]"',
        "input": "prompt",
        "outputs": ["image/png"],
        "progress_events": ["queued", "loading", "generating", "completed", "failed"],
    }


def test_gateway_models_endpoint_exposes_bridge_activation_for_all_bridge_models():
    status, payload = decode(gateway.handle_openai_request("GET", "/v1/models", None))

    assert status == 200
    bridge_models = [item for item in payload["data"] if item["runtime"] == "bridge"]

    assert {item["id"] for item in bridge_models} >= {
        "qwen-image",
        "flux-1-schnell",
        "kokoro-82m",
        "chatterbox",
        "dia-1.6b",
        "ace-step-3.5b",
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b",
        "ltx-video",
    }
    for item in bridge_models:
        assert item["bridge"]["schema_version"] == "utopic-bridge/v1"
        assert item["bridge"]["engine"] == item["engine"]
        assert item["bridge"]["command"] == f"utopic-bridge {item['engine']}"
        assert item["bridge"]["environment_variable"].startswith("UTOPIC_BRIDGE_")
        assert item["bridge"]["install_hint"]
        assert item["bridge"]["input"] in {"prompt", "input"}
        assert item["bridge"]["outputs"] == item["outputs"]

    by_id = {item["id"]: item for item in bridge_models}
    assert by_id["ltx-video"]["repo"] == "Lightricks/LTX-Video"
    assert by_id["ltx-video"]["bridge"]["command"] == "utopic-bridge ltx"
    assert by_id["ltx-video"]["bridge"]["environment_variable"] == "UTOPIC_BRIDGE_LTX_COMMAND"


def test_gateway_image_generation_reports_packaged_bridge_dependency_gap():
    request = {
        "model": "qwen-image",
        "prompt": "a precise product photo of a glass teapot",
        "size": "1024x1024",
    }

    status, payload = decode(
        gateway.handle_openai_request("POST", "/v1/images/generations", request)
    )

    assert status == 501
    assert payload["error"]["code"] == "bridge_dependency_missing"
    assert payload["error"]["model"] == "qwen-image"
    assert payload["error"]["modality"] == "image"
    assert payload["error"]["engine"] == "diffusers"
    assert payload["error"]["install_hint"] == 'pip install "utopic[image]"'
    assert "diffusers bridge dependencies are not installed" in payload["error"]["message"]


def test_gateway_uses_packaged_bridge_command_by_default_for_bridge_models(monkeypatch):
    monkeypatch.delenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", raising=False)
    monkeypatch.delenv("UTOPIC_BRIDGE_COMMAND", raising=False)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "a precise product photo of a glass teapot"},
        )
    )

    assert status == 501
    assert payload["error"]["code"] == "bridge_dependency_missing"
    assert payload["error"]["engine"] == "diffusers"
    assert payload["error"]["model"] == "qwen-image"
    assert payload["error"]["modality"] == "image"
    assert payload["error"]["install_hint"] == 'pip install "utopic[image]"'


def test_gateway_exposes_openai_routes_for_each_bridge_modality():
    cases = [
        (
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "a teapot"},
            "image",
            "diffusers",
            'pip install "utopic[image]"',
        ),
        (
            "/v1/audio/speech",
            {"model": "kokoro-82m", "input": "hello"},
            "tts",
            "kokoro",
            (
                'pip install "utopic[tts]" && python -m pip install '
                "https://github.com/explosion/spacy-models/releases/download/"
                "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
            ),
        ),
        (
            "/v1/audio/speech",
            {"model": "chatterbox", "input": "hello"},
            "tts",
            "chatterbox",
            'pip install "utopic[chatterbox]"',
        ),
        (
            "/v1/audio/generations",
            {"model": "ace-step-3.5b", "prompt": "ambient piano"},
            "music",
            "ace-step",
            'pip install "utopic[music]" && pip install git+https://github.com/ace-step/ACE-Step.git',
        ),
        (
            "/v1/videos/generations",
            {"model": "wan2.1-t2v-1.3b", "prompt": "waves"},
            "video",
            "wan",
            'pip install "utopic[video]"',
        ),
        (
            "/v1/responses",
            {"model": "flux-1-schnell", "prompt": "a red cube"},
            "image",
            "diffusers",
            'pip install "utopic[image]"',
        ),
    ]

    for endpoint, request, modality, engine, install_hint in cases:
        status, payload = decode(gateway.handle_openai_request("POST", endpoint, request))

        assert status == 501
        assert payload["error"]["code"] == "bridge_dependency_missing"
        assert payload["error"]["modality"] == modality
        assert payload["error"]["engine"] == engine
        assert payload["error"]["install_hint"] == install_hint


def test_every_bridge_catalog_model_has_openai_and_mcp_runtime_surface():
    generation_tool_by_modality = {
        "image": "utopic_generate_image",
        "tts": "utopic_speak",
        "music": "utopic_generate_music",
        "video": "utopic_generate_video",
    }
    request_by_modality = {
        "image": {"prompt": "a small ceramic cup"},
        "tts": {"input": "hello from utopic"},
        "music": {"prompt": "quiet piano"},
        "video": {"prompt": "waves rolling onto a beach"},
    }

    bridge_models = [entry for entry in gateway.models.list_models() if entry.runtime == "bridge"]
    assert bridge_models

    for entry in bridge_models:
        assert entry.modality in generation_tool_by_modality
        assert "/v1/responses" in entry.endpoints
        assert any(endpoint != "/v1/responses" for endpoint in entry.endpoints)

        modality_endpoint = next(endpoint for endpoint in entry.endpoints if endpoint != "/v1/responses")
        request = {"model": entry.id, **request_by_modality[entry.modality]}
        status, payload = decode(gateway.handle_openai_request("POST", modality_endpoint, request))

        assert status in {501, 502}, entry.id
        assert payload["error"]["model"] == entry.id
        assert payload["error"]["modality"] == entry.modality
        assert payload["error"]["engine"] == entry.engine
        assert payload["error"]["code"].startswith("bridge_")

        responses_request = {
            "model": entry.id,
            "input": request_by_modality[entry.modality].get("prompt")
            or request_by_modality[entry.modality]["input"],
        }
        status, payload = decode(gateway.handle_openai_request("POST", "/v1/responses", responses_request))

        assert status in {501, 502}, entry.id
        assert payload["error"]["model"] == entry.id
        assert payload["error"]["modality"] == entry.modality
        assert payload["error"]["engine"] == entry.engine

        status, payload = decode(
            gateway.handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": entry.id,
                    "method": "tools/call",
                    "params": {
                        "name": generation_tool_by_modality[entry.modality],
                        "arguments": request,
                    },
                }
            )
        )

        assert status == 200
        assert payload["result"]["isError"] is True
        tool_payload = json.loads(payload["result"]["content"][0]["text"])
        assert tool_payload["error"]["model"] == entry.id
        assert tool_payload["error"]["modality"] == entry.modality
        assert tool_payload["error"]["engine"] == entry.engine


def test_gateway_invokes_configured_bridge_and_returns_artifact_contract(tmp_path, monkeypatch):
    script = tmp_path / "fake_bridge.py"
    script.write_text(
        """
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
out_dir = pathlib.Path(request["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / "image.png"
artifact.write_bytes(b"png")
with pathlib.Path(request["progress_path"]).open("a", encoding="utf-8") as progress:
    progress.write(json.dumps({"event": "generating", "progress": 0.5, "message": "half"}) + "\\n")
    progress.write(json.dumps({"event": "completed", "progress": 1.0, "message": "done"}) + "\\n")
print(json.dumps({
    "artifacts": [{"type": "image/png", "path": str(artifact), "metadata": {"seed": 7}}],
    "metadata": {
        "engine_version": "fake",
        "repo": request["repo"],
        "model_cache_path": request["model_cache_path"],
        "metadata": request["metadata"],
    }
}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {
                "model": "qwen-image",
                "prompt": "a precise product photo of a glass teapot",
                "size": "1024x1024",
                "seed": 7,
            },
        )
    )

    assert status == 200
    assert payload["object"] == "utopic.artifact.response"
    assert payload["model"] == "qwen-image"
    assert payload["modality"] == "image"
    assert payload["engine"] == "diffusers"
    assert payload["progress_url"].startswith("/v1/utopic/runs/")
    assert payload["artifacts"] == [
        {
            "type": "image/png",
            "path": payload["artifacts"][0]["path"],
            "url": f"file://{payload['artifacts'][0]['path']}",
            "metadata": {"seed": 7},
        }
    ]
    assert pathlib_path_exists(payload["artifacts"][0]["path"])
    assert payload["data"] == [{"url": f"file://{payload['artifacts'][0]['path']}"}]
    assert payload["metadata"] == {
        "engine_version": "fake",
        "repo": "Qwen/Qwen-Image",
        "model_cache_path": str(gateway.model_cache_path("qwen-image")),
        "metadata": {
            "outputs": ["image/png"],
            "hardware": ["gb10", "cuda"],
            "repo": "Qwen/Qwen-Image",
            "url": "https://huggingface.co/Qwen/Qwen-Image",
        },
    }
    assert payload["progress"] == [
        {"event": "generating", "progress": 0.5, "message": "half"},
        {"event": "completed", "progress": 1.0, "message": "done"},
    ]

    status, progress_payload = decode(
        gateway.handle_openai_request("GET", payload["progress_url"], None)
    )

    assert status == 200
    assert progress_payload == {
        "object": "list",
        "data": payload["progress"],
    }


def test_gateway_image_generation_supports_b64_json_response_format(tmp_path, monkeypatch):
    script = tmp_path / "fake_bridge.py"
    script.write_text(
        """
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
out_dir = pathlib.Path(request["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / "image.png"
artifact.write_bytes(b"png")
print(json.dumps({
    "artifacts": [{"type": "image/png", "path": str(artifact), "metadata": {"seed": 11}}],
    "metadata": {"engine_version": "fake"}
}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {
                "model": "qwen-image",
                "prompt": "a precise product photo of a glass teapot",
                "response_format": "b64_json",
            },
        )
    )

    assert status == 200
    assert payload["object"] == "utopic.artifact.response"
    assert payload["artifacts"][0]["url"].startswith("file://")
    assert payload["data"] == [{"b64_json": base64.b64encode(b"png").decode("ascii")}]


def test_gateway_rejects_bridge_artifacts_outside_run_output_dir(tmp_path, monkeypatch):
    script = tmp_path / "bad_bridge.py"
    outside_path = tmp_path / "outside.png"
    script.write_text(
        f"""
import json
import pathlib

path = pathlib.Path({str(outside_path)!r})
path.write_bytes(b"png")
print(json.dumps({{
    "artifacts": [{{"type": "image/png", "path": str(path), "metadata": {{}}}}],
    "metadata": {{"engine_version": "bad"}}
}}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "test"},
        )
    )

    assert status == 502
    assert payload["error"]["code"] == "bridge_engine_failed"
    assert "bridge returned no artifacts" in payload["error"]["message"]


def test_gateway_rejects_bridge_artifacts_with_undeclared_output_type(tmp_path, monkeypatch):
    script = tmp_path / "wrong_type_bridge.py"
    script.write_text(
        """
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
out_dir = pathlib.Path(request["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / "image.jpg"
path.write_bytes(b"jpg")
print(json.dumps({
    "artifacts": [{"type": "image/jpeg", "path": str(path), "metadata": {}}],
    "metadata": {"engine_version": "wrong-type"}
}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "test"},
        )
    )

    assert status == 502
    assert payload["error"]["code"] == "bridge_engine_failed"
    assert "unsupported artifact type image/jpeg" in payload["error"]["message"]


def test_responses_endpoint_normalizes_structured_input_for_image_bridge(tmp_path, monkeypatch):
    script = tmp_path / "fake_image_bridge.py"
    captured_path = tmp_path / "captured.json"
    script.write_text(
        f"""
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
pathlib.Path({str(captured_path)!r}).write_text(json.dumps(request), encoding="utf-8")
out_dir = pathlib.Path(request["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / "image.png"
artifact.write_bytes(b"png")
print(json.dumps({{
    "artifacts": [{{"type": "image/png", "path": str(artifact), "metadata": {{}}}}],
    "metadata": {{"schema_version": "utopic-bridge/v1", "engine": "diffusers"}}
}}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/responses",
            {
                "model": "flux-1-schnell",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "a red cube on a steel table"}
                        ],
                    }
                ],
                "size": "512x512",
            },
        )
    )

    assert status == 200
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["type"] == "output_image"
    captured = json.loads(captured_path.read_text(encoding="utf-8"))
    assert captured["endpoint"] == "/v1/responses"
    assert captured["input"] == {"prompt": "a red cube on a steel table"}
    assert captured["parameters"]["size"] == "512x512"


def test_responses_endpoint_normalizes_structured_input_for_tts_bridge(tmp_path, monkeypatch):
    script = tmp_path / "fake_tts_bridge.py"
    captured_path = tmp_path / "captured.json"
    script.write_text(
        f"""
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
pathlib.Path({str(captured_path)!r}).write_text(json.dumps(request), encoding="utf-8")
out_dir = pathlib.Path(request["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / "speech.wav"
artifact.write_bytes(b"wav")
print(json.dumps({{
    "artifacts": [{{"type": "audio/wav", "path": str(artifact), "metadata": {{"voice": "af_heart"}}}}],
    "metadata": {{"schema_version": "utopic-bridge/v1", "engine": "kokoro"}}
}}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_KOKORO_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/responses",
            {
                "model": "kokoro-82m",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "say hello to local users"}
                        ],
                    }
                ],
                "voice": "af_heart",
            },
        )
    )

    assert status == 200
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["type"] == "output_audio"
    captured = json.loads(captured_path.read_text(encoding="utf-8"))
    assert captured["input"] == {"input": "say hello to local users"}
    assert captured["parameters"]["voice"] == "af_heart"


def test_responses_endpoint_for_native_text_uses_chat_proxy_and_wraps_response(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return 200

        def read(self):
            return (
                b'{"id":"chatcmpl-test","created":123,'
                b'"choices":[{"message":{"content":"hello from native"}}]}'
            )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(gateway.urllib.request, "urlopen", fake_urlopen)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/responses",
            {
                "model": "diffusiongemma-26b-a4b-q4",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
            },
            native_base_url="http://native.local/v1",
        )
    )

    assert status == 200
    assert captured == {
        "url": "http://native.local/v1/chat/completions",
        "data": {
            "model": "diffusiongemma-26b-a4b-q4",
            "messages": [{"role": "user", "content": "hello"}],
        },
        "timeout": 300,
    }
    assert payload["id"] == "resp_chatcmpl-test"
    assert payload["object"] == "response"
    assert payload["model"] == "diffusiongemma-26b-a4b-q4"
    assert payload["output_text"] == "hello from native"


def test_responses_endpoint_for_native_text_accepts_prompt_fallback(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return 200

        def read(self):
            return b'{"id":"chatcmpl-test","choices":[{"message":{"content":"hello"}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(gateway.urllib.request, "urlopen", fake_urlopen)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/responses",
            {
                "model": "diffusiongemma-26b-a4b-q4",
                "prompt": "hello through prompt fallback",
                "max_output_tokens": 12,
            },
            native_base_url="http://native.local/v1",
        )
    )

    assert status == 200
    assert captured["url"] == "http://native.local/v1/chat/completions"
    assert captured["data"] == {
        "model": "diffusiongemma-26b-a4b-q4",
        "messages": [{"role": "user", "content": "hello through prompt fallback"}],
        "max_tokens": 12,
    }
    assert payload["object"] == "response"
    assert payload["output_text"] == "hello"


def test_packaged_bridge_reports_missing_dependencies_for_known_engine(capsys):
    request = {
        "schema_version": "utopic-bridge/v1",
        "model": "qwen-image",
        "engine": "diffusers",
        "modality": "image",
        "input": {"prompt": "test"},
        "output_dir": "/tmp/utopic-test-output",
        "progress_path": "/tmp/utopic-test-progress.jsonl",
    }

    assert bridge.main(["diffusers"], stdin=json.dumps(request)) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["error"]["code"] == "bridge_dependency_missing"
    assert payload["error"]["engine"] == "diffusers"
    assert "pip install" in payload["error"]["install_hint"]
    assert payload["metadata"]["schema_version"] == "utopic-bridge/v1"


def test_gateway_surfaces_structured_bridge_adapter_errors(tmp_path, monkeypatch):
    script = tmp_path / "missing_bridge.py"
    script.write_text(
        """
import json
print(json.dumps({
    "error": {
        "code": "bridge_dependency_missing",
        "message": "install diffusers",
        "engine": "diffusers",
        "install_hint": "pip install diffusers torch"
    },
    "metadata": {"schema_version": "utopic-bridge/v1"}
}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "test"},
        )
    )

    assert status == 501
    assert payload["error"] | {"run_id": "<run>", "progress_url": "<progress>"} == {
        "code": "bridge_dependency_missing",
        "message": "install diffusers",
        "engine": "diffusers",
        "install_hint": "pip install diffusers torch",
        "model": "qwen-image",
        "modality": "image",
        "progress": [],
        "run_id": "<run>",
        "progress_url": "<progress>",
    }
    assert payload["error"]["run_id"].startswith("run_")
    assert payload["error"]["progress_url"] == f"/v1/utopic/runs/{payload['error']['run_id']}/events"


def test_gateway_bridge_failures_keep_progress_url_and_events(tmp_path, monkeypatch):
    script = tmp_path / "failing_bridge.py"
    script.write_text(
        """
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
progress_path = pathlib.Path(request["progress_path"])
with progress_path.open("a", encoding="utf-8") as progress:
    progress.write(json.dumps({"event": "loading", "progress": 0.1, "message": "loading model"}) + "\\n")
    progress.write(json.dumps({"event": "failed", "progress": 1.0, "message": "missing kernel"}) + "\\n")
print(json.dumps({
    "error": {
        "code": "bridge_adapter_api_mismatch",
        "message": "missing kernel",
        "engine": "diffusers",
        "install_hint": "pip install --upgrade diffusers torch"
    },
    "metadata": {"schema_version": "utopic-bridge/v1"}
}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", f"{sys.executable} {script}")

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "a glass teapot"},
        )
    )

    assert status == 502
    assert payload["error"]["code"] == "bridge_adapter_api_mismatch"
    assert payload["error"]["run_id"].startswith("run_")
    assert payload["error"]["progress_url"] == f"/v1/utopic/runs/{payload['error']['run_id']}/events"
    assert payload["error"]["progress"] == [
        {"event": "loading", "progress": 0.1, "message": "loading model"},
        {"event": "failed", "progress": 1.0, "message": "missing kernel"},
    ]

    status, progress_payload = decode(
        gateway.handle_openai_request("GET", payload["error"]["progress_url"], None)
    )

    assert status == 200
    assert progress_payload == {"object": "list", "data": payload["error"]["progress"]}

    status, mcp_payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "utopic_generate_image",
                    "arguments": {"model": "qwen-image", "prompt": "a glass teapot"},
                },
            }
        )
    )

    assert status == 200
    assert mcp_payload["result"]["isError"] is True
    mcp_error = json.loads(mcp_payload["result"]["content"][0]["text"])["error"]
    assert mcp_error["run_id"].startswith("run_")
    assert mcp_error["progress_url"] == f"/v1/utopic/runs/{mcp_error['run_id']}/events"
    assert mcp_error["progress"] == payload["error"]["progress"]


def pathlib_path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()


def test_gateway_forwards_native_text_to_configured_openai_server(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return 200

        def read(self):
            return b'{"id":"chatcmpl-test","choices":[{"message":{"content":"hi"}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(gateway.urllib.request, "urlopen", fake_urlopen)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "diffusiongemma-26b-a4b-q4",
                "messages": [{"role": "user", "content": "hi"}],
            },
            native_base_url="http://native.local/v1",
        )
    )

    assert status == 200
    assert payload["id"] == "chatcmpl-test"
    assert captured == {
        "url": "http://native.local/v1/chat/completions",
        "data": {
            "model": "diffusiongemma-26b-a4b-q4",
            "messages": [{"role": "user", "content": "hi"}],
        },
        "timeout": 300,
    }


def test_gateway_mcp_chat_prompt_normalizes_to_native_openai_messages(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return 200

        def read(self):
            return b'{"id":"chatcmpl-test","choices":[{"message":{"content":"hi from native"}}]}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(gateway.urllib.request, "urlopen", fake_urlopen)

    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "utopic_chat",
                    "arguments": {
                        "model": "diffusiongemma-26b-a4b-q4",
                        "prompt": "hello from mcp",
                        "max_tokens": 16,
                    },
                },
            },
            native_base_url="http://native.local/v1",
        )
    )

    assert status == 200
    assert payload["result"]["isError"] is False
    result = json.loads(payload["result"]["content"][0]["text"])
    assert result["id"] == "chatcmpl-test"
    assert captured == {
        "url": "http://native.local/v1/chat/completions",
        "data": {
            "model": "diffusiongemma-26b-a4b-q4",
            "messages": [{"role": "user", "content": "hello from mcp"}],
            "max_tokens": 16,
        },
        "timeout": 300,
    }


def test_gateway_mcp_lists_and_dispatches_multimodal_tools():
    status, payload = decode(
        gateway.handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    )

    assert status == 200
    tool_names = {tool["name"] for tool in payload["result"]["tools"]}
    assert {
        "utopic_chat",
        "utopic_generate_image",
        "utopic_speak",
        "utopic_generate_music",
        "utopic_generate_video",
        "utopic_models_list",
        "utopic_models_check",
        "utopic_models_pull",
    } <= tool_names

    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "utopic_generate_image",
                    "arguments": {
                        "model": "qwen-image",
                        "prompt": "a precise product photo of a glass teapot",
                    },
                },
            }
        )
    )

    assert status == 200
    assert payload["result"]["isError"] is True
    assert payload["result"]["content"][0]["type"] == "text"
    assert "bridge_dependency_missing" in payload["result"]["content"][0]["text"]


def test_gateway_mcp_checks_model_readiness(monkeypatch):
    monkeypatch.setattr(
        gateway.models,
        "_all_model_checks",
        lambda: {
            "object": "utopic.model_check.list",
            "ready": False,
            "summary": {"ready": 1, "not_ready": 1, "total": 2},
            "data": [{"id": "diffusiongemma-26b-a4b-q4", "ready": True}],
        },
    )

    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "tools/call",
                "params": {
                    "name": "utopic_models_check",
                    "arguments": {"all": True},
                },
            }
        )
    )

    assert status == 200
    assert payload["result"]["isError"] is True
    result = json.loads(payload["result"]["content"][0]["text"])
    assert result["object"] == "utopic.model_check.list"
    assert result["summary"] == {"ready": 1, "not_ready": 1, "total": 2}

    monkeypatch.setattr(
        gateway.models,
        "model_check",
        lambda model_id: {
            "id": model_id,
            "object": "utopic.model_check",
            "ready": True,
        },
    )

    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 43,
                "method": "tools/call",
                "params": {
                    "name": "utopic_models_check",
                    "arguments": {"model": "qwen-image"},
                },
            }
        )
    )

    assert status == 200
    assert payload["result"]["isError"] is False
    result = json.loads(payload["result"]["content"][0]["text"])
    assert result == {"id": "qwen-image", "object": "utopic.model_check", "ready": True}


def test_gateway_mcp_pulls_all_models(monkeypatch):
    calls = []

    def fake_pull_all(*, force=False):
        calls.append(force)
        return {
            "object": "utopic.model_pull.list",
            "data": [
                {
                    "id": "diffusiongemma-26b-a4b-q4",
                    "path": "/models/diffusiongemma.gguf",
                    "runtime": "native",
                    "modality": "text",
                },
                {
                    "id": "qwen-image",
                    "path": "/models/qwen-image",
                    "runtime": "bridge",
                    "modality": "image",
                },
            ],
        }

    monkeypatch.setattr(gateway.models, "_pull_all_models", fake_pull_all)

    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 44,
                "method": "tools/call",
                "params": {
                    "name": "utopic_models_pull",
                    "arguments": {"all": True, "force": True},
                },
            }
        )
    )

    assert status == 200
    assert payload["result"]["isError"] is False
    assert calls == [True]
    result = json.loads(payload["result"]["content"][0]["text"])
    assert result["object"] == "utopic.model_pull.list"
    assert [item["id"] for item in result["data"]] == [
        "diffusiongemma-26b-a4b-q4",
        "qwen-image",
    ]


def test_gateway_mcp_rejects_pull_all_with_model_argument():
    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 45,
                "method": "tools/call",
                "params": {
                    "name": "utopic_models_pull",
                    "arguments": {"all": True, "model": "qwen-image"},
                },
            }
        )
    )

    assert status == 200
    assert payload["result"]["isError"] is True
    assert "pull accepts either a model alias or all=true, not both" in payload["result"]["content"][0]["text"]


def test_gateway_mcp_initialize_and_ping():
    status, payload = decode(
        gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "clientInfo": {"name": "test-client", "version": "0.0.1"},
                },
            }
        )
    )

    assert status == 200
    assert payload == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "utopic-runtime", "version": __version__},
        },
    }

    status, payload = decode(
        gateway.handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    )

    assert status == 200
    assert payload == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_gateway_mcp_dispatches_bridge_tools_through_same_runtime(tmp_path, monkeypatch):
    script = tmp_path / "fake_bridge.py"
    captured_path = tmp_path / "captured.jsonl"
    script.write_text(
        f"""
import json
import pathlib
import sys

request = json.loads(sys.stdin.read())
with pathlib.Path({str(captured_path)!r}).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(request) + "\\n")
out_dir = pathlib.Path(request["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / ("speech.wav" if request["modality"] == "tts" else "music.wav")
artifact.write_bytes(b"wav")
print(json.dumps({{
    "artifacts": [{{"type": "audio/wav", "path": str(artifact), "metadata": {{"tool": request["engine"]}}}}],
    "metadata": {{"schema_version": "utopic-bridge/v1", "engine": request["engine"]}}
}}))
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UTOPIC_BRIDGE_KOKORO_COMMAND", f"{sys.executable} {script}")
    monkeypatch.setenv("UTOPIC_BRIDGE_ACE_STEP_COMMAND", f"{sys.executable} {script}")

    for request_id, name, arguments in [
        (
            10,
            "utopic_speak",
            {"model": "kokoro-82m", "input": "hello from mcp", "voice": "af_heart"},
        ),
        (
            11,
            "utopic_generate_music",
            {"model": "ace-step-3.5b", "prompt": "ambient piano from mcp"},
        ),
    ]:
        status, payload = decode(
            gateway.handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
            )
        )

        assert status == 200
        assert payload["id"] == request_id
        assert payload["result"]["isError"] is False
        tool_payload = json.loads(payload["result"]["content"][0]["text"])
        assert tool_payload["object"] == "utopic.artifact.response"
        assert tool_payload["artifacts"][0]["type"] == "audio/wav"

    captured = [
        json.loads(line)
        for line in captured_path.read_text(encoding="utf-8").splitlines()
    ]
    assert captured[0]["endpoint"] == "/v1/audio/speech"
    assert captured[0]["input"] == {"input": "hello from mcp"}
    assert captured[0]["parameters"]["voice"] == "af_heart"
    assert captured[1]["endpoint"] == "/v1/audio/generations"
    assert captured[1]["input"] == {"prompt": "ambient piano from mcp"}


def test_gateway_cli_help_and_version(capsys):
    assert gateway.main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"utopic-runtime {__version__}\n"

    assert gateway.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "usage: utopic-runtime" in captured.out
    assert "--host HOST" in captured.out
    assert "--port PORT" in captured.out
    assert "--native-base-url URL" in captured.out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--host="], "expected a value after --host"),
        (["--host", "--port", "8911"], "expected a value after --host"),
        (["--port="], "expected a value after --port"),
        (["--port", "--native-base-url", "http://127.0.0.1:8910"], "expected a value after --port"),
        (["--native-base-url="], "expected a value after --native-base-url"),
        (["--native-base-url", "--host", "127.0.0.1"], "expected a value after --native-base-url"),
    ],
)
def test_gateway_cli_rejects_missing_option_values(args, message, capsys):
    assert gateway.main(args) == 1
    captured = capsys.readouterr()
    assert f"utopic-runtime: {message}" in captured.err


def test_top_level_cli_routes_gateway_command(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.gateway, "main", lambda args: calls.append(args) or 0)

    assert cli.main(["gateway", "--port", "9123"]) == 0
    assert calls == [["--port", "9123"]]


def test_gateway_cli_handles_keyboard_interrupt_without_traceback(monkeypatch, capsys):
    def interrupt(_host, _port, native_base_url=None):
        assert native_base_url is None
        raise KeyboardInterrupt

    monkeypatch.setattr(gateway, "serve", interrupt)

    assert gateway.main(["--port", "9123"]) == 130
    captured = capsys.readouterr()
    assert "KeyboardInterrupt" not in captured.err


def test_gateway_cli_reports_bind_errors_without_traceback(monkeypatch, capsys):
    def fail_bind(_host, _port, native_base_url=None):
        assert native_base_url == "http://127.0.0.1:8910"
        raise PermissionError("bind denied")

    monkeypatch.setattr(gateway, "serve", fail_bind)

    assert (
        gateway.main(
            [
                "--port",
                "9123",
                "--native-base-url",
                "http://127.0.0.1:8910",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "utopic-runtime: failed to start server: bind denied" in captured.err
    assert "Traceback" not in captured.err
