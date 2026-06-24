import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utopic import __version__, bridge, cli, gateway, mcp

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


def test_gateway_models_endpoint_exposes_multimodal_runtime_metadata(monkeypatch):
    monkeypatch.setenv("UTOPIC_EXPERIMENTAL_BRIDGE", "1")

    status, payload = decode(gateway.handle_openai_request("GET", "/v1/models", None))

    assert status == 200
    by_id = {item["id"]: item for item in payload["data"]}
    assert by_id["diffusiongemma-26b-a4b-q4"]["modality"] == "text"
    assert by_id["diffusiongemma-26b-a4b-q4"]["runtime"] == "native"
    assert by_id["qwen-image"]["modality"] == "image"
    assert by_id["qwen-image"]["runtime"] == "planned_native"
    assert by_id["qwen-image"]["repo"] == "Qwen/Qwen-Image"
    assert by_id["qwen-image"]["url"] == "https://huggingface.co/Qwen/Qwen-Image"
    assert "/v1/images/generations" in by_id["qwen-image"]["endpoints"]
    assert "experimental_bridge" not in by_id["qwen-image"]
    assert by_id["krea-2-raw"]["modality"] == "image"
    assert by_id["krea-2-raw"]["engine"] == "diffusers"
    assert by_id["krea-2-raw"]["repo"] == "krea/Krea-2-Raw"
    assert by_id["cosmos3-super"]["modality"] == "image"
    assert by_id["cosmos3-super"]["engine"] == "cosmos"
    assert by_id["cosmos3-super"]["repo"] == "nvidia/Cosmos3-Super-Text2Image"
    assert by_id["cosmos3-super"]["requirements"]["min_gpu_memory_gib"] == 96
    assert by_id["cosmos3-super"]["requirements"]["allow_cpu"] is False
    assert by_id["zuna"]["modality"] == "misc"
    assert by_id["zuna"]["engine"] == "artifact"
    assert by_id["zuna"]["runtime"] == "planned_native"
    assert by_id["zuna"]["repo"] == "Zyphra/ZUNA"
    assert "/v1/utopic/misc/generations" in by_id["zuna"]["endpoints"]
    assert "experimental_bridge" not in by_id["zuna"]


def test_gateway_models_endpoint_keeps_planned_models_native_only(monkeypatch):
    monkeypatch.setenv("UTOPIC_EXPERIMENTAL_BRIDGE", "1")

    status, payload = decode(gateway.handle_openai_request("GET", "/v1/models", None))

    assert status == 200
    planned_models = [
        item for item in payload["data"] if item["runtime"] == "planned_native"
    ]

    assert {item["id"] for item in planned_models} >= {
        "qwen-image",
        "flux-1-schnell",
        "krea-2-raw",
        "cosmos3-super",
        "kokoro-82m",
        "chatterbox",
        "dia-1.6b",
        "ace-step-3.5b",
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b",
        "ltx-video",
        "zuna",
    }
    for item in planned_models:
        assert item["runtime"] == "planned_native"
        assert item["native_status"] == "planned"
        assert item["runner"] == "utopic-runner"
        assert "experimental_bridge" not in item

    by_id = {item["id"]: item for item in planned_models}
    assert by_id["ltx-video"]["repo"] == "Lightricks/LTX-Video"
    assert by_id["ltx-video"]["runner"] == "utopic-runner"
    assert by_id["cosmos3-super"]["runner"] == "utopic-runner"
    assert by_id["cosmos3-super"]["requirements"]["min_gpu_memory_gib"] == 96


def test_gateway_cosmos_returns_oom_preflight(monkeypatch):
    monkeypatch.setattr(
        gateway,
        "_detect_runtime_capacity",
        lambda: {
            "backend": "metal",
            "device": "Apple M4 Pro",
            "gpu_memory_gib": 40.0,
        },
        raising=False,
    )

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "cosmos3-super", "prompt": "a glass city at sunrise"},
        )
    )

    assert status == 507
    assert payload["error"]["code"] == "oom"
    assert payload["error"]["model"] == "cosmos3-super"
    assert payload["error"]["required_gpu_memory_gib"] == 96
    assert payload["error"]["detected"]["device"] == "Apple M4 Pro"
    assert "requires at least 96 GiB GPU memory" in payload["error"]["message"]


def test_gateway_image_generation_reports_native_runner_not_ready_by_default():
    request = {
        "model": "qwen-image",
        "prompt": "a precise product photo of a glass teapot",
        "size": "1024x1024",
    }

    status, payload = decode(
        gateway.handle_openai_request("POST", "/v1/images/generations", request)
    )

    assert status == 503
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["model"] == "qwen-image"
    assert payload["error"]["modality"] == "image"
    assert payload["error"]["engine"] == "diffusers"
    assert payload["error"]["native_status"] == "planned"
    assert "behind utopic-runner" in payload["error"]["message"]


def test_gateway_native_artifact_model_routes_to_native_runner(monkeypatch):
    entry = gateway.models.ModelEntry(
        id="unit-native-image",
        name="Unit Native Image",
        family="unit",
        filename="unit-native-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="native",
        native_status="ready",
        runner="utopic-runner",
        endpoints=("/v1/images/generations",),
        outputs=("image/png",),
    )
    captured = {}
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)

    def fake_generation(runner_entry, endpoint, request):
        captured["entry"] = runner_entry
        captured["endpoint"] = endpoint
        captured["request"] = request
        return {
            "ok": True,
            "type": "image",
            "backend": "metal",
            "device": "Apple M4 Pro",
            "artifacts": [{"type": "image/png", "url": "file:///tmp/unit-native-image.png"}],
            "metrics": {"total_ms": 12.5},
        }

    monkeypatch.setattr(gateway.native_runner, "generation", fake_generation)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": entry.id, "prompt": "a native image"},
        )
    )

    assert status == 200
    assert payload["model"] == entry.id
    assert payload["metadata"]["runtime"] == "native-runner"
    assert payload["metadata"]["runner"] == "utopic-runner"
    assert payload["metadata"]["device"] == "Apple M4 Pro"
    assert payload["data"] == [{"url": "file:///tmp/unit-native-image.png"}]
    assert captured["entry"] is entry
    assert captured["endpoint"] == "/v1/images/generations"
    assert captured["request"]["prompt"] == "a native image"


def test_gateway_does_not_use_packaged_bridge_command_by_default_for_bridge_models(monkeypatch):
    monkeypatch.delenv("UTOPIC_BRIDGE_DIFFUSERS_COMMAND", raising=False)
    monkeypatch.delenv("UTOPIC_BRIDGE_COMMAND", raising=False)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "a precise product photo of a glass teapot"},
        )
    )

    assert status == 503
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["engine"] == "diffusers"
    assert payload["error"]["model"] == "qwen-image"
    assert payload["error"]["modality"] == "image"
    assert payload["error"]["native_status"] == "planned"


def test_gateway_exposes_openai_routes_for_each_bridge_modality():
    cases = [
        (
            "/v1/images/generations",
            {"model": "qwen-image", "prompt": "a teapot"},
            "image",
            "diffusers",
        ),
        (
            "/v1/audio/speech",
            {"model": "kokoro-82m", "input": "hello"},
            "tts",
            "kokoro",
        ),
        (
            "/v1/audio/speech",
            {"model": "chatterbox", "input": "hello"},
            "tts",
            "chatterbox",
        ),
        (
            "/v1/audio/generations",
            {"model": "ace-step-3.5b", "prompt": "ambient piano"},
            "music",
            "ace-step",
        ),
        (
            "/v1/videos/generations",
            {"model": "wan2.1-t2v-1.3b", "prompt": "waves"},
            "video",
            "wan",
        ),
        (
            "/v1/responses",
            {"model": "flux-1-schnell", "prompt": "a red cube"},
            "image",
            "diffusers",
        ),
    ]

    for endpoint, request, modality, engine in cases:
        status, payload = decode(gateway.handle_openai_request("POST", endpoint, request))

        assert status == 503
        assert payload["error"]["code"] == "unsupported_model"
        assert payload["error"]["modality"] == modality
        assert payload["error"]["engine"] == engine
        assert payload["error"]["native_status"] == "planned"


def test_every_planned_catalog_model_has_openai_and_mcp_runtime_surface():
    generation_tool_by_modality = {
        "image": "utopic_generate_image",
        "tts": "utopic_generate_speech",
        "music": "utopic_generate_music",
        "video": "utopic_generate_video",
        "misc": "utopic_generate_misc",
    }
    request_by_modality = {
        "image": {"prompt": "a small ceramic cup"},
        "tts": {"input": "hello from utopic"},
        "music": {"prompt": "quiet piano"},
        "video": {"prompt": "waves rolling onto a beach"},
        "misc": {"artifact": "/tmp/input.bin"},
    }

    planned_models = [
        entry for entry in gateway.models.list_models() if entry.runtime == "planned_native"
    ]
    assert planned_models

    for entry in planned_models:
        assert entry.modality in generation_tool_by_modality
        assert "/v1/responses" in entry.endpoints
        assert any(endpoint != "/v1/responses" for endpoint in entry.endpoints)

        modality_endpoint = next(endpoint for endpoint in entry.endpoints if endpoint != "/v1/responses")
        request = {"model": entry.id, **request_by_modality[entry.modality]}
        status, payload = decode(gateway.handle_openai_request("POST", modality_endpoint, request))

        assert status in {503, 507}, entry.id
        assert payload["error"]["model"] == entry.id
        assert payload["error"]["modality"] == entry.modality
        assert payload["error"]["engine"] == entry.engine
        assert payload["error"]["code"] in {"unsupported_model", "oom"}

        responses_request = {
            "model": entry.id,
            "input": request_by_modality[entry.modality].get("prompt")
            or request_by_modality[entry.modality].get("input")
            or request_by_modality[entry.modality]["artifact"],
        }
        status, payload = decode(gateway.handle_openai_request("POST", "/v1/responses", responses_request))

        assert status in {503, 507}, entry.id
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



def test_gateway_native_artifact_model_supports_b64_json_response_format(monkeypatch, tmp_path):
    artifact = tmp_path / "native-image.png"
    artifact.write_bytes(b"png")
    entry = gateway.models.ModelEntry(
        id="unit-native-image-b64",
        name="Unit Native Image B64",
        family="unit",
        filename="unit-native-image-b64",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="native",
        native_status="ready",
        runner="utopic-runner",
        endpoints=("/v1/images/generations", "/v1/responses"),
        outputs=("image/png",),
    )
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)

    def fake_generation(runner_entry, endpoint, request):
        return {
            "ok": True,
            "type": "image",
            "backend": "cpu",
            "artifacts": [{"type": "image/png", "path": str(artifact)}],
            "metrics": {},
        }

    monkeypatch.setattr(gateway.native_runner, "generation", fake_generation)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/images/generations",
            {
                "model": entry.id,
                "prompt": "a precise product photo of a glass teapot",
                "response_format": "b64_json",
            },
        )
    )

    assert status == 200
    assert payload["metadata"]["runtime"] == "native-runner"
    assert payload["data"] == [{"b64_json": base64.b64encode(b"png").decode("ascii")}]


def test_responses_endpoint_normalizes_structured_input_for_native_runner(monkeypatch):
    entry = gateway.models.ModelEntry(
        id="unit-native-image-responses",
        name="Unit Native Image Responses",
        family="unit",
        filename="unit-native-image-responses",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="native-image",
        runtime="native",
        native_status="ready",
        runner="utopic-runner",
        endpoints=("/v1/images/generations", "/v1/responses"),
        outputs=("image/png",),
    )
    captured = {}
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)

    def fake_generation(runner_entry, endpoint, request):
        captured["entry"] = runner_entry
        captured["endpoint"] = endpoint
        captured["request"] = request
        return {
            "ok": True,
            "type": "image",
            "backend": "cpu",
            "artifacts": [{"type": "image/png", "url": "file:///tmp/native-response.png"}],
            "metrics": {},
        }

    monkeypatch.setattr(gateway.native_runner, "generation", fake_generation)

    status, payload = decode(
        gateway.handle_openai_request(
            "POST",
            "/v1/responses",
            {
                "model": entry.id,
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
    assert captured["entry"] is entry
    assert captured["endpoint"] == "/v1/responses"
    assert captured["request"]["prompt"] == "a red cube on a steel table"
    assert captured["request"]["size"] == "512x512"
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["type"] == "output_image"

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


def test_packaged_bridge_reports_retired_for_known_engine(capsys, monkeypatch):
    monkeypatch.delenv("UTOPIC_EXPERIMENTAL_BRIDGE", raising=False)
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

    assert payload["error"]["code"] == "native_runner_required"
    assert payload["error"]["engine"] == "diffusers"
    assert payload["error"]["install_hint"] == "utopic setup"
    assert "native runner" in payload["error"]["message"]
    assert payload["metadata"]["schema_version"] == "utopic-bridge/v1"


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
        "utopic_generate_speech",
        "utopic_speak",
        "utopic_generate_music",
        "utopic_generate_video",
        "utopic_generate_misc",
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
    error = json.loads(payload["result"]["content"][0]["text"])["error"]
    assert error["code"] == "unsupported_model"
    assert error["native_status"] == "planned"


def test_gateway_mcp_tool_definitions_are_clear_for_agents():
    status, payload = decode(
        gateway.handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    )

    assert status == 200
    by_name = {tool["name"]: tool for tool in payload["result"]["tools"]}
    all_descriptions = "\n".join(tool["description"] for tool in by_name.values()).lower()

    for required_phrase in [
        "local",
        "offline",
        "private",
        "image",
        "speech",
        "music",
        "video",
        "readiness",
        "hardware",
        "outputs",
    ]:
        assert required_phrase in all_descriptions

    chat = by_name["utopic_chat"]
    assert "OpenAI-compatible".lower() in chat["description"].lower()
    assert "diffusiongemma-26b-a4b-q4" in chat["inputSchema"]["properties"]["model"]["description"]
    assert "max_tokens" in chat["inputSchema"]["properties"]

    video = by_name["utopic_generate_video"]
    assert "gb10" in video["description"].lower()
    assert "utopic_models_check" in video["description"]

    models_check = by_name["utopic_models_check"]
    assert "missing-dependency" in models_check["description"]
    assert "OOM" in models_check["description"]


def test_native_stdio_mcp_schema_points_agents_to_runtime_mcp_for_multimodal_tools():
    source = (REPO_ROOT / "python" / "utopic" / "core" / "native" / "mcp_server.cpp").read_text(
        encoding="utf-8"
    )

    assert "local/offline Utopic diffusion GGUF model" in source
    assert "utopic-runtime /mcp endpoint" in source
    assert "Maximum completion tokens" in source


def test_runtime_stdio_mcp_lists_full_gateway_tool_catalog():
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    assert mcp._runtime_stdio(stdin, stdout, native_base_url=None) == 0
    response = json.loads(stdout.getvalue())

    assert response["id"] == 1
    tool_names = {tool["name"] for tool in response["result"]["tools"]}
    assert {
        "utopic_chat",
        "utopic_generate_image",
        "utopic_generate_speech",
        "utopic_speak",
        "utopic_generate_music",
        "utopic_generate_video",
        "utopic_generate_misc",
        "utopic_models_check",
    } <= tool_names


def test_runtime_stdio_mcp_reports_packaged_model_readiness():
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "utopic_models_check",
            "arguments": {"model": "diffusiongemma-26b-a4b-q4"},
        },
    }
    stdin = io.StringIO(json.dumps(request) + "\n")
    stdout = io.StringIO()

    assert mcp._runtime_stdio(stdin, stdout, native_base_url=None) == 0
    response = json.loads(stdout.getvalue())
    payload = json.loads(response["result"]["content"][0]["text"])

    assert response["result"]["isError"] is False
    assert payload["id"] == "diffusiongemma-26b-a4b-q4"
    assert payload["status"] in {"missing_model_file", "size_mismatch"}
    assert "utopic models pull diffusiongemma-26b-a4b-q4" in payload["next_steps"]


def test_runtime_stdio_mcp_suppresses_notification_responses():
    requests = [
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
    ]
    stdin = io.StringIO("".join(json.dumps(request) + "\n" for request in requests))
    stdout = io.StringIO()

    assert mcp._runtime_stdio(stdin, stdout, native_base_url=None) == 0
    lines = [line for line in stdout.getvalue().splitlines() if line]

    assert len(lines) == 1
    response = json.loads(lines[0])
    assert response["id"] == 3
    assert response["result"] == {}


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
    assert payload["result"]["isError"] is False
    result = json.loads(payload["result"]["content"][0]["text"])
    assert payload["result"]["structuredContent"] == result
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
    assert payload["result"]["structuredContent"] == result
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
                    "runtime": "planned_native",
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


def test_gateway_mcp_dispatches_planned_tools_to_native_readiness_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("UTOPIC_EXPERIMENTAL_BRIDGE", "1")
    monkeypatch.setenv("UTOPIC_BRIDGE_KOKORO_COMMAND", "python -m should_not_run")
    monkeypatch.setenv("UTOPIC_BRIDGE_ACE_STEP_COMMAND", "python -m should_not_run")
    monkeypatch.setenv("UTOPIC_BRIDGE_ARTIFACT_COMMAND", "python -m should_not_run")
    misc_source = tmp_path / "source.bin"
    misc_source.write_bytes(b"misc")

    for request_id, name, arguments in [
        (
            10,
            "utopic_generate_speech",
            {"model": "kokoro-82m", "input": "hello from mcp", "voice": "af_heart"},
        ),
        (
            11,
            "utopic_generate_music",
            {"model": "ace-step-3.5b", "prompt": "ambient piano from mcp"},
        ),
        (
            12,
            "utopic_generate_misc",
            {"model": "zuna", "artifact": str(misc_source)},
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
        assert payload["result"]["isError"] is True
        tool_payload = json.loads(payload["result"]["content"][0]["text"])
        assert payload["result"]["structuredContent"] == tool_payload
        assert tool_payload["error"]["code"] == "unsupported_model"
        assert tool_payload["error"]["model"] == arguments["model"]
        assert tool_payload["error"]["modality"] in {"tts", "music", "misc"}
        assert tool_payload["error"]["native_status"] == "planned"


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
