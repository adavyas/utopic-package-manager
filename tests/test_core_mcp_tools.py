import io
import json
from pathlib import Path

from utopic import gateway, mcp


def test_gateway_mcp_tool_definitions_are_agent_friendly():
    by_name = {tool["name"]: tool for tool in gateway.MCP_TOOLS}

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
    } <= set(by_name)

    all_descriptions = "\n".join(tool["description"] for tool in gateway.MCP_TOOLS).lower()
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

    chat_schema = by_name["utopic_chat"]["inputSchema"]["properties"]
    assert "diffusiongemma-26b-a4b-q4" in chat_schema["model"]["description"]
    assert "OpenAI-compatible".lower() in by_name["utopic_chat"]["description"].lower()

    video_description = by_name["utopic_generate_video"]["description"].lower()
    assert "gb10" in video_description
    assert "utopic_models_check" in video_description


def test_gateway_mcp_planned_modalities_are_described_as_native_readiness_surfaces():
    by_name = {tool["name"]: tool for tool in gateway.MCP_TOOLS}
    planned_tool_names = {
        "utopic_generate_image",
        "utopic_generate_speech",
        "utopic_generate_music",
        "utopic_generate_video",
        "utopic_generate_misc",
    }

    for name in planned_tool_names:
        description = by_name[name]["description"].lower()
        assert "native runner" in description
        assert "planned" in description
        assert "readiness" in description
        assert "experimental bridge" in description
        assert "generate local" not in description
        assert "local image generation" not in description
        assert "local music" not in description
        assert "local video" not in description


def test_native_stdio_mcp_schema_points_agents_to_runtime_mcp_for_multimodal_tools():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root / "python" / "utopic" / "core" / "native" / "mcp_server.cpp"
    ).read_text(encoding="utf-8")

    assert "local/offline Utopic diffusion GGUF model" in source
    assert "utopic-runtime /mcp endpoint" in source
    assert "Maximum completion tokens" in source


def test_runtime_stdio_mcp_lists_all_gateway_tools():
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
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert {
        "utopic_chat",
        "utopic_generate_image",
        "utopic_generate_speech",
        "utopic_speak",
        "utopic_generate_music",
        "utopic_generate_video",
        "utopic_models_check",
    } <= names


def test_runtime_stdio_mcp_model_check_reports_missing_model():
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
    assert response["result"]["isError"] is True
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["id"] == "diffusiongemma-26b-a4b-q4"
    assert payload["ready"] is False
    assert payload["status"] == "missing_model_file"


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


def test_gateway_native_text_falls_back_to_runner_without_server(monkeypatch, tmp_path):
    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake", encoding="utf-8")
    entry = gateway.models.ModelEntry(
        id="unit-text",
        name="Unit Text",
        family="unit",
        filename="model.gguf",
        url="https://example.invalid/model.gguf",
        size="1 MiB",
        recommended=True,
        description="unit",
    )
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)
    monkeypatch.setattr(gateway.models, "list_models", lambda: [entry])
    monkeypatch.setattr(type(entry), "path", property(lambda self: model_path))
    captured = {}

    def fake_chat_completion(runner_entry, request):
        captured["entry"] = runner_entry
        captured["request"] = request
        return {
            "ok": True,
            "type": "text",
            "text": "hello from runner",
            "artifacts": [],
            "backend": "metal",
            "metrics": {"prompt_tokens": 3, "answer_tokens": 4},
        }

    monkeypatch.setattr(gateway.native_runner, "chat_completion", fake_chat_completion)

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/chat/completions",
        {
            "model": entry.id,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
        native_base_url=None,
    )

    payload = json.loads(body)
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "hello from runner"
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert payload["metadata"]["runtime"] == "native-runner"
    assert captured["entry"] is entry
    assert captured["request"]["messages"][0]["content"] == "hi"


def test_gateway_raw_gguf_model_uses_runner(monkeypatch, tmp_path):
    model_path = tmp_path / "raw-model.gguf"
    model_path.write_text("model", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: None)

    def fake_chat_completion(runner_entry, request):
        captured["entry"] = runner_entry
        captured["request"] = request
        return {"ok": True, "type": "text", "text": "raw path works", "metrics": {}}

    monkeypatch.setattr(gateway.native_runner, "chat_completion", fake_chat_completion)

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/chat/completions",
        {"model": str(model_path), "messages": [{"role": "user", "content": "hi"}]},
        native_base_url=None,
    )

    payload = json.loads(body)
    assert status == 200
    assert payload["model"] == str(model_path)
    assert payload["choices"][0]["message"]["content"] == "raw path works"
    assert captured["entry"].id == str(model_path)
    assert captured["entry"].path == model_path
    assert captured["request"]["messages"][0]["content"] == "hi"


def test_gateway_active_text_model_alias_uses_runner(monkeypatch, tmp_path):
    model_path = tmp_path / "active.gguf"
    model_path.write_text("model", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: None)

    def fake_chat_completion(runner_entry, request):
        captured["entry"] = runner_entry
        captured["request"] = request
        return {"ok": True, "type": "text", "text": "active model works", "metrics": {}}

    monkeypatch.setattr(gateway.native_runner, "chat_completion", fake_chat_completion)

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/chat/completions",
        {"model": "utopic", "messages": [{"role": "user", "content": "hi"}]},
        native_base_url=None,
        active_text_model_path=model_path,
    )

    payload = json.loads(body)
    assert status == 200
    assert payload["model"] == "utopic"
    assert payload["choices"][0]["message"]["content"] == "active model works"
    assert captured["entry"].id == "utopic"
    assert captured["entry"].path == model_path


def test_gateway_mcp_chat_active_text_model_alias_uses_runner(monkeypatch, tmp_path):
    model_path = tmp_path / "active-mcp.gguf"
    model_path.write_text("model", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: None)

    def fake_chat_completion(runner_entry, request):
        captured["entry"] = runner_entry
        captured["request"] = request
        return {"ok": True, "type": "text", "text": "active mcp model works", "metrics": {}}

    monkeypatch.setattr(gateway.native_runner, "chat_completion", fake_chat_completion)

    status, _headers, body = gateway.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "utopic_chat",
                "arguments": {"model": "utopic", "prompt": "hi"},
            },
        },
        native_base_url=None,
        active_text_model_path=model_path,
    )

    payload = json.loads(body)
    content = json.loads(payload["result"]["content"][0]["text"])
    assert status == 200
    assert payload["id"] == 8
    assert payload["result"]["isError"] is False
    assert content["choices"][0]["message"]["content"] == "active mcp model works"
    assert captured["entry"].id == "utopic"
    assert captured["entry"].path == model_path
    assert captured["request"]["messages"][0]["content"] == "hi"


def test_gateway_native_base_url_still_takes_priority_over_runner(monkeypatch, tmp_path):
    entry = gateway.models.ModelEntry(
        id="unit-text",
        name="Unit Text",
        family="unit",
        filename="model.gguf",
        url="https://example.invalid/model.gguf",
        size="1 MiB",
        recommended=True,
        description="unit",
    )
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)
    monkeypatch.setattr(gateway.native_runner, "chat_completion", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runner should not be used")))
    monkeypatch.setattr(
        gateway,
        "_forward_native_text",
        lambda base_url, path, request: (
            200,
            {"content-type": "application/json"},
            json.dumps(
                {
                    "id": "chatcmpl-unit",
                    "object": "chat.completion",
                    "created": 1,
                    "model": entry.id,
                    "choices": [{"message": {"content": "from server"}}],
                }
            ).encode("utf-8"),
        ),
    )

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/chat/completions",
        {"model": entry.id, "messages": [{"role": "user", "content": "hi"}]},
        native_base_url="http://127.0.0.1:8910",
    )

    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "from server"


def test_gateway_bridge_generation_reports_native_runner_not_ready(monkeypatch):
    entry = gateway.models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="diffusers",
        runtime="bridge",
        endpoints=("/v1/images/generations",),
        outputs=("image",),
    )
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)
    captured = {}

    def fake_generation(runner_entry, endpoint, request):
        captured["entry"] = runner_entry
        captured["endpoint"] = endpoint
        captured["request"] = request
        return {
            "ok": False,
            "error": {
                "code": "unsupported_model",
                "message": "native runner task is not implemented yet",
                "detail": {
                    "task": "image",
                    "model": runner_entry.id,
                    "native_status": runner_entry.native_status,
                },
            },
        }

    monkeypatch.setattr(gateway.native_runner, "generation", fake_generation)

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/images/generations",
        {"model": entry.id, "prompt": "a native runner test"},
    )

    payload = json.loads(body)
    assert status == 503
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_model"
    assert payload["error"]["detail"]["native_status"] == "planned"
    assert captured["entry"] is entry
    assert captured["endpoint"] == "/v1/images/generations"
    assert captured["request"]["prompt"] == "a native runner test"


def test_gateway_ignores_bridge_command_without_experimental_gate(monkeypatch):
    entry = gateway.models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="diffusers",
        runtime="bridge",
        endpoints=("/v1/images/generations",),
        outputs=("image",),
    )
    captured = {}
    monkeypatch.setenv("UTOPIC_BRIDGE_COMMAND", "python -m should_not_run")
    monkeypatch.delenv("UTOPIC_EXPERIMENTAL_BRIDGE", raising=False)
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)
    monkeypatch.setattr(gateway, "_run_bridge", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bridge should not run")))

    def fake_generation(runner_entry, endpoint, request):
        captured["entry"] = runner_entry
        captured["endpoint"] = endpoint
        return {
            "ok": False,
            "error": {
                "code": "unsupported_model",
                "message": "native runner task is not implemented yet",
                "detail": {"task": runner_entry.modality, "model": runner_entry.id},
            },
        }

    monkeypatch.setattr(gateway.native_runner, "generation", fake_generation)

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/images/generations",
        {"model": entry.id, "prompt": "native only"},
    )

    payload = json.loads(body)
    assert status == 503
    assert payload["error"]["code"] == "unsupported_model"
    assert captured["entry"] is entry
    assert captured["endpoint"] == "/v1/images/generations"


def test_gateway_allows_bridge_command_when_experimental_gate_is_enabled(monkeypatch):
    entry = gateway.models.ModelEntry(
        id="unit-image",
        name="Unit Image",
        family="unit",
        filename="unit-image",
        url="https://example.invalid/unit-image",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="image",
        engine="diffusers",
        runtime="bridge",
        endpoints=("/v1/images/generations",),
        outputs=("image",),
    )
    captured = {}
    monkeypatch.setenv("UTOPIC_BRIDGE_COMMAND", "python -m experimental_bridge")
    monkeypatch.setenv("UTOPIC_EXPERIMENTAL_BRIDGE", "1")
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)
    monkeypatch.setattr(gateway.native_runner, "generation", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runner should not run")))

    def fake_run_bridge(runner_entry, endpoint, request, command):
        captured["entry"] = runner_entry
        captured["endpoint"] = endpoint
        captured["command"] = command
        return 200, {"content-type": "application/json"}, b'{"ok": true}'

    monkeypatch.setattr(gateway, "_run_bridge", fake_run_bridge)

    status, _headers, body = gateway.handle_openai_request(
        "POST",
        "/v1/images/generations",
        {"model": entry.id, "prompt": "experimental"},
    )

    assert status == 200
    assert json.loads(body)["ok"] is True
    assert captured["entry"] is entry
    assert captured["endpoint"] == "/v1/images/generations"
    assert captured["command"] == ["python", "-m", "experimental_bridge"]


def test_gateway_mcp_generate_speech_is_canonical_tts_tool(monkeypatch):
    entry = gateway.models.ModelEntry(
        id="unit-tts",
        name="Unit TTS",
        family="unit",
        filename="unit-tts",
        url="https://example.invalid/unit-tts",
        size="1 GiB",
        recommended=False,
        description="unit",
        modality="tts",
        engine="kokoro",
        runtime="bridge",
        endpoints=("/v1/audio/speech",),
        outputs=("audio/wav",),
    )
    monkeypatch.setattr(gateway.models, "get_model", lambda model_id: entry if model_id == entry.id else None)
    captured = {}

    def fake_generation(runner_entry, endpoint, request):
        captured["entry"] = runner_entry
        captured["endpoint"] = endpoint
        captured["request"] = request
        return {
            "ok": False,
            "error": {
                "code": "unsupported_model",
                "message": "native runner task is not implemented yet",
                "detail": {"task": "tts", "model": runner_entry.id},
            },
        }

    monkeypatch.setattr(gateway.native_runner, "generation", fake_generation)

    status, _headers, body = gateway.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "utopic_generate_speech",
                "arguments": {"model": entry.id, "input": "hello", "voice": "af_heart"},
            },
        }
    )

    payload = json.loads(body)
    assert status == 200
    assert payload["id"] == 9
    assert payload["result"]["isError"] is True
    assert captured["entry"] is entry
    assert captured["endpoint"] == "/v1/audio/speech"
    assert captured["request"]["input"] == "hello"
    assert captured["request"]["voice"] == "af_heart"
