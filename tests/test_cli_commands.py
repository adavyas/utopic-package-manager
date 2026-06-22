from pathlib import Path

import pytest

from utopic import chat, cli, models


def test_chat_launch_sets_runtime_paths_and_executes_node(monkeypatch, tmp_path):
    script = tmp_path / "utopic-chat.js"
    script.write_text("console.log('chat')\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(chat, "_chat_script", lambda: script)
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(chat.installer, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat.installer, "setup", lambda argv: captured.setdefault("setup", list(argv)) or 0)
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: captured.update(command=command, env=env, check=check))

    assert chat.launch(["dream-7b-q4"]) == 0

    assert captured["setup"] == []
    assert captured["command"] == ["/usr/bin/node", str(script), "dream-7b-q4"]
    assert captured["env"]["UTOPIC_BIN_DIR"] == str(tmp_path / "bin")
    assert captured["env"]["UTOPIC_MODELS_DIR"] == str(tmp_path / "cache" / "models")
    assert captured["check"] is True


def test_chat_launch_skips_setup_when_server_binary_exists(monkeypatch, tmp_path):
    script = tmp_path / "utopic-chat.js"
    script.write_text("console.log('chat')\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")
    setup_calls = []

    monkeypatch.setattr(chat, "_chat_script", lambda: script)
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(chat.installer, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat.installer, "setup", lambda argv: setup_calls.append(list(argv)) or 0)
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: None)

    assert chat.launch(["dream-7b-q4"]) == 0

    assert setup_calls == []


def test_chat_launch_skips_setup_for_existing_server(monkeypatch, tmp_path):
    script = tmp_path / "utopic-chat.js"
    script.write_text("console.log('chat')\n", encoding="utf-8")
    setup_calls = []
    captured = {}

    monkeypatch.setattr(chat, "_chat_script", lambda: script)
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "bin_dir", lambda: tmp_path / "missing-bin")
    monkeypatch.setattr(chat.installer, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat.installer, "setup", lambda argv: setup_calls.append(list(argv)) or 0)
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: captured.update(command=command))

    assert chat.launch(["--server", "http://127.0.0.1:8910"]) == 0

    assert setup_calls == []
    assert captured["command"] == [
        "/usr/bin/node",
        str(script),
        "--server",
        "http://127.0.0.1:8910",
    ]


def test_chat_help_does_not_run_setup(monkeypatch, tmp_path):
    script = tmp_path / "utopic-chat.js"
    script.write_text("console.log('help')\n", encoding="utf-8")
    setup_calls = []

    monkeypatch.setattr(chat, "_chat_script", lambda: script)
    monkeypatch.setattr(chat.shutil, "which", lambda name: pytest.fail("help should not require node"))
    monkeypatch.setattr(chat.installer, "bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(chat.installer, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat.installer, "setup", lambda argv: setup_calls.append(list(argv)) or 0)
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: pytest.fail("help should not launch node"))

    assert chat.launch(["--help"]) == 0

    assert setup_calls == []


def test_chat_launch_reports_missing_node_before_setup(monkeypatch):
    setup_calls = []

    monkeypatch.setattr(chat.shutil, "which", lambda name: None)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: setup_calls.append(list(argv)) or 0)

    assert chat.launch(["dream-7b-q4"]) == 1
    assert setup_calls == []


def test_cli_run_with_prompt_delegates_to_native_one_shot(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: calls.append((name, list(argv))))

    cli.main(["run", "-m", "model.gguf", "-p", "hello", "-n", "8"])

    assert calls == [
        ("setup", True, "utopic"),
        ("utopic", ["-m", "model.gguf", "-p", "hello", "-n", "8"]),
    ]


def test_cli_run_without_prompt_starts_openai_server(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: Path("/models/dream.gguf"))
    monkeypatch.setattr(
        cli,
        "_run_server",
        lambda model_path, server_args, host, port: calls.append(
            ("server", model_path, list(server_args), host, port)
        )
        or 0,
    )

    assert cli.main(["run", "dream-7b-q4", "--port", "8999", "-ngl", "99"]) == 0

    assert calls == [
        ("setup", True, "utopic_server"),
        ("server", "/models/dream.gguf", ["--port", "8999", "-ngl", "99"], "127.0.0.1", "8999"),
    ]


def test_cli_run_allows_server_flags_before_positional_model(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: Path(f"/models/{value}.gguf"))
    monkeypatch.setattr(
        cli,
        "_run_server",
        lambda model_path, server_args, host, port: calls.append(
            ("server", model_path, list(server_args), host, port)
        )
        or 0,
    )

    assert cli.main(["run", "--port", "8999", "-ngl", "99", "dream-7b-q4"]) == 0

    assert calls == [
        ("setup", True, "utopic_server"),
        ("server", "/models/dream-7b-q4.gguf", ["--port", "8999", "-ngl", "99"], "127.0.0.1", "8999"),
    ]


def test_cli_run_normalizes_wildcard_host_for_client_url(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: Path("/models/dream.gguf"))
    monkeypatch.setattr(
        cli,
        "_run_server",
        lambda model_path, server_args, host, port: calls.append(
            ("server", model_path, list(server_args), host, port)
        )
        or 0,
    )

    assert cli.main(["run", "dream-7b-q4", "--host", "0.0.0.0", "--port", "8999"]) == 0
    assert calls == [
        ("server", "/models/dream.gguf", ["--host", "0.0.0.0", "--port", "8999"], "0.0.0.0", "8999")
    ]
    assert cli._server_url("0.0.0.0", "8999") == "http://127.0.0.1:8999/v1/chat/completions"
    assert cli._server_health_url("::", "8999") == "http://127.0.0.1:8999/health"


def test_cli_run_without_arguments_uses_default_model_and_starts_server(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: Path("/models/default.gguf"))
    monkeypatch.setattr(
        cli,
        "_run_server",
        lambda model_path, server_args, host, port: calls.append(
            ("server", model_path, list(server_args), host, port)
        )
        or 0,
    )

    assert cli.main(["run"]) == 0

    assert calls == [
        ("setup", True, "utopic_server"),
        ("server", "/models/default.gguf", [], "127.0.0.1", "8910"),
    ]


def test_cli_run_server_reports_missing_binary_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: Path("/models/default.gguf"))
    monkeypatch.setattr(
        cli._native,
        "binary_path",
        lambda name: (_ for _ in ()).throw(RuntimeError("native binary missing")),
    )

    assert cli.main(["run", "--no-setup"]) == 1

    captured = capsys.readouterr()
    assert "utopic run: native binary missing" in captured.err


def test_cli_run_no_setup_checks_server_binary_before_default_model_download(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(
        cli.models,
        "ensure_model",
        lambda value=None: pytest.fail("should not download a model when the server binary is missing"),
    )
    monkeypatch.setattr(
        cli._native,
        "binary_path",
        lambda name: (_ for _ in ()).throw(RuntimeError("native binary missing")),
    )

    assert cli.main(["run", "--no-setup"]) == 1

    captured = capsys.readouterr()
    assert "utopic run: native binary missing" in captured.err


def test_cli_run_prompt_reports_missing_binary_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(
        cli._native,
        "main",
        lambda name, argv: (_ for _ in ()).throw(RuntimeError("native binary missing")),
    )

    assert cli.main(["run", "--no-setup", "-m", "/models/default.gguf", "-p", "hi"]) == 1

    captured = capsys.readouterr()
    assert "utopic run: native binary missing" in captured.err


def test_cli_wait_for_health_reports_early_server_exit(monkeypatch):
    class ExitedProcess:
        def poll(self):
            return -9

    monkeypatch.setattr(cli.time, "monotonic", lambda: 0)

    with pytest.raises(RuntimeError, match="signal 9"):
        cli._wait_for_health(ExitedProcess(), "http://127.0.0.1:8910/health")


def test_cli_run_server_returns_failure_for_signal_exit(monkeypatch, tmp_path):
    class ExitedProcess:
        returncode = -9

        def poll(self):
            return -9

        def terminate(self):
            pytest.fail("already exited process should not be terminated")

    monkeypatch.setattr(cli._native, "binary_path", lambda name: tmp_path / name)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv: ExitedProcess())
    monkeypatch.setattr(cli.time, "monotonic", lambda: 0)

    assert cli._run_server("/models/dream.gguf", [], "127.0.0.1", "8910") == 1


def test_cli_run_server_reaps_process_after_health_timeout(monkeypatch, tmp_path):
    events = []

    class RunningProcess:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            self.returncode = 143
            return self.returncode

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(cli._native, "binary_path", lambda name: tmp_path / name)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv: RunningProcess())
    monkeypatch.setattr(cli, "_wait_for_health", lambda process, health_url: (_ for _ in ()).throw(RuntimeError("timed out")))

    assert cli._run_server("/models/dream.gguf", [], "127.0.0.1", "8910") == 143
    assert events == ["terminate", ("wait", 5)]


def test_cli_run_server_prints_openai_url_after_health(monkeypatch, tmp_path, capsys):
    events = []

    class HealthyProcess:
        returncode = 0

        def poll(self):
            events.append("poll")
            return None

        def wait(self):
            events.append("wait")
            return self.returncode

    process = HealthyProcess()
    monkeypatch.setattr(cli._native, "binary_path", lambda name: tmp_path / name)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda argv: events.append(("popen", argv)) or process)
    monkeypatch.setattr(cli, "_wait_for_health", lambda process, health_url: events.append(("health", health_url)))

    assert cli._run_server("/models/dream.gguf", ["--port", "8999"], "0.0.0.0", "8999") == 0

    captured = capsys.readouterr()
    assert "OpenAI-compatible URL: http://127.0.0.1:8999/v1/chat/completions" in captured.out
    assert events == [
        ("popen", [str(tmp_path / "utopic_server"), "-m", "/models/dream.gguf", "--port", "8999"]),
        ("health", "http://127.0.0.1:8999/health"),
        "wait",
    ]


def test_model_catalog_resolves_hf_download_url():
    entry = models.get_model("dream-7b-q4")

    assert entry is not None
    assert entry.filename.endswith(".gguf")
    assert entry.url.startswith("https://huggingface.co/")


def test_model_resolve_treats_gguf_value_as_local_path():
    resolved = models.resolve_model("/tmp/example.gguf")

    assert resolved == Path("/tmp/example.gguf")


def test_model_resolve_treats_windows_style_path_as_local_path():
    resolved = models.resolve_model("C:\\models\\example.bin")

    assert resolved == Path("C:\\models\\example.bin")


def test_model_pull_reuses_existing_download(monkeypatch, tmp_path):
    catalog = tmp_path / "models.json"
    model_file = tmp_path / "models" / "example.gguf"
    model_file.parent.mkdir()
    model_file.write_text("already here", encoding="utf-8")
    catalog.write_text(
        """
[
  {
    "id": "example",
    "name": "Example",
    "family": "test",
    "filename": "example.gguf",
    "url": "https://example.invalid/example.gguf",
    "size": "1 B",
    "recommended": true,
    "description": "Test model"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(model_file.parent))
    monkeypatch.setattr(models, "_copy_stream_with_progress", lambda url, destination: pytest.fail("should not download"))

    assert models.pull_model("example") == model_file


def test_model_pull_removes_partial_file_on_download_failure(monkeypatch, tmp_path):
    catalog = tmp_path / "models.json"
    models_dir = tmp_path / "models"
    catalog.write_text(
        """
[
  {
    "id": "example",
    "name": "Example",
    "family": "test",
    "filename": "example.gguf",
    "url": "https://example.invalid/example.gguf",
    "size": "1 B",
    "recommended": true,
    "description": "Test model"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(models_dir))

    def fail_download(_url, destination):
        destination.write_text("partial", encoding="utf-8")
        raise OSError("network down")

    monkeypatch.setattr(models, "_copy_stream_with_progress", fail_download)

    with pytest.raises(RuntimeError, match="Failed to pull example"):
        models.pull_model("example")

    assert not (models_dir / "example.gguf.partial").exists()
