import subprocess
from pathlib import Path

import pytest

from utopic import chat, cli, models


def _stub_server_binary(monkeypatch):
    monkeypatch.setattr(cli._native, "binary_path", lambda name: Path(f"/fake/bin/{name}"))


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
    monkeypatch.setattr(chat.installer, "native_installation_is_current", lambda binary_names: True)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: setup_calls.append(list(argv)) or 0)
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: None)

    assert chat.launch(["dream-7b-q4"]) == 0

    assert setup_calls == []


def test_chat_launch_runs_setup_when_server_cache_is_stale(monkeypatch, tmp_path):
    script = tmp_path / "utopic-chat.js"
    script.write_text("console.log('chat')\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(chat, "_chat_script", lambda: script)
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(chat.installer, "cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(chat.installer, "native_installation_is_current", lambda binary_names: False)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: captured.setdefault("setup", list(argv)) or 0)
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: captured.update(command=command))

    assert chat.launch(["dream-7b-q4"]) == 0

    assert captured["setup"] == []
    assert captured["command"] == ["/usr/bin/node", str(script), "dream-7b-q4"]


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


def test_chat_launch_rejects_unknown_options_before_setup(monkeypatch, capsys):
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: pytest.fail("should not run setup"))
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: pytest.fail("should not launch node"))

    assert chat.launch(["--bogus"]) == 1

    captured = capsys.readouterr()
    assert "utopic chat: unknown option: --bogus" in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["dream-7b-q4", "llada-8b-q4"],
        ["-m", "dream-7b-q4", "llada-8b-q4"],
        ["-m", "dream-7b-q4", "-m", "llada-8b-q4"],
    ],
)
def test_chat_launch_rejects_extra_model_arguments_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: pytest.fail("should not run setup"))
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: pytest.fail("should not launch node"))

    assert chat.launch(args) == 1

    captured = capsys.readouterr()
    assert "utopic chat: expected at most one model argument" in captured.err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--server="], "expected a value after --server"),
        (["--server", "--model", "dream-7b-q4"], "expected a value after --server"),
        (["--host="], "expected a value after --host"),
        (["--host", "--port", "8910"], "expected a value after --host"),
        (["--port="], "expected a value after --port"),
        (["--port", "--host", "127.0.0.1"], "expected a value after --port"),
        (["-ngl", "--ctx-size", "4096"], "expected a value after -ngl"),
        (["--ctx-size="], "expected a value after --ctx-size"),
        (["--ctx-size", "--port", "8910"], "expected a value after --ctx-size"),
        (["--max-tokens="], "expected a value after --max-tokens"),
        (["--max-tokens", "--temperature", "0"], "expected a value after --max-tokens"),
        (["--temperature="], "expected a value after --temperature"),
        (["--temperature", "--max-tokens", "16"], "expected a value after --temperature"),
    ],
)
def test_chat_launch_rejects_missing_option_values_before_setup(monkeypatch, capsys, args, message):
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: pytest.fail("should not run setup"))
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: pytest.fail("should not launch node"))

    assert chat.launch(args) == 1

    captured = capsys.readouterr()
    assert f"utopic chat: {message}" in captured.err


def test_cli_run_with_prompt_delegates_to_native_one_shot(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: calls.append((name, list(argv))))

    cli.main(["run", "-m", "model.gguf", "-p", "hello", "-n", "8"])

    assert calls == [
        ("setup", True, "utopic"),
        ("utopic", ["-m", "model.gguf", "-p", "hello", "-n", "8"]),
    ]


def test_cli_run_with_prompt_resolves_model_alias(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: calls.append(("model", value)) or Path("/models/dream.gguf"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: calls.append((name, list(argv))))

    cli.main(["run", "-m", "dream-7b-q4", "-p", "hello"])

    assert calls == [
        ("setup", True, "utopic"),
        ("model", "dream-7b-q4"),
        ("utopic", ["-m", "/models/dream.gguf", "-p", "hello"]),
    ]


def test_cli_run_with_prompt_resolves_positional_model_alias(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: calls.append(("model", value)) or Path("/models/dream.gguf"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: calls.append((name, list(argv))))

    cli.main(["run", "dream-7b-q4", "-p", "hello", "-n", "8"])

    assert calls == [
        ("setup", True, "utopic"),
        ("model", "dream-7b-q4"),
        ("utopic", ["-m", "/models/dream.gguf", "-p", "hello", "-n", "8"]),
    ]


def test_cli_run_with_prompt_without_model_uses_default_model(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: calls.append(("model", value)) or Path("/models/default.gguf"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: calls.append((name, list(argv))))

    cli.main(["run", "-p", "hello", "-n", "8"])

    assert calls == [
        ("setup", True, "utopic"),
        ("model", None),
        ("utopic", ["-m", "/models/default.gguf", "-p", "hello", "-n", "8"]),
    ]


def test_cli_run_prompt_allows_negative_numeric_prompt_values(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: calls.append(("model", value)) or Path("/models/default.gguf"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: calls.append((name, list(argv))))

    cli.main(["run", "-p", "hello", "--seed", "-1"])

    assert calls == [
        ("setup", True, "utopic"),
        ("model", None),
        ("utopic", ["-m", "/models/default.gguf", "-p", "hello", "--seed", "-1"]),
    ]


@pytest.mark.parametrize("args", [["--model="], ["-m", ""]])
def test_cli_run_rejects_empty_model_values_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert "utopic run: expected a value after -m/--model" in captured.err


@pytest.mark.parametrize("args", [["-m", "--port", "8910"], ["--model", "-ngl", "99"], ["--model=-ngl", "99"]])
def test_cli_run_rejects_missing_model_values_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert "utopic run: expected a value after -m/--model" in captured.err


@pytest.mark.parametrize("args", [["--model=", "-p", "hi"], ["-m", "", "-p", "hi"]])
def test_cli_run_prompt_rejects_empty_model_values_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: pytest.fail("should not run native cli"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert "utopic run: expected a value after -m/--model" in captured.err


@pytest.mark.parametrize("args", [["-m", "--port", "8910", "-p", "hi"], ["--model", "-ngl", "99", "-p", "hi"], ["--model=-ngl", "99", "-p", "hi"]])
def test_cli_run_prompt_rejects_missing_model_values_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: pytest.fail("should not run native cli"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert "utopic run: expected a value after -m/--model" in captured.err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-p"], "expected a value after -p"),
        (["-p", "--steps", "32"], "expected a value after -p"),
        (["--prompt"], "expected a value after --prompt"),
        (["--prompt="], "expected a value after --prompt"),
        (["-p", "hi", "--steps"], "expected a value after --steps"),
        (["-p", "hi", "--steps", "-ngl"], "expected a value after --steps"),
        (["-p", "hi", "--steps="], "expected a value after --steps"),
        (["-p", "hi", "--schema"], "expected a value after --schema"),
        (["-p", "hi", "--schema", "--tools", "tools.json"], "expected a value after --schema"),
        (["-p", "hi", "--schema="], "expected a value after --schema"),
    ],
)
def test_cli_run_prompt_rejects_missing_prompt_option_values_before_setup(monkeypatch, capsys, args, message):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: pytest.fail("should not run native cli"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert f"utopic run: {message}" in captured.err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-p", "hi", "-n", "0"], "-n must be a positive integer"),
        (["-p", "hi", "-n", "many"], "-n must be a positive integer"),
        (["-p", "hi", "--steps", "0"], "--steps must be a positive integer"),
        (["-p", "hi", "--steps", "fast"], "--steps must be a positive integer"),
        (["-p", "hi", "--diffusion-block-length", "0"], "--diffusion-block-length must be a positive integer"),
        (["-p", "hi", "--canvas", "-1"], "--canvas must be a non-negative integer"),
        (["-p", "hi", "--canvas", "wide"], "--canvas must be a non-negative integer"),
        (["-p", "hi", "--eb-steps", "-1"], "--eb-steps must be a non-negative integer"),
        (["-p", "hi", "--slot-len", "0"], "--slot-len must be a positive integer"),
        (["-p", "hi", "--converge", "-1"], "--converge must be a non-negative integer"),
        (["-p", "hi", "--temp", "-0.1"], "--temp must be a non-negative number"),
        (["-p", "hi", "--temp", "warm"], "--temp must be a non-negative number"),
        (["-p", "hi", "--seed", "abc"], "--seed must be an integer"),
    ],
)
def test_cli_run_prompt_rejects_invalid_numeric_prompt_values_before_setup(monkeypatch, capsys, args, message):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: pytest.fail("should not run native cli"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert f"utopic run: {message}" in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["dream-7b-q4", "llada-8b-q4"],
        ["-m", "dream-7b-q4", "llada-8b-q4"],
        ["-m", "dream-7b-q4", "-m", "llada-8b-q4"],
        ["dream-7b-q4", "llada-8b-q4", "-p", "hi"],
        ["-m", "dream-7b-q4", "llada-8b-q4", "-p", "hi"],
        ["-m", "dream-7b-q4", "-m", "llada-8b-q4", "-p", "hi"],
    ],
)
def test_cli_run_rejects_extra_model_arguments_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))
    monkeypatch.setattr(cli._native, "main", lambda name, argv: pytest.fail("should not run native cli"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert "utopic run: expected at most one model argument" in captured.err


@pytest.mark.parametrize("args", [["--model=-ngl"], ["--model", "-ngl"]])
def test_chat_launch_rejects_option_like_model_values_before_setup(monkeypatch, capsys, args):
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: pytest.fail("should not run setup"))
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: pytest.fail("should not launch node"))

    assert chat.launch(args) == 1

    captured = capsys.readouterr()
    assert "utopic chat: expected a value after -m/--model" in captured.err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--max-tokens=0"], "--max-tokens must be a positive integer"),
        (["--max-tokens=-5"], "--max-tokens must be a positive integer"),
        (["--max-tokens", "-5"], "--max-tokens must be a positive integer"),
        (["--max-tokens=1.5"], "--max-tokens must be a positive integer"),
        (["--temperature=-1"], "--temperature must be a non-negative number"),
        (["--temperature", "-1"], "--temperature must be a non-negative number"),
        (["--temperature=inf"], "--temperature must be a non-negative number"),
        (["--temperature=nan"], "--temperature must be a non-negative number"),
        (["--temperature", "nan"], "--temperature must be a non-negative number"),
    ],
)
def test_chat_launch_rejects_invalid_sampling_values_before_setup(monkeypatch, capsys, args, message):
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    monkeypatch.setattr(chat.installer, "setup", lambda argv: pytest.fail("should not run setup"))
    monkeypatch.setattr(chat.subprocess, "run", lambda command, env, check: pytest.fail("should not launch node"))

    assert chat.launch(args) == 1

    captured = capsys.readouterr()
    assert f"utopic chat: {message}" in captured.err


def test_cli_ensure_setup_rebuilds_stale_native_cache(monkeypatch):
    calls = []

    monkeypatch.setattr(cli.installer, "native_installation_is_current", lambda binary_names: False)
    monkeypatch.setattr(cli.installer, "setup", lambda argv: calls.append(list(argv)) or 0)

    cli._ensure_setup(True, "utopic_server")

    assert calls == [[]]


def test_cli_ensure_setup_skips_current_native_cache(monkeypatch):
    calls = []

    monkeypatch.setattr(cli.installer, "native_installation_is_current", lambda binary_names: True)
    monkeypatch.setattr(cli.installer, "setup", lambda argv: calls.append(list(argv)) or 0)

    cli._ensure_setup(True, "utopic_server")

    assert calls == []


def test_cli_setup_reports_subprocess_failures_without_traceback(monkeypatch, capsys):
    def fail_setup(argv):
        raise subprocess.CalledProcessError(2, ["cmake", "-B", "/tmp/build"])

    monkeypatch.setattr(cli.installer, "setup", fail_setup)

    assert cli.main(["setup"]) == 2

    captured = capsys.readouterr()
    assert "utopic setup: command failed: cmake -B /tmp/build" in captured.err
    assert "Traceback" not in captured.err
    assert "CalledProcessError" not in captured.err


def test_cli_setup_reports_runtime_failures_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.installer,
        "setup",
        lambda argv: (_ for _ in ()).throw(RuntimeError("native source missing")),
    )

    assert cli.main(["setup"]) == 1

    captured = capsys.readouterr()
    assert "utopic setup: native source missing" in captured.err
    assert "Traceback" not in captured.err


def test_cli_run_without_prompt_starts_openai_server(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": calls.append(("setup", enabled, binary_name)))
    _stub_server_binary(monkeypatch)
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
    _stub_server_binary(monkeypatch)
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


def test_cli_run_rejects_unknown_server_options_before_setup(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": pytest.fail("should not run setup"))
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))
    monkeypatch.setattr(cli._native, "binary_path", lambda name: pytest.fail("should not inspect native binaries"))

    assert cli.main(["run", "--bogus"]) == 1

    captured = capsys.readouterr()
    assert "utopic run: unknown option: --bogus" in captured.err


@pytest.mark.parametrize("flag", ["--host", "--port", "-ngl", "--ctx-size"])
def test_cli_run_rejects_missing_server_flag_values(monkeypatch, capsys, flag):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))

    assert cli.main(["run", flag]) == 1

    captured = capsys.readouterr()
    assert f"utopic run: expected a value after {flag}" in captured.err


@pytest.mark.parametrize("arg", ["--host=", "--port=", "--ctx-size="])
def test_cli_run_rejects_empty_equals_server_flag_values(monkeypatch, capsys, arg):
    flag = arg[:-1]
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))

    assert cli.main(["run", arg]) == 1

    captured = capsys.readouterr()
    assert f"utopic run: expected a value after {flag}" in captured.err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--port", "abc"], "--port must be an integer from 1 to 65535"),
        (["--port=0"], "--port must be an integer from 1 to 65535"),
        (["--port", "65536"], "--port must be an integer from 1 to 65535"),
        (["-ngl", "-1"], "-ngl must be a non-negative integer"),
        (["-ngl", "1.5"], "-ngl must be a non-negative integer"),
        (["--ctx-size=-1"], "--ctx-size must be a positive integer"),
        (["--ctx-size", "4.5"], "--ctx-size must be a positive integer"),
    ],
)
def test_cli_run_rejects_invalid_server_numeric_flags(monkeypatch, capsys, args, message):
    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    monkeypatch.setattr(cli.models, "ensure_model", lambda value=None: pytest.fail("should not resolve a model"))

    assert cli.main(["run", *args]) == 1

    captured = capsys.readouterr()
    assert f"utopic run: {message}" in captured.err


def test_cli_run_normalizes_wildcard_host_for_client_url(monkeypatch):
    calls = []

    monkeypatch.setattr(cli, "_ensure_setup", lambda enabled=True, binary_name="utopic": None)
    _stub_server_binary(monkeypatch)
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
    _stub_server_binary(monkeypatch)
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


def test_model_list_reports_invalid_catalog_json(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    catalog.write_text("{not-json", encoding="utf-8")
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))

    assert models.main(["list"]) == 1

    captured = capsys.readouterr()
    assert "utopic models: Failed to read model catalog" in captured.err
    assert "Traceback" not in captured.err


def test_model_list_reports_non_list_catalog(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    catalog.write_text('{"id": "not-a-list"}', encoding="utf-8")
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))

    assert models.main(["list"]) == 1

    captured = capsys.readouterr()
    assert "utopic models: Model catalog" in captured.err
    assert "must contain a JSON list" in captured.err
    assert "Traceback" not in captured.err


def test_model_list_reports_incomplete_catalog_entry(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    catalog.write_text(
        """
[
  {
    "id": "missing-fields",
    "name": "Missing Fields"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))

    assert models.main(["list"]) == 1

    captured = capsys.readouterr()
    assert "utopic models: Invalid model catalog entry 0" in captured.err
    assert "Traceback" not in captured.err


def test_model_list_reports_empty_catalog(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    catalog.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))

    assert models.main(["list"]) == 1

    captured = capsys.readouterr()
    assert "utopic models: Utopic model catalog is empty" in captured.err
    assert "Traceback" not in captured.err


def test_model_list_reports_non_object_catalog_entry(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    catalog.write_text("[null]", encoding="utf-8")
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))

    assert models.main(["list"]) == 1

    captured = capsys.readouterr()
    assert "utopic models: Invalid model catalog entry 0: expected a JSON object" in captured.err
    assert "Traceback" not in captured.err


def test_model_list_reports_wrong_catalog_field_type(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    catalog.write_text(
        """
[
  {
    "id": 42,
    "name": "Wrong Type",
    "family": "test",
    "filename": "wrong-type.gguf",
    "url": "https://example.invalid/wrong-type.gguf",
    "size": "1 B",
    "recommended": true,
    "description": "Wrong field type"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))

    assert models.main(["list"]) == 1

    captured = capsys.readouterr()
    assert "utopic models: Invalid model catalog entry 0: id must be a string" in captured.err
    assert "Traceback" not in captured.err


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


def test_model_pull_rejects_catalog_filename_outside_models_dir(monkeypatch, tmp_path):
    catalog = tmp_path / "models.json"
    models_dir = tmp_path / "models"
    catalog.write_text(
        """
[
  {
    "id": "escape",
    "name": "Escape",
    "family": "test",
    "filename": "../escape.gguf",
    "url": "https://example.invalid/escape.gguf",
    "size": "1 B",
    "recommended": true,
    "description": "Unsafe test model"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(models_dir))
    monkeypatch.setattr(models, "_copy_stream_with_progress", lambda url, destination: pytest.fail("should not download"))

    with pytest.raises(RuntimeError, match="unsafe model filename"):
        models.pull_model("escape")

    assert not (tmp_path / "escape.gguf").exists()
    assert not (models_dir / "escape.gguf").exists()
    assert not (models_dir / "escape.gguf.partial").exists()


def test_model_pull_rejects_non_http_catalog_url(monkeypatch, tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"local file should not be copied")
    catalog = tmp_path / "models.json"
    models_dir = tmp_path / "models"
    catalog.write_text(
        f"""
[
  {{
    "id": "local-file",
    "name": "Local File",
    "family": "test",
    "filename": "local-file.gguf",
    "url": "{source.as_uri()}",
    "size": "1 B",
    "recommended": true,
    "description": "Unsafe local file URL"
  }}
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(models_dir))

    with pytest.raises(RuntimeError, match="unsupported model URL protocol"):
        models.pull_model("local-file")

    assert not (models_dir / "local-file.gguf").exists()
    assert not (models_dir / "local-file.gguf.partial").exists()


def test_model_pull_rejects_catalog_url_without_host(monkeypatch, tmp_path):
    catalog = tmp_path / "models.json"
    models_dir = tmp_path / "models"
    catalog.write_text(
        """
[
  {
    "id": "missing-host",
    "name": "Missing Host",
    "family": "test",
    "filename": "missing-host.gguf",
    "url": "https:///missing-host.gguf",
    "size": "1 B",
    "recommended": true,
    "description": "Malformed URL"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(models_dir))

    with pytest.raises(RuntimeError, match="must include a host"):
        models.pull_model("missing-host")

    assert not (models_dir / "missing-host.gguf").exists()
    assert not (models_dir / "missing-host.gguf.partial").exists()


def test_model_pull_rejects_malformed_catalog_url(monkeypatch, tmp_path):
    catalog = tmp_path / "models.json"
    models_dir = tmp_path / "models"
    catalog.write_text(
        """
[
  {
    "id": "bad-url",
    "name": "Bad URL",
    "family": "test",
    "filename": "bad-url.gguf",
    "url": "https://[bad]/bad-url.gguf",
    "size": "1 B",
    "recommended": true,
    "description": "Malformed URL"
  }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("UTOPIC_MODELS_CATALOG", str(catalog))
    monkeypatch.setenv("UTOPIC_MODELS_DIR", str(models_dir))

    with pytest.raises(RuntimeError, match="model URL for 'bad-url' must be a URL"):
        models.pull_model("bad-url")

    assert not (models_dir / "bad-url.gguf").exists()
    assert not (models_dir / "bad-url.gguf.partial").exists()


def test_model_pull_redownloads_zero_byte_cached_model(monkeypatch, tmp_path):
    catalog = tmp_path / "models.json"
    model_file = tmp_path / "models" / "example.gguf"
    model_file.parent.mkdir()
    model_file.write_bytes(b"")
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

    def download(_url, destination):
        destination.write_bytes(b"model")

    monkeypatch.setattr(models, "_copy_stream_with_progress", download)

    assert models.pull_model("example") == model_file
    assert model_file.read_bytes() == b"model"


def test_model_pull_rejects_zero_byte_download(monkeypatch, tmp_path):
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

    def empty_download(_url, destination):
        destination.write_bytes(b"")

    monkeypatch.setattr(models, "_copy_stream_with_progress", empty_download)

    with pytest.raises(RuntimeError, match="Failed to pull example"):
        models.pull_model("example")

    assert not (models_dir / "example.gguf").exists()
    assert not (models_dir / "example.gguf.partial").exists()


def test_model_list_marks_zero_byte_cached_model_not_downloaded(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "models.json"
    model_file = tmp_path / "models" / "example.gguf"
    model_file.parent.mkdir()
    model_file.write_bytes(b"")
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

    assert models.main(["list"]) == 0

    captured = capsys.readouterr()
    assert "not downloaded" in captured.out
    assert "downloaded" not in captured.out.replace("not downloaded", "")


def test_model_stream_download_rejects_truncated_content_length(monkeypatch, tmp_path):
    class TruncatedResponse:
        headers = {"content-length": "16"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if getattr(self, "_sent", False):
                return b""
            self._sent = True
            return b"partial"

    destination = tmp_path / "model.gguf.partial"
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url: TruncatedResponse())

    with pytest.raises(OSError, match="downloaded 7 of 16 bytes"):
        models._copy_stream_with_progress("https://example.invalid/model.gguf", destination)

    assert destination.read_bytes() == b"partial"


def test_model_stream_download_rejects_invalid_content_length(monkeypatch, tmp_path):
    class InvalidLengthResponse:
        headers = {"content-length": "not-a-number"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b""

    destination = tmp_path / "model.gguf.partial"
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda url: InvalidLengthResponse())

    with pytest.raises(OSError, match="invalid content-length: not-a-number"):
        models._copy_stream_with_progress("https://example.invalid/model.gguf", destination)

    assert not destination.exists()


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
