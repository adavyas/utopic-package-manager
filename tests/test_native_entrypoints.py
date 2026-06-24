from utopic import acp, mcp, server


def test_server_entrypoint_version_does_not_require_native_binary(monkeypatch, capsys):
    monkeypatch.setattr(server.sys, "argv", ["utopic-server", "--version"])
    monkeypatch.setattr(server, "_main", lambda binary_name: (_ for _ in ()).throw(AssertionError("should not launch native binary")))

    assert server.main() == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("utopic-server ")
    assert captured.err == ""


def test_mcp_entrypoint_version_does_not_require_native_binary(monkeypatch, capsys):
    monkeypatch.setattr(mcp.sys, "argv", ["utopic-mcp", "--version"])
    monkeypatch.setattr(mcp, "_main", lambda binary_name: (_ for _ in ()).throw(AssertionError("should not launch native binary")))

    assert mcp.main() == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("utopic-mcp ")
    assert captured.err == ""


def test_acp_entrypoint_version_does_not_require_native_binary(monkeypatch, capsys):
    monkeypatch.setattr(acp.sys, "argv", ["utopic-acp", "--version"])
    monkeypatch.setattr(acp, "_main", lambda binary_name: (_ for _ in ()).throw(AssertionError("should not launch native binary")))

    assert acp.main() == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("utopic-acp ")
    assert captured.err == ""


def test_server_entrypoint_help_does_not_require_native_binary(monkeypatch, capsys):
    monkeypatch.setattr(server.sys, "argv", ["utopic-server", "--help"])
    monkeypatch.setattr(server, "_main", lambda binary_name: (_ for _ in ()).throw(AssertionError("should not launch native binary")))

    assert server.main() == 0

    captured = capsys.readouterr()
    assert "usage: utopic-server" in captured.out
    assert "OpenAI-compatible" in captured.out
    assert captured.err == ""


def test_mcp_entrypoint_help_does_not_require_native_binary(monkeypatch, capsys):
    monkeypatch.setattr(mcp.sys, "argv", ["utopic-mcp", "--help"])
    monkeypatch.setattr(mcp, "_main", lambda binary_name: (_ for _ in ()).throw(AssertionError("should not launch native binary")))

    assert mcp.main() == 0

    captured = capsys.readouterr()
    assert "usage: utopic-mcp" in captured.out
    assert "Model Context Protocol" in captured.out
    assert captured.err == ""


def test_acp_entrypoint_help_does_not_require_native_binary(monkeypatch, capsys):
    monkeypatch.setattr(acp.sys, "argv", ["utopic-acp", "--help"])
    monkeypatch.setattr(acp, "_main", lambda binary_name: (_ for _ in ()).throw(AssertionError("should not launch native binary")))

    assert acp.main() == 0

    captured = capsys.readouterr()
    assert "usage: utopic-acp" in captured.out
    assert "Agent Client Protocol" in captured.out
    assert captured.err == ""


def test_server_entrypoint_reports_missing_binary_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        server,
        "_main",
        lambda binary_name: (_ for _ in ()).throw(RuntimeError("native binary missing")),
    )

    assert server.main() == 1

    captured = capsys.readouterr()
    assert "utopic-server: native binary missing" in captured.err
    assert "Traceback" not in captured.err


def test_mcp_entrypoint_reports_missing_binary_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(mcp.sys, "argv", ["utopic-mcp", "--native"])
    monkeypatch.setattr(
        mcp,
        "_main",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("native binary missing")),
    )

    assert mcp.main() == 1

    captured = capsys.readouterr()
    assert "utopic-mcp: native binary missing" in captured.err
    assert "Traceback" not in captured.err


def test_acp_entrypoint_reports_missing_binary_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        acp,
        "_main",
        lambda binary_name: (_ for _ in ()).throw(RuntimeError("native binary missing")),
    )

    assert acp.main() == 1

    captured = capsys.readouterr()
    assert "utopic-acp: native binary missing" in captured.err
    assert "Traceback" not in captured.err
