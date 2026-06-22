from utopic import acp, mcp, server


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
    monkeypatch.setattr(
        mcp,
        "_main",
        lambda binary_name: (_ for _ in ()).throw(RuntimeError("native binary missing")),
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
