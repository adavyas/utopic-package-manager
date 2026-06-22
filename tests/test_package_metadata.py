from pathlib import Path

from utopic import __version__


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_native_mcp_server_info_uses_package_version_constant():
    identity = (REPO_ROOT / "python" / "utopic" / "native" / "utopic_identity.h").read_text(
        encoding="utf-8"
    )
    mcp_server = (REPO_ROOT / "python" / "utopic" / "native" / "mcp_server.cpp").read_text(
        encoding="utf-8"
    )

    assert f'project_version = "{__version__}"' in identity
    assert '{ "version", project_version }' in mcp_server
    assert '{ "version", "0.1.0" }' not in mcp_server
