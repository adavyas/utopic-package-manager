import ast
import json
import re
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


def test_release_version_literals_match_package_version():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    chat_ts = (REPO_ROOT / "node" / "utopic-chat.ts").read_text(encoding="utf-8")
    chat_js = (
        REPO_ROOT / "python" / "utopic" / "node" / "utopic-chat.js"
    ).read_text(encoding="utf-8")
    identity = (REPO_ROOT / "python" / "utopic" / "native" / "utopic_identity.h").read_text(
        encoding="utf-8"
    )

    setup_call = next(
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    setup_version = next(
        keyword.value.value
        for keyword in setup_call.keywords
        if keyword.arg == "version" and isinstance(keyword.value, ast.Constant)
    )

    assert f'version = "{__version__}"' in pyproject
    assert setup_version == __version__
    assert re.search(rf'const VERSION = "{re.escape(__version__)}";', chat_ts)
    assert re.search(rf'const VERSION = "{re.escape(__version__)}";', chat_js)
    assert f'project_version = "{__version__}"' in identity


def test_chat_check_script_rejects_stale_bundled_javascript():
    package_json = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    check_script = package_json["scripts"]["check:chat"]

    assert "npm run build:chat" in check_script
    assert "git diff --exit-code -- python/utopic/node/utopic-chat.js" in check_script


def test_readme_uses_release_build_commands():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m build" in readme
    assert "python -m twine check dist/*" in readme
    assert "python -m pip wheel . --no-deps -w dist/" not in readme


def test_release_workflow_smokes_installed_prompt_flag_normalization():
    workflow = (REPO_ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )

    assert "import textwrap" in workflow
    assert "prompt_probe = textwrap.dedent" in workflow
    assert '"--model=dream-7b-q4"' in workflow
    assert '"--prompt=hello"' in workflow
    assert '["-m", "/models/dream.gguf", "-p", "hello"' in workflow
