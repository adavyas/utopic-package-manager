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


def test_python_module_entrypoint_is_shipped():
    module_entrypoint = REPO_ROOT / "python" / "utopic" / "__main__.py"

    assert module_entrypoint.exists()
    assert "from .cli import main" in module_entrypoint.read_text(encoding="utf-8")


def test_readme_pinned_install_example_matches_package_version():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert f"utopic=={__version__}" in readme
    assert "utopic==0.1.3" not in readme


def test_chat_check_script_rejects_stale_bundled_javascript():
    package_json = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    check_script = package_json["scripts"]["check:chat"]

    assert "npm run build:chat" in check_script
    assert "git diff --exit-code -- python/utopic/node/utopic-chat.js" in check_script


def test_gitignore_covers_generated_release_artifacts():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".pytest_cache/" in gitignore
    assert "*.egg-info/" in gitignore
    assert "__pycache__/" in gitignore
    assert "node_modules/" in gitignore
    assert "dist/" in gitignore
    assert "build/" in gitignore


def test_readme_uses_release_build_commands():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m build" in readme
    assert "python -m twine check dist/*" in readme
    assert "python -m pip wheel . --no-deps -w dist/" not in readme


def test_readme_documents_manual_release_validation_without_publishing():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Manual `workflow_dispatch` runs validate release artifacts only." in readme
    assert "Only a published GitHub Release can run the PyPI publish job." in readme


def test_release_workflow_only_publishes_from_github_releases():
    workflow = (REPO_ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "pypi-publish:" in workflow
    assert "if: github.event_name == 'release'" in workflow
    assert "id-token: write" in workflow
    assert "name: pypi" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow


def test_release_workflow_checks_pypi_version_before_publishing():
    workflow = (REPO_ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )

    assert "Check PyPI version availability" in workflow
    assert "https://pypi.org/pypi/utopic/json" in workflow
    assert "version is already published on PyPI" in workflow
    assert workflow.index("Check PyPI version availability") < workflow.index(
        "pypa/gh-action-pypi-publish@release/v1"
    )


def test_ci_workflow_runs_on_commits_without_publishing():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"

    assert workflow_path.exists()
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "PYTHONPATH=python pytest tests -q" in workflow
    assert "npm run check:chat" in workflow
    assert "python -m pip install --upgrade build pytest twine uv" in workflow
    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "Smoke test built distributions" in workflow
    assert "Smoke test uv tool install" in workflow
    assert "UV_TOOL_DIR" in workflow
    assert '"uv",' in workflow
    assert '"tool",' in workflow
    assert '"install",' in workflow
    assert '"--no-index",' in workflow
    assert '"--find-links",' in workflow
    assert '"dist",' in workflow
    assert '"utopic",' in workflow
    assert '["setup", "--backend", "cpu", "--dry-run", "--jobs", "1"]' in workflow
    assert "Installed Node-free chat fallback smoke failed" in workflow
    assert 'utopic_server = bin_dir / ("utopic-server.exe" if os.name == "nt" else "utopic-server")' in workflow
    assert '[str(utopic), "setup", "--version"]' in workflow
    assert '[str(utopic), "models", "--version"]' in workflow
    assert '[str(utopic), "run", "--version"]' in workflow
    assert '[str(utopic_server), "--help"]' in workflow
    assert "Invalid UTOPIC_BUILD_JOBS did not fail cleanly" in workflow
    assert "Installed run prompt normalization smoke failed" in workflow
    assert 'PATH=""' in workflow
    assert '[str(utopic), "--help"]' in workflow
    assert "pypi-publish" not in workflow
    assert "pypa/gh-action-pypi-publish" not in workflow


def test_release_workflow_smokes_installed_node_free_chat_fallback():
    workflow = (REPO_ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )

    assert "Installed Node-free chat fallback smoke failed" in workflow
    assert 'PATH=""' in workflow
    assert "node-free fallback ok" in workflow
    assert "utopic chat: Node.js was not found; using the built-in Python chat fallback." in workflow


def test_readme_distinguishes_server_mode_from_chat_mode():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "`utopic run` is the server process, not an interactive prompt." in readme
    assert "utopic chat --server http://127.0.0.1:8910" in readme


def test_readme_documents_chat_tui_and_node_free_fallback():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "When Node.js 18 or newer is on `PATH`, `utopic chat` uses the bundled TypeScript/Node TUI." in readme
    assert "If Node is missing, `utopic chat` falls back to a minimal built-in Python chat loop" in readme
    assert "install Node.js 18 or newer for the richer TUI." in readme


def test_readme_documents_supported_models_without_prohibited_mentions():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    catalog_path = REPO_ROOT / "python" / "utopic" / "models.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert "## Models" in readme
    for entry in catalog:
        assert entry["id"] in readme
        assert entry["name"] in readme
    assert all(entry["family"] != "diffusiongemma" for entry in catalog)
    assert "DiffusionGemma is not exposed as a one-command curated download yet" in readme
    assert "GB10/DGX Spark with" in readme
    assert "CUDA compiler/toolkit mismatch" in readme
    assert "LLaDA2.0" not in readme
    assert "LLaDA 2.0" not in readme


def test_release_workflow_smokes_installed_prompt_flag_normalization():
    workflow = (REPO_ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )

    assert '[str(python), "-m", "utopic", "--help"]' in workflow
    assert '[str(python), "-m", "utopic", "--version"]' in workflow
    assert "import textwrap" in workflow
    assert "prompt_probe = textwrap.dedent" in workflow
    assert '"--model=dream-7b-q4"' in workflow
    assert '"--prompt=hello"' in workflow
    assert '["-m", "/models/dream.gguf", "-p", "hello"' in workflow


def test_workflows_smoke_installed_doctor_command():
    workflows = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "python-publish.yml",
    ]

    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert '[str(utopic), "doctor", "--help"]' in workflow
        assert '[str(utopic), "doctor", "--version"]' in workflow
