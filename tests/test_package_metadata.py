import ast
import json
import re
import sys
from pathlib import Path

from utopic import __version__
from utopic import installer


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_NATIVE_DIR = REPO_ROOT / "python" / "utopic" / "core" / "native"
CORE_CATALOG_PATH = (
    REPO_ROOT
    / "python"
    / "utopic"
    / "core"
    / "python"
    / "utopic_core"
    / "models.json"
)
EXPECTED_NATIVE_REF = "5698e53af77b9c81a4324c599baf2bc1ac6d82fb"

REQUIRED_NATIVE_RUNNER_FILES = {
    "runner.cpp",
    "runner_contract.cpp",
    "runner_contract.h",
    "runner_plugin.cpp",
    "runner_plugin.h",
    "runner_plugin_api.h",
    "runner_tasks.cpp",
    "runner_tasks.h",
    "audio_engine.cpp",
    "audio_engine.h",
    "sherpa_tts_plugin.cpp",
    "image_engine.cpp",
    "image_engine.h",
    "video_engine.cpp",
    "video_engine.h",
}


def test_native_mcp_server_info_uses_package_version_constant():
    identity = (REPO_ROOT / "python" / "utopic" / "core" / "native" / "utopic_identity.h").read_text(
        encoding="utf-8"
    )
    mcp_server = (REPO_ROOT / "python" / "utopic" / "core" / "native" / "mcp_server.cpp").read_text(
        encoding="utf-8"
    )

    assert f'project_version = "{__version__}"' in identity
    assert '{ "version", project_version }' in mcp_server
    assert '{ "version", "0.1.0" }' not in mcp_server


def test_release_version_literals_match_package_version():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    chat_js = (
        REPO_ROOT / "python" / "utopic" / "core" / "python" / "utopic_core" / "node" / "utopic-chat.js"
    ).read_text(encoding="utf-8")
    identity = (REPO_ROOT / "python" / "utopic" / "core" / "native" / "utopic_identity.h").read_text(
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
    assert re.search(rf'const VERSION = "{re.escape(__version__)}";', chat_js)
    assert f'project_version = "{__version__}"' in identity


def test_native_ref_metadata_matches_installer_pin():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["utopic"]["native-ref"] == installer.UTOPIC_NATIVE_REF
    assert installer.UTOPIC_NATIVE_REF == EXPECTED_NATIVE_REF


def test_vendored_core_marks_native_image_model_ready():
    catalog = json.loads(CORE_CATALOG_PATH.read_text(encoding="utf-8"))
    entry = next(item for item in catalog if item["id"] == "flux-1-schnell-q4-native")

    assert entry["runtime"] == "native"
    assert entry["engine"] == "stable-diffusion-cpp"
    assert entry["runner"] == "utopic_runner"
    assert entry["native_status"] == "ready"


def test_vendored_core_exposes_native_plugin_abi_header():
    header = CORE_NATIVE_DIR / "runner_plugin_api.h"

    assert header.is_file()
    source = header.read_text(encoding="utf-8")
    assert "UTOPIC_NATIVE_PLUGIN_DEFAULT_ENTRYPOINT" in source
    assert "utopic_native_generate_fn" in source


def test_package_manager_no_longer_owns_legacy_native_source():
    assert not (REPO_ROOT / "python" / "utopic" / "native").exists()


def test_package_manager_no_longer_owns_typescript_chat_source():
    assert not (REPO_ROOT / "node" / "utopic-chat.ts").exists()


def test_vendor_script_sources_chat_artifact_from_typescript_build_output():
    vendor_script = (REPO_ROOT / "scripts" / "vendor_core.py").read_text(encoding="utf-8")

    assert 'run(["npm", "ci"], cwd=chat_dir)' in vendor_script
    assert 'run(["npm", "run", "build"], cwd=chat_dir)' in vendor_script
    assert 'tmp / "chat" / "dist" / "utopic-chat.js"' in vendor_script
    assert 'tmp / "python" / "utopic_core" / "node"' not in vendor_script


def test_vendored_core_layout_exists():
    assert (REPO_ROOT / "python" / "utopic" / "cmake" / "CMakeLists.txt").exists()
    assert not (REPO_ROOT / "python" / "utopic" / "core" / "native" / "CMakeLists.txt").exists()
    assert (REPO_ROOT / "python" / "utopic" / "core" / "native" / "main.cpp").exists()
    assert (
        REPO_ROOT
        / "python"
        / "utopic"
        / "core"
        / "python"
        / "utopic_core"
        / "models.json"
    ).exists()
    assert (
        REPO_ROOT
        / "python"
        / "utopic"
        / "core"
        / "python"
        / "utopic_core"
        / "node"
        / "utopic-chat.js"
    ).exists()


def test_vendored_core_includes_native_runner_sources():
    native_dir = REPO_ROOT / "python" / "utopic" / "core" / "native"

    missing = sorted(
        filename
        for filename in REQUIRED_NATIVE_RUNNER_FILES
        if not (native_dir / filename).exists()
    )

    assert missing == []


def test_package_cmake_builds_native_runner_and_multimodal_sources():
    cmake = (REPO_ROOT / "python" / "utopic" / "cmake" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "add_executable(utopic_runner" in cmake
    for filename in (
        "runner.cpp",
        "runner_contract.cpp",
        "runner_tasks.cpp",
        "runner_plugin.cpp",
        "audio_engine.cpp",
        "video_engine.cpp",
    ):
        assert f'${{UTOPIC_NATIVE_SOURCE_DIR}}/{filename}' in cmake
    assert "UTOPIC_ENABLE_STABLE_DIFFUSION" in cmake
    assert "${UTOPIC_NATIVE_SOURCE_DIR}/image_engine.cpp" in cmake
    assert "${UTOPIC_STABLE_DIFFUSION_DIR}/include/stable-diffusion.h" in cmake
    assert 'add_subdirectory("${UTOPIC_STABLE_DIFFUSION_DIR}"' in cmake
    assert 'INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_DIR};${LLAMA_DIR}/ggml/include"' in cmake
    assert "target_link_libraries(utopic_runner PRIVATE ${UTOPIC_RUNNER_LIBS})" in cmake
    assert "stable-diffusion" in cmake
    assert "target_compile_definitions(utopic_runner PRIVATE" in cmake


def test_package_cmake_builds_optional_sherpa_tts_native_plugin():
    cmake = (REPO_ROOT / "python" / "utopic" / "cmake" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert "UTOPIC_ENABLE_SHERPA_ONNX" in cmake
    assert "UTOPIC_SHERPA_ONNX_DIR" in cmake
    assert "sherpa-onnx/c-api/c-api.h" in cmake
    assert "add_library(utopic_sherpa_tts SHARED" in cmake
    assert "${UTOPIC_NATIVE_SOURCE_DIR}/sherpa_tts_plugin.cpp" in cmake
    assert "${UTOPIC_NATIVE_SOURCE_DIR}/audio_engine.cpp" in cmake
    assert "target_link_libraries(utopic_sherpa_tts PRIVATE ${UTOPIC_SHERPA_ONNX_LIBRARY})" in cmake


def test_gateway_console_script_is_declared():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")

    assert 'utopic-runtime = "utopic.gateway:main"' in pyproject
    assert 'utopic-bridge = "utopic.bridge:main"' in pyproject
    assert '"utopic-runtime=utopic.gateway:main"' in setup_source
    assert '"utopic-bridge=utopic.bridge:main"' in setup_source
    assert "utopic.gateway:main" in ast.unparse(setup_tree)
    assert "utopic.bridge:main" in ast.unparse(setup_tree)


def test_python_version_range_matches_bridge_dependency_support():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    setup_python_requires = next(
        keyword.value.value
        for keyword in setup_call.keywords
        if keyword.arg == "python_requires" and isinstance(keyword.value, ast.Constant)
    )

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    assert setup_python_requires == ">=3.10,<3.13"


def test_github_workflows_use_supported_python_versions():
    workflows = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "python-publish.yml",
    ]

    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert 'python-version: "3.x"' not in workflow
        assert '- "3.9"' not in workflow
        assert '- "3.13"' not in workflow
        assert '- "3.10"' in workflow
        assert '- "3.11"' in workflow
        assert '- "3.12"' in workflow
        assert 'python-version: "3.12"' in workflow


def test_bridge_optional_dependency_extras_are_declared():
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    pyproject_extras = pyproject["project"]["optional-dependencies"]

    expected_extras = {"image", "tts", "chatterbox", "music", "video", "bridge", "all"}

    assert expected_extras <= set(pyproject_extras)
    assert "extras_require" not in setup_source
    assert "diffusers>=0.35.0" in pyproject_extras["image"]
    assert "torchvision>=0.21.0" in pyproject_extras["image"]
    assert "kokoro>=0.9.0" in pyproject_extras["tts"]
    assert "torchvision>=0.21.0" in pyproject_extras["tts"]
    assert "chatterbox-tts>=0.1.0" not in pyproject_extras["tts"]
    assert "chatterbox-tts>=0.1.0" in pyproject_extras["chatterbox"]
    assert "setuptools<81" in pyproject_extras["chatterbox"]
    assert "chatterbox-tts>=0.1.0" not in pyproject_extras["bridge"]
    assert "soundfile>=0.12.0" in pyproject_extras["music"]
    assert "torchcodec>=0.8.0" in pyproject_extras["music"]
    assert "imageio>=2.34.0" in pyproject_extras["video"]
    assert "torchvision>=0.21.0" in pyproject_extras["video"]
    assert sorted(set(pyproject_extras["image"] + pyproject_extras["tts"] + pyproject_extras["music"] + pyproject_extras["video"])) == pyproject_extras["bridge"]
    assert pyproject_extras["bridge"] == pyproject_extras["all"]
    assert all(
        "git+" not in dependency
        for dependencies in pyproject_extras.values()
        for dependency in dependencies
    )


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

    assert "node --check python/utopic/core/python/utopic_core/node/utopic-chat.js" in check_script
    assert "npm run build:chat" not in check_script


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
    assert "python/utopic/core/python/utopic_core/models.json" in workflow
    assert "python/utopic/cmake/CMakeLists.txt" in workflow
    assert "python/utopic/core/native/main.cpp" in workflow
    assert "python/utopic/core/python/utopic_core/node/utopic-chat.js" in workflow
    assert "utopic/core/python/utopic_core/models.json" in workflow
    assert "utopic/cmake/CMakeLists.txt" in workflow
    assert "utopic/core/native/main.cpp" in workflow
    assert "utopic/core/python/utopic_core/node/utopic-chat.js" in workflow
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
    assert "Installed stale-Node chat fallback smoke failed" in workflow
    assert "fake-node-bin" in workflow
    assert "v16.20.2" in workflow
    assert "stale-node fallback ok" in workflow
    assert "Node.js 18 or newer is required; found v16.20.2; using the built-in Python chat fallback." in workflow
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

    assert "python/utopic/core/python/utopic_core/models.json" in workflow
    assert "python/utopic/cmake/CMakeLists.txt" in workflow
    assert "python/utopic/core/native/main.cpp" in workflow
    assert "python/utopic/core/python/utopic_core/node/utopic-chat.js" in workflow
    assert "utopic/core/python/utopic_core/models.json" in workflow
    assert "utopic/cmake/CMakeLists.txt" in workflow
    assert "utopic/core/native/main.cpp" in workflow
    assert "utopic/core/python/utopic_core/node/utopic-chat.js" in workflow
    assert "Installed Node-free chat fallback smoke failed" in workflow
    assert 'PATH=""' in workflow
    assert "node-free fallback ok" in workflow
    assert "Installed stale-Node chat fallback smoke failed" in workflow
    assert "fake-node-bin" in workflow
    assert "v16.20.2" in workflow
    assert "stale-node fallback ok" in workflow
    assert "utopic chat: Node.js was not found; using the built-in Python chat fallback." in workflow
    assert "Node.js 18 or newer is required; found v16.20.2; using the built-in Python chat fallback." in workflow


def test_workflows_smoke_installed_openai_v1_server_base_url():
    workflows = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "python-publish.yml",
    ]

    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "Installed OpenAI /v1 server-base chat smoke failed" in workflow
        assert '"chat",' in workflow
        assert '"--server",' in workflow
        assert 'f"http://127.0.0.1:{server.server_port}/v1"' in workflow
        assert 'self.path != "/health"' in workflow
        assert 'self.path != "/v1/chat/completions"' in workflow
        assert 'payload.get("messages", [])[-1:] != [{"role": "user", "content": "hello"}]' in workflow
        assert 'payload.get("messages") != [{"role": "user", "content": "hello"}]' not in workflow
        assert "openai-v1 server-base ok" in workflow


def test_readme_distinguishes_server_mode_from_chat_mode():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "`utopic run` is the server process, not an interactive prompt." in readme
    assert "utopic chat --server http://127.0.0.1:8910" in readme


def test_readme_documents_chat_tui_and_node_free_fallback():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "When Node.js 18 or newer is on `PATH`, `utopic chat` uses the bundled TypeScript/Node TUI with a `>>>` prompt and streaming output." in readme
    assert "If Node is missing or older than 18, `utopic chat` falls back to a minimal built-in Python chat loop" in readme
    assert "install Node.js 18 or newer for the richer TUI." in readme


def test_readme_documents_supported_models_without_prohibited_mentions():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    catalog = json.loads(CORE_CATALOG_PATH.read_text(encoding="utf-8"))

    assert "## Models" in readme
    for entry in catalog:
        assert entry["id"] in readme
        assert entry["name"] in readme
    assert all(entry["family"] != "diffusiongemma" for entry in catalog)
    assert any(entry["family"] == "diffusion-gemma" for entry in catalog)
    assert "DiffusionGemma is exposed as curated aliases" in readme
    assert "diffusiongemma-26b-a4b-q4" in readme
    assert "qwen-image" in readme
    assert "wan2.1-t2v-14b" in readme
    assert "utopic gateway --port 8911" in readme
    assert "utopic-bridge/v1" in readme
    assert "utopic-bridge diffusers" in readme
    assert "MCP `initialize`, `ping`, `tools/list`, and `tools/call`" in readme
    assert '"repo": "Qwen/Qwen-Image"' in readme
    assert "`repo` is the upstream model source" in readme
    assert "UTOPIC_BRIDGE_DIFFUSERS_COMMAND" in readme
    assert "By default, the gateway runs the packaged bridge as" in readme
    assert "`python -m utopic.bridge <engine>`" in readme
    assert "utopic models check qwen-image" in readme
    assert "utopic models check --all" in readme
    assert "utopic_models_check" in readme
    assert 'uv pip install "utopic[image]"' in readme
    assert 'uv pip install "utopic[tts]"' in readme
    assert 'uv pip install "utopic[chatterbox]"' in readme
    assert 'uv pip install "utopic[music]"' in readme
    assert "including TorchCodec" in readme
    assert "ACE-Step currently works best in a Python 3.10 bridge environment" in readme
    assert "uv pip install git+https://github.com/ace-step/ACE-Step.git" in readme
    assert 'uv pip install "utopic[video]"' in readme
    assert 'uv pip install "utopic[bridge]"' in readme
    assert "utopic run qwen-image" in readme
    assert "bridge-only models start the gateway without starting a native text server" in readme.lower()
    assert "utopic-bridge diffusers --check" in readme
    assert "torch/torchvision versions are incompatible" in readme
    assert "/v1/utopic/runs/{run_id}/events" in readme
    assert "GB10/DGX Spark, a 6x RTX 4090 host, and a 4x A100 host." in readme
    assert "DiffusionGemma Q4_K_M," in readme
    assert "Q5_K_M, Q6_K, and Q8_0 all pull, size-check, load, fully offload" in readme
    assert "Q8_0 all pull, size-check, load, fully offload" in readme
    assert "DiffusionGemma Q4_K_M native C++ smoke tests on GB10/DGX Spark" in readme
    assert "Q5_K_M, Q6_K, and Q8_0 native C++ smoke tests on 4x A100 CUDA" in readme
    assert "Q4_K_M also completes native C++" in readme
    assert "smoke tests on GB10/DGX Spark and 6x RTX 4090 CUDA" in readme
    assert "Native text generation" in readme
    assert "CUDA compiler/toolkit mismatch" in readme
    assert "LLaDA2.0" not in readme
    assert "LLaDA 2.0" not in readme


def test_model_catalog_declares_runtime_schema_for_every_entry():
    catalog = json.loads(CORE_CATALOG_PATH.read_text(encoding="utf-8"))

    required_fields = {
        "modality",
        "engine",
        "runtime",
        "hardware",
        "endpoints",
        "outputs",
    }
    valid_modalities = {"text", "image", "tts", "music", "video", "misc"}
    valid_runtimes = {"native", "bridge"}

    for entry in catalog:
        assert required_fields <= set(entry), entry["id"]
        assert entry["modality"] in valid_modalities
        assert isinstance(entry["engine"], str) and entry["engine"]
        assert entry["runtime"] in valid_runtimes
        assert isinstance(entry["hardware"], list) and entry["hardware"]
        assert all(isinstance(item, str) and item for item in entry["hardware"])
        assert isinstance(entry["endpoints"], list) and entry["endpoints"]
        assert all(isinstance(item, str) and item.startswith("/v1/") for item in entry["endpoints"])
        assert isinstance(entry["outputs"], list) and entry["outputs"]
        assert all(isinstance(item, str) and item for item in entry["outputs"])


def test_catalog_defaults_to_diffusiongemma_and_excludes_legacy_masked_models():
    catalog = json.loads(CORE_CATALOG_PATH.read_text(encoding="utf-8"))

    recommended = [entry["id"] for entry in catalog if entry["recommended"]]
    families = {entry["family"] for entry in catalog}

    assert recommended == ["diffusiongemma-26b-a4b-q4"]
    assert "dream" not in families
    assert "llada" not in families


def test_model_catalog_includes_first_multimodal_model_set():
    catalog = json.loads(CORE_CATALOG_PATH.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in catalog}

    expected = {
        "diffusiongemma-26b-a4b-q4": ("text", "native"),
        "diffusiongemma-26b-a4b-q5": ("text", "native"),
        "diffusiongemma-26b-a4b-q6": ("text", "native"),
        "diffusiongemma-26b-a4b-q8": ("text", "native"),
        "qwen-image": ("image", "bridge"),
        "flux-1-schnell": ("image", "bridge"),
        "krea-2-raw": ("image", "bridge"),
        "cosmos3-super": ("image", "bridge"),
        "kokoro-82m": ("tts", "native"),
        "chatterbox": ("tts", "bridge"),
        "dia-1.6b": ("tts", "bridge"),
        "ace-step-3.5b": ("music", "bridge"),
        "wan2.1-t2v-1.3b": ("video", "bridge"),
        "wan2.1-t2v-14b": ("video", "bridge"),
        "ltx-video": ("video", "bridge"),
    }

    for model_id, (modality, runtime) in expected.items():
        assert model_id in by_id
        assert by_id[model_id]["modality"] == modality
        assert by_id[model_id]["runtime"] == runtime

    assert by_id["cosmos3-super"]["requirements"]["min_gpu_memory_gib"] == 96
    assert by_id["cosmos3-super"]["requirements"]["allow_cpu"] is False
    assert "mac-48gb" not in by_id["krea-2-raw"]["hardware"]


def test_release_workflow_smokes_installed_prompt_flag_normalization():
    workflow = (REPO_ROOT / ".github" / "workflows" / "python-publish.yml").read_text(
        encoding="utf-8"
    )

    assert '[str(python), "-m", "utopic", "--help"]' in workflow
    assert '[str(python), "-m", "utopic", "--version"]' in workflow
    assert "import textwrap" in workflow
    assert "prompt_probe = textwrap.dedent" in workflow
    assert '"--model=diffusiongemma-26b-a4b-q4"' in workflow
    assert '"--prompt=hello"' in workflow
    assert '["-m", "/models/diffusiongemma.gguf", "-p", "hello"' in workflow


def test_workflows_smoke_installed_doctor_command():
    workflows = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "python-publish.yml",
    ]

    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert '[str(utopic), "doctor", "--help"]' in workflow
        assert '[str(utopic), "doctor", "--version"]' in workflow
