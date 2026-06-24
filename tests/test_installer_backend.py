import subprocess
from pathlib import Path

import pytest

from utopic import installer


def _write_executable(path: Path) -> None:
    path.write_text("binary", encoding="utf-8")
    path.chmod(0o755)


def test_dry_run_quotes_arguments_that_need_shell_escaping(capsys):
    installer._run(["cmake", "-DCMAKE_BUILD_RPATH=/a;/b"], dry_run=True)

    assert capsys.readouterr().out == "+ cmake '-DCMAKE_BUILD_RPATH=/a;/b'\n"


def test_setup_installs_native_json_runner_binary():
    assert "utopic_runner" in installer.BIN_NAMES


def test_auto_backend_prefers_metal_when_available(monkeypatch):
    monkeypatch.setattr(installer, "_detect_metal_device", lambda: "Apple M4 Pro")
    monkeypatch.setattr(installer, "_detect_cuda_architectures", lambda: "80")
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: Path("/usr/local/cuda/bin/nvcc"))

    decision = installer._resolve_backend("auto", None)

    assert decision.backend == "metal"
    assert decision.device == "Apple M4 Pro"
    assert decision.reason == "Metal device available"


def test_detect_metal_device_falls_back_to_system_profiler_when_probe_fails(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "/usr/bin/clang++":
            return subprocess.CompletedProcess(command, 1, "", "compile denied")
        if command == ["system_profiler", "SPDisplaysDataType"]:
            return subprocess.CompletedProcess(
                command,
                0,
                """
Graphics/Displays:

    Apple M4 Pro:

      Chipset Model: Apple M4 Pro
      Type: GPU
      Metal: Supported
""",
                "",
            )
        raise AssertionError(command)

    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/clang++" if name == "clang++" else None)
    monkeypatch.setattr(installer.subprocess, "run", fake_run)

    assert installer._detect_metal_device() == "Apple M4 Pro"
    assert calls[0][0] == "/usr/bin/clang++"
    assert calls[1] == ["system_profiler", "SPDisplaysDataType"]


def test_auto_backend_uses_cuda_when_cuda_compiler_is_available(monkeypatch):
    monkeypatch.setattr(installer, "_detect_metal_device", lambda: None)
    monkeypatch.setattr(installer, "_detect_cuda_architectures", lambda: "80")
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: Path("/usr/local/cuda/bin/nvcc"))

    decision = installer._resolve_backend("auto", None)

    assert decision.backend == "cuda"
    assert decision.device == "CUDA arch 80"
    assert decision.cuda_architectures == "80"
    assert decision.reason == "NVIDIA CUDA compiler available"


def test_auto_backend_falls_back_to_cpu_without_gpu_backend(monkeypatch):
    monkeypatch.setattr(installer, "_detect_metal_device", lambda: None)
    monkeypatch.setattr(installer, "_detect_cuda_architectures", lambda: None)
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: None)

    decision = installer._resolve_backend("auto", None)

    assert decision.backend == "cpu"
    assert decision.device == "CPU"
    assert decision.reason == "No usable Metal device or CUDA compiler found"


def test_metal_backend_adds_explicit_cmake_flags(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="metal",
        cuda_architectures=None,
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-DGGML_METAL=ON" in configure
    assert "-DGGML_CUDA=OFF" in configure


def test_cuda_build_disables_graphs_for_gb10_architecture(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: None)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-DGGML_CUDA=ON" in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=121" in configure
    assert "-DGGML_CUDA_GRAPHS=OFF" in configure


def test_cuda_build_keeps_graphs_for_non_gb10_architecture(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: None)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="89",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-DGGML_CUDA_GRAPHS=ON" in configure


def test_cuda_build_resets_cached_cuda_matmul_force_options(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: None)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-DGGML_CUDA_FORCE_CUBLAS=OFF" in configure
    assert "-DGGML_CUDA_FORCE_MMQ=OFF" in configure


def test_cuda_build_sets_toolkit_root_for_selected_cuda_compiler(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        installer,
        "_find_cuda_compiler",
        lambda cuda_architectures=None: Path("/usr/local/cuda-13.0/bin/nvcc"),
    )
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc" in configure
    assert "-DCUDAToolkit_ROOT=/usr/local/cuda-13.0" in configure


def test_cuda_build_clears_stale_cmake_cuda_toolkit_cache(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        installer,
        "_find_cuda_compiler",
        lambda cuda_architectures=None: Path("/usr/local/cuda-13.0/bin/nvcc"),
    )
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-UCUDAToolkit_*" in configure
    assert "-U_cmake_CUDAToolkit_*" in configure
    assert "-UFIND_PACKAGE_MESSAGE_DETAILS_CUDAToolkit" in configure


def test_cuda_build_clears_stale_legacy_cuda_library_cache(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        installer,
        "_find_cuda_compiler",
        lambda cuda_architectures=None: Path("/usr/local/cuda-13.0/bin/nvcc"),
    )
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-UCUDA_*" in configure


def test_cuda_build_adds_toolkit_library_dirs_to_build_rpath(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        installer,
        "_find_cuda_compiler",
        lambda cuda_architectures=None: Path("/usr/local/cuda-13.0/bin/nvcc"),
    )
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert (
        "-DCMAKE_BUILD_RPATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib;"
        "/usr/local/cuda-13.0/lib64;/usr/local/cuda-13.0/lib"
    ) in configure


def test_cuda_graphs_environment_override_wins_over_gb10_default(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setenv("UTOPIC_CUDA_GRAPHS", "ON")
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: None)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_llama(
        tmp_path,
        backend="cuda",
        cuda_architectures="121",
        jobs=None,
        dry_run=True,
    )

    configure = commands[0]
    assert "-DGGML_CUDA_GRAPHS=ON" in configure


def test_install_binaries_ad_hoc_signs_macos_executables(monkeypatch, tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    for name in installer.BIN_NAMES:
        _write_executable(build_dir / name)
    dest_dir = tmp_path / "bin"
    commands = []

    monkeypatch.setattr(installer, "bin_dir", lambda: dest_dir)
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._install_binaries(build_dir)

    assert commands == [
        ["codesign", "--force", "--sign", "-", dest_dir / name]
        for name in installer.BIN_NAMES
    ]


def test_install_binaries_copies_optional_shared_plugins(monkeypatch, tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    for name in installer.BIN_NAMES:
        _write_executable(build_dir / name)
    for name in installer.SHARED_PLUGIN_NAMES:
        _write_executable(build_dir / f"{name}.dylib")
    dest_dir = tmp_path / "bin"
    commands = []

    monkeypatch.setattr(installer, "bin_dir", lambda: dest_dir)
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._install_binaries(build_dir)

    for name in installer.SHARED_PLUGIN_NAMES:
        plugin = dest_dir / f"{name}.dylib"
        assert plugin.is_file()
        assert plugin.read_text(encoding="utf-8") == "binary"

    assert commands == [
        ["codesign", "--force", "--sign", "-", dest_dir / name]
        for name in installer.BIN_NAMES
    ] + [
        ["codesign", "--force", "--sign", "-", dest_dir / f"{name}.dylib"]
        for name in installer.SHARED_PLUGIN_NAMES
    ]


def test_managed_source_checkout_recovers_non_git_cache(monkeypatch, tmp_path):
    dest = tmp_path / "src" / "llama.cpp"
    dest.mkdir(parents=True)
    stale_file = dest / "partial-download.txt"
    stale_file.write_text("not a git checkout", encoding="utf-8")
    commands = []

    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._clone_or_checkout(
        "https://example.test/llama.cpp.git",
        "main",
        dest,
        dry_run=False,
        reset=True,
    )

    assert not stale_file.exists()
    assert commands[0] == ["git", "clone", "https://example.test/llama.cpp.git", dest]
    assert commands[1] == ["git", "checkout", "main"]
    assert commands[2] == ["git", "reset", "--hard", "main"]


def test_managed_source_checkout_recovers_symlinked_non_git_cache(monkeypatch, tmp_path):
    target = tmp_path / "real-source"
    target.mkdir()
    target_marker = target / "keep.txt"
    target_marker.write_text("keep", encoding="utf-8")
    dest = tmp_path / "src" / "llama.cpp"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(target, target_is_directory=True)
    commands = []

    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._clone_or_checkout(
        "https://example.test/llama.cpp.git",
        "main",
        dest,
        dry_run=False,
        reset=True,
    )

    assert not dest.exists()
    assert target_marker.read_text(encoding="utf-8") == "keep"
    assert commands[0] == ["git", "clone", "https://example.test/llama.cpp.git", dest]
    assert commands[1] == ["git", "checkout", "main"]
    assert commands[2] == ["git", "reset", "--hard", "main"]


def test_managed_source_checkout_dry_run_previews_reclone_for_non_git_cache(monkeypatch, tmp_path):
    dest = tmp_path / "src" / "llama.cpp"
    dest.mkdir(parents=True)
    stale_file = dest / "partial-download.txt"
    stale_file.write_text("not a git checkout", encoding="utf-8")
    commands = []

    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._clone_or_checkout(
        "https://example.test/llama.cpp.git",
        "main",
        dest,
        dry_run=True,
        reset=True,
    )

    assert stale_file.exists()
    assert commands[0] == ["git", "clone", "https://example.test/llama.cpp.git", dest]
    assert commands[1] == ["git", "checkout", "main"]
    assert commands[2] == ["git", "reset", "--hard", "main"]


def test_managed_source_checkout_recovers_file_at_cache_path(monkeypatch, tmp_path):
    dest = tmp_path / "src" / "llama.cpp"
    dest.parent.mkdir(parents=True)
    dest.write_text("not a directory", encoding="utf-8")
    commands = []

    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._clone_or_checkout(
        "https://example.test/llama.cpp.git",
        "main",
        dest,
        dry_run=False,
        reset=True,
    )

    assert not dest.exists()
    assert commands[0] == ["git", "clone", "https://example.test/llama.cpp.git", dest]
    assert commands[1] == ["git", "checkout", "main"]
    assert commands[2] == ["git", "reset", "--hard", "main"]


def test_managed_source_checkout_repairs_wrong_origin_url(monkeypatch, tmp_path):
    dest = tmp_path / "src" / "llama.cpp"
    dest.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=dest, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://wrong.example/llama.cpp.git"],
        cwd=dest,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = []

    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._clone_or_checkout(
        "https://example.test/llama.cpp.git",
        "refs/pull/24423/head",
        dest,
        dry_run=False,
        reset=True,
    )

    assert commands[0] == ["git", "remote", "set-url", "origin", "https://example.test/llama.cpp.git"]
    assert commands[1] == ["git", "fetch", "--all", "--tags"]
    assert commands[2] == ["git", "fetch", "origin", "refs/pull/24423/head"]
    assert commands[3] == ["git", "checkout", "FETCH_HEAD"]
    assert commands[4] == ["git", "reset", "--hard", "FETCH_HEAD"]


def test_default_llama_ref_is_pinned_for_reproducible_native_builds():
    assert installer.LLAMA_REF == "ef5e2dcce"
    assert not installer.LLAMA_REF.startswith("refs/")


def test_default_stable_diffusion_ref_is_pinned_for_reproducible_native_image_builds():
    assert installer.STABLE_DIFFUSION_REPO == "https://github.com/leejet/stable-diffusion.cpp.git"
    assert installer.STABLE_DIFFUSION_REF == "8caa3f908ae6d4a4bef531e73b9a969f266a3d1f"


def test_default_stable_diffusion_dir_uses_package_source_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("UTOPIC_HOME", str(tmp_path / "cache"))

    assert installer.default_stable_diffusion_dir() == tmp_path / "cache" / "src" / "stable-diffusion.cpp"


def test_default_stable_diffusion_dir_can_be_overridden(monkeypatch, tmp_path):
    source_dir = tmp_path / "external" / "stable-diffusion.cpp"
    monkeypatch.setenv("UTOPIC_STABLE_DIFFUSION_DIR", str(source_dir))

    assert installer.default_stable_diffusion_dir() == source_dir


def test_build_utopic_uses_package_owned_cmake_with_native_source_variable(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cmake_dir = tmp_path / "site-packages" / "utopic" / "cmake"
    native_dir = tmp_path / "site-packages" / "utopic" / "core" / "native"
    llama_dir = cache_root / "src" / "llama.cpp"
    build_dir = cache_root / "build" / "utopic"

    cmake_dir.mkdir(parents=True)
    native_dir.mkdir(parents=True)
    llama_dir.mkdir(parents=True)

    commands = []
    monkeypatch.setattr(installer, "build_root", lambda: cache_root / "build")
    monkeypatch.setattr(installer, "PACKAGED_CMAKE_DIR", cmake_dir)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_utopic(native_dir, llama_dir, jobs=None, dry_run=False)

    assert commands[0][:5] == ["cmake", "-B", build_dir, "-S", cmake_dir]
    assert f"-DUTOPIC_NATIVE_SOURCE_DIR={native_dir}" in commands[0]
    assert "-DUTOPIC_ENABLE_STABLE_DIFFUSION=OFF" in commands[0]


def test_build_utopic_enables_stable_diffusion_when_source_is_available(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cmake_dir = tmp_path / "site-packages" / "utopic" / "cmake"
    native_dir = tmp_path / "site-packages" / "utopic" / "core" / "native"
    llama_dir = cache_root / "src" / "llama.cpp"
    stable_diffusion_dir = cache_root / "src" / "stable-diffusion.cpp"
    build_dir = cache_root / "build" / "utopic"

    cmake_dir.mkdir(parents=True)
    native_dir.mkdir(parents=True)
    llama_dir.mkdir(parents=True)
    stable_diffusion_dir.mkdir(parents=True)

    commands = []
    monkeypatch.setattr(installer, "build_root", lambda: cache_root / "build")
    monkeypatch.setattr(installer, "PACKAGED_CMAKE_DIR", cmake_dir)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_utopic(
        native_dir,
        llama_dir,
        stable_diffusion_dir=stable_diffusion_dir,
        jobs=None,
        dry_run=False,
    )

    assert commands[0][:5] == ["cmake", "-B", build_dir, "-S", cmake_dir]
    assert f"-DUTOPIC_STABLE_DIFFUSION_DIR={stable_diffusion_dir}" in commands[0]
    assert "-DUTOPIC_ENABLE_STABLE_DIFFUSION=ON" in commands[0]


def test_build_utopic_enables_sherpa_onnx_when_source_is_available(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cmake_dir = tmp_path / "site-packages" / "utopic" / "cmake"
    native_dir = tmp_path / "site-packages" / "utopic" / "core" / "native"
    llama_dir = cache_root / "src" / "llama.cpp"
    sherpa_onnx_dir = cache_root / "src" / "sherpa-onnx"
    build_dir = cache_root / "build" / "utopic"

    cmake_dir.mkdir(parents=True)
    native_dir.mkdir(parents=True)
    llama_dir.mkdir(parents=True)
    sherpa_onnx_dir.mkdir(parents=True)

    commands = []
    monkeypatch.setattr(installer, "build_root", lambda: cache_root / "build")
    monkeypatch.setattr(installer, "PACKAGED_CMAKE_DIR", cmake_dir)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_utopic(
        native_dir,
        llama_dir,
        sherpa_onnx_dir=sherpa_onnx_dir,
        jobs=None,
        dry_run=False,
    )

    assert commands[0][:5] == ["cmake", "-B", build_dir, "-S", cmake_dir]
    assert f"-DUTOPIC_SHERPA_ONNX_DIR={sherpa_onnx_dir}" in commands[0]
    assert "-DUTOPIC_ENABLE_SHERPA_ONNX=ON" in commands[0]


def test_build_utopic_clears_stale_cmake_cache_when_package_cmake_source_changes(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    old_cmake = cache_root / "old-package" / "cmake"
    new_cmake = tmp_path / "site-packages" / "utopic" / "cmake"
    native_dir = tmp_path / "site-packages" / "utopic" / "core" / "native"
    llama_dir = cache_root / "src" / "llama.cpp"
    build_dir = cache_root / "build" / "utopic"

    old_cmake.mkdir(parents=True)
    new_cmake.mkdir(parents=True)
    native_dir.mkdir(parents=True)
    llama_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={old_cmake}\n",
        encoding="utf-8",
    )
    stale_marker = build_dir / "stale-object.o"
    stale_marker.write_text("old build output", encoding="utf-8")

    commands = []
    monkeypatch.setattr(installer, "build_root", lambda: cache_root / "build")
    monkeypatch.setattr(installer, "PACKAGED_CMAKE_DIR", new_cmake)
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_utopic(native_dir, llama_dir, jobs=None, dry_run=False)

    assert not stale_marker.exists()
    assert commands[0][:5] == ["cmake", "-B", build_dir, "-S", new_cmake]


def test_prepare_cmake_build_dir_unlinks_stale_symlinked_cache_without_touching_target(tmp_path):
    old_source = tmp_path / "old-source"
    new_source = tmp_path / "new-source"
    target = tmp_path / "real-build-cache"
    build_dir = tmp_path / "build" / "utopic"

    old_source.mkdir()
    new_source.mkdir()
    target.mkdir()
    (target / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={old_source}\n",
        encoding="utf-8",
    )
    target_marker = target / "keep.txt"
    target_marker.write_text("keep", encoding="utf-8")
    build_dir.parent.mkdir()
    build_dir.symlink_to(target, target_is_directory=True)

    installer._prepare_cmake_build_dir(build_dir, new_source, dry_run=False)

    assert not build_dir.exists()
    assert target_marker.read_text(encoding="utf-8") == "keep"


def test_native_installation_is_not_current_without_metadata(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_is_not_current_when_auto_best_backend_changes(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    installed_decision = installer.BackendDecision(
        backend="cpu",
        reason="old",
        device="CPU",
    )
    new_decision = installer.BackendDecision(
        backend="cuda",
        reason="new",
        device="CUDA arch 80",
        cuda_architectures="80",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        installed_decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: new_decision)

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_keeps_auto_metal_cache_without_reprobing(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    decision = installer.BackendDecision(
        backend="metal",
        reason="old",
        device="Apple M4 Pro",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: pytest.fail("should not reprobe metal cache"))

    assert installer.native_installation_is_current(("utopic_server",)) is True


def test_native_installation_keeps_auto_cuda_cache_without_requiring_compiler(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    decision = installer.BackendDecision(
        backend="cuda",
        reason="old",
        device="CUDA arch 80",
        cuda_architectures="80",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: pytest.fail("should not require CUDA build tools for a CUDA runtime cache"))

    assert installer.native_installation_is_current(("utopic_server",)) is True


def test_native_installation_is_not_current_when_explicit_backend_changes(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    old_decision = installer.BackendDecision(
        backend="metal",
        reason="old",
        device="Apple M4 Pro",
    )

    monkeypatch.setenv("UTOPIC_BACKEND", "cpu")
    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        old_decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_is_not_current_when_cuda_architecture_override_changes(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    old_decision = installer.BackendDecision(
        backend="cuda",
        reason="old",
        device="CUDA arch 89",
        cuda_architectures="89",
    )

    monkeypatch.setenv("UTOPIC_CUDA_ARCHITECTURES", "120")
    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        old_decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_is_not_current_when_cuda_graphs_override_changes(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    old_decision = installer.BackendDecision(
        backend="cuda",
        reason="old",
        device="CUDA arch 121",
        cuda_architectures="121",
        cuda_graphs="OFF",
    )

    monkeypatch.setenv("UTOPIC_CUDA_GRAPHS", "ON")
    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        old_decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_is_current_when_metadata_matches(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    decision = installer.BackendDecision(
        backend="cpu",
        reason="No usable Metal device or CUDA compiler found",
        device="CPU",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    installer._write_install_metadata(
        decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is True


def test_native_installation_is_not_current_when_cached_binary_is_not_executable(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")
    decision = installer.BackendDecision(
        backend="cpu",
        reason="No usable Metal device or CUDA compiler found",
        device="CPU",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    installer._write_install_metadata(
        decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_accepts_different_request_that_resolves_to_same_backend(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(bin_dir / "utopic_server")
    decision = installer.BackendDecision(
        backend="cpu",
        reason="No usable Metal device or CUDA compiler found",
        device="CPU",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "default_llama_dir", lambda: tmp_path / "src" / "llama.cpp")
    monkeypatch.setattr(installer, "default_native_dir", lambda: tmp_path / "site" / "utopic" / "native")
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    installer._write_install_metadata(
        decision,
        requested_backend="cpu",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is True


def test_setup_writes_install_metadata_after_success(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    build_dir = tmp_path / "build" / "utopic"
    llama_dir = tmp_path / "src" / "llama.cpp"
    stable_diffusion_dir = tmp_path / "src" / "stable-diffusion.cpp"
    native_dir = tmp_path / "site" / "utopic" / "native"
    decision = installer.BackendDecision(
        backend="cpu",
        reason="Requested by --backend cpu",
        device="CPU",
    )

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    monkeypatch.setattr(installer, "_print_backend_decision", lambda decision, requested: None)
    monkeypatch.setattr(installer, "_verify_llama_apis", lambda llama_dir: None)
    monkeypatch.setattr(installer, "_build_llama", lambda *args, **kwargs: None)
    monkeypatch.setattr(installer, "_build_utopic", lambda *args, **kwargs: build_dir)

    def install_binaries(build_dir_arg):
        assert build_dir_arg == build_dir
        bin_dir.mkdir(parents=True)
        for name in installer.BIN_NAMES:
            _write_executable(bin_dir / name)

    monkeypatch.setattr(installer, "_install_binaries", install_binaries)

    assert installer.setup(
        [
            "--backend",
            "cpu",
            "--llama-dir",
            str(llama_dir),
            "--native-dir",
            str(native_dir),
            "--stable-diffusion-dir",
            str(stable_diffusion_dir),
        ]
    ) == 0

    metadata = installer._read_install_metadata()
    assert metadata is not None
    assert metadata["backend"] == "cpu"
    assert metadata["requested_backend"] == "cpu"
    assert metadata["llama_dir"] == str(installer._normalize_path(llama_dir))
    assert metadata["native_dir"] == str(installer._normalize_path(native_dir))
    assert metadata["stable_diffusion_dir"] == str(installer._normalize_path(stable_diffusion_dir))


def test_setup_manages_stable_diffusion_source_for_native_image_engine(monkeypatch, tmp_path):
    llama_dir = tmp_path / "src" / "llama.cpp"
    stable_diffusion_dir = tmp_path / "src" / "stable-diffusion.cpp"
    native_dir = tmp_path / "site" / "utopic" / "native"
    build_dir = tmp_path / "build" / "utopic"
    decision = installer.BackendDecision(
        backend="cpu",
        reason="Requested by --backend cpu",
        device="CPU",
    )
    cloned = []
    observed_stable_diffusion_dirs = []

    monkeypatch.setattr(installer, "default_llama_dir", lambda: llama_dir)
    monkeypatch.setattr(installer, "default_stable_diffusion_dir", lambda: stable_diffusion_dir)
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    monkeypatch.setattr(installer, "_print_backend_decision", lambda decision, requested: None)
    monkeypatch.setattr(installer, "_verify_llama_apis", lambda llama_dir: None)
    monkeypatch.setattr(installer, "_build_llama", lambda *args, **kwargs: None)
    monkeypatch.setattr(installer, "_install_binaries", lambda build_dir: None)
    monkeypatch.setattr(installer, "_write_install_metadata", lambda *args, **kwargs: None)

    def clone_or_checkout(repo, ref, dest, **kwargs):
        cloned.append((repo, ref, dest))

    def build_utopic(native_dir_arg, llama_dir_arg, **kwargs):
        observed_stable_diffusion_dirs.append(kwargs["stable_diffusion_dir"])
        return build_dir

    monkeypatch.setattr(installer, "_clone_or_checkout", clone_or_checkout)
    monkeypatch.setattr(installer, "_build_utopic", build_utopic)

    assert installer.setup(
        [
            "--backend",
            "cpu",
            "--native-dir",
            str(native_dir),
        ]
    ) == 0

    assert (
        installer.STABLE_DIFFUSION_REPO,
        installer.STABLE_DIFFUSION_REF,
        stable_diffusion_dir,
    ) in cloned
    assert observed_stable_diffusion_dirs == [stable_diffusion_dir]


def test_setup_jobs_argument_overrides_invalid_environment(monkeypatch, tmp_path):
    llama_dir = tmp_path / "src" / "llama.cpp"
    native_dir = tmp_path / "site" / "utopic" / "native"
    observed_jobs = []
    decision = installer.BackendDecision(
        backend="cpu",
        reason="Requested by --backend cpu",
        device="CPU",
    )

    monkeypatch.setenv("UTOPIC_BUILD_JOBS", "fast")
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    monkeypatch.setattr(installer, "_print_backend_decision", lambda decision, requested: None)
    monkeypatch.setattr(
        installer,
        "_build_llama",
        lambda *args, **kwargs: observed_jobs.append(("llama", kwargs["jobs"])),
    )
    monkeypatch.setattr(
        installer,
        "_build_utopic",
        lambda *args, **kwargs: observed_jobs.append(("utopic", kwargs["jobs"])) or tmp_path / "build" / "utopic",
    )

    assert installer.setup(
        [
            "--dry-run",
            "--backend",
            "cpu",
            "--jobs",
            "2",
            "--llama-dir",
            str(llama_dir),
            "--native-dir",
            str(native_dir),
        ]
    ) == 0

    assert observed_jobs == [("llama", 2), ("utopic", 2)]


def test_setup_rejects_invalid_jobs_environment_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("UTOPIC_BUILD_JOBS", "fast")

    with pytest.raises(SystemExit) as exc_info:
        installer.setup(["--dry-run"])

    assert exc_info.value.code == 2
    assert "UTOPIC_BUILD_JOBS must be a positive integer" in capsys.readouterr().err


def test_setup_rejects_invalid_cuda_graphs_environment_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("UTOPIC_CUDA_GRAPHS", "sometimes")
    monkeypatch.setattr(installer, "_detect_cuda_architectures", lambda: "121")

    with pytest.raises(SystemExit) as exc_info:
        installer.setup(["--dry-run", "--backend", "cuda"])

    assert exc_info.value.code == 2
    assert "UTOPIC_CUDA_GRAPHS must be one of" in capsys.readouterr().err


def test_setup_force_clears_stale_build_cache_before_rebuild(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    build_root = tmp_path / "build"
    llama_dir = tmp_path / "src" / "llama.cpp"
    stale_build_file = build_root / "utopic" / "stale-object.o"
    stale_llama_file = llama_dir / "build" / "stale-object.o"
    stable_diffusion_dir = tmp_path / "src" / "stable-diffusion.cpp"
    native_dir = tmp_path / "site" / "utopic" / "native"
    decision = installer.BackendDecision(
        backend="cpu",
        reason="Requested by --backend cpu",
        device="CPU",
    )
    observed = []

    bin_dir.mkdir(parents=True)
    (bin_dir / "utopic_server").write_text("stale binary", encoding="utf-8")
    stale_build_file.parent.mkdir(parents=True)
    stale_build_file.write_text("stale utopic build", encoding="utf-8")
    stale_llama_file.parent.mkdir(parents=True)
    stale_llama_file.write_text("stale llama build", encoding="utf-8")

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)
    monkeypatch.setattr(installer, "build_root", lambda: build_root)
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    monkeypatch.setattr(installer, "_print_backend_decision", lambda decision, requested: None)
    monkeypatch.setattr(installer, "_verify_llama_apis", lambda llama_dir: None)
    monkeypatch.setattr(
        installer,
        "_build_llama",
        lambda *args, **kwargs: observed.append(
            ("llama", stale_llama_file.exists(), stale_build_file.exists())
        ),
    )
    monkeypatch.setattr(
        installer,
        "_build_utopic",
        lambda *args, **kwargs: observed.append(
            ("utopic", stale_llama_file.exists(), stale_build_file.exists())
        )
        or tmp_path / "build-output",
    )
    monkeypatch.setattr(installer, "_install_binaries", lambda build_dir: None)
    monkeypatch.setattr(installer, "_write_install_metadata", lambda *args, **kwargs: None)

    assert installer.setup(
        [
            "--force",
            "--backend",
            "cpu",
            "--llama-dir",
            str(llama_dir),
            "--native-dir",
            str(native_dir),
            "--stable-diffusion-dir",
            str(stable_diffusion_dir),
        ]
    ) == 0

    assert observed == [("llama", False, False), ("utopic", False, False)]


def test_remove_path_unlinks_symlinked_directory_without_touching_target(tmp_path):
    target = tmp_path / "real-cache"
    target.mkdir()
    target_marker = target / "keep.txt"
    target_marker.write_text("keep", encoding="utf-8")
    link = tmp_path / "cache-link"
    link.symlink_to(target, target_is_directory=True)

    installer._remove_path(link, dry_run=False)

    assert not link.exists()
    assert target_marker.read_text(encoding="utf-8") == "keep"


def test_setup_help_describes_force_clean_rebuild(capsys):
    with pytest.raises(SystemExit) as exc_info:
        installer.setup(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "Remove cached binaries and build directories" in help_text
    assert "rebuilding." in help_text
