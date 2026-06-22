from pathlib import Path

import pytest

from utopic import installer


def test_auto_backend_prefers_metal_when_available(monkeypatch):
    monkeypatch.setattr(installer, "_detect_metal_device", lambda: "Apple M4 Pro")
    monkeypatch.setattr(installer, "_detect_cuda_architectures", lambda: "80")
    monkeypatch.setattr(installer, "_find_cuda_compiler", lambda cuda_architectures=None: Path("/usr/local/cuda/bin/nvcc"))

    decision = installer._resolve_backend("auto", None)

    assert decision.backend == "metal"
    assert decision.device == "Apple M4 Pro"
    assert decision.reason == "Metal device available"


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


def test_build_utopic_clears_stale_cmake_cache_when_source_changes(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    old_source = cache_root / "src" / "Utopic" / "native"
    new_source = tmp_path / "site-packages" / "utopic" / "native"
    llama_dir = cache_root / "src" / "llama.cpp"
    build_dir = cache_root / "build" / "utopic"

    old_source.mkdir(parents=True)
    new_source.mkdir(parents=True)
    llama_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    (new_source / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={old_source}\n",
        encoding="utf-8",
    )
    stale_marker = build_dir / "stale-object.o"
    stale_marker.write_text("old build output", encoding="utf-8")

    commands = []
    monkeypatch.setattr(installer, "build_root", lambda: cache_root / "build")
    monkeypatch.setattr(installer, "_run", lambda command, **kwargs: commands.append(command))

    installer._build_utopic(new_source, llama_dir, jobs=None, dry_run=False)

    assert not stale_marker.exists()
    assert commands[0][:5] == ["cmake", "-B", build_dir, "-S", new_source]


def test_native_installation_is_not_current_without_metadata(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")

    monkeypatch.setattr(installer, "bin_dir", lambda: bin_dir)

    assert installer.native_installation_is_current(("utopic_server",)) is False


def test_native_installation_is_current_when_auto_probe_changes(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")
    installed_decision = installer.BackendDecision(
        backend="metal",
        reason="old",
        device="Apple M4 Pro",
    )
    new_decision = installer.BackendDecision(
        backend="cpu",
        reason="new",
        device="CPU",
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

    assert installer.native_installation_is_current(("utopic_server",)) is True


def test_native_installation_is_not_current_when_explicit_backend_changes(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")
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
    (bin_dir / "utopic_server").write_text("binary", encoding="utf-8")
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


def test_native_installation_is_current_when_metadata_matches(monkeypatch, tmp_path):
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
    monkeypatch.setattr(installer, "_resolve_backend", lambda requested, arch: decision)
    installer._write_install_metadata(
        decision,
        requested_backend="auto",
        llama_dir=installer.default_llama_dir(),
        native_dir=installer.default_native_dir(),
    )

    assert installer.native_installation_is_current(("utopic_server",)) is True


def test_native_installation_accepts_different_request_that_resolves_to_same_backend(monkeypatch, tmp_path):
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
            (bin_dir / name).write_text("binary", encoding="utf-8")

    monkeypatch.setattr(installer, "_install_binaries", install_binaries)

    assert installer.setup(
        [
            "--backend",
            "cpu",
            "--llama-dir",
            str(llama_dir),
            "--native-dir",
            str(native_dir),
        ]
    ) == 0

    metadata = installer._read_install_metadata()
    assert metadata is not None
    assert metadata["backend"] == "cpu"
    assert metadata["requested_backend"] == "cpu"
    assert metadata["llama_dir"] == str(installer._normalize_path(llama_dir))
    assert metadata["native_dir"] == str(installer._normalize_path(native_dir))


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


def test_setup_force_clears_stale_build_cache_before_rebuild(monkeypatch, tmp_path):
    bin_dir = tmp_path / "bin"
    build_root = tmp_path / "build"
    llama_dir = tmp_path / "src" / "llama.cpp"
    stale_build_file = build_root / "utopic" / "stale-object.o"
    stale_llama_file = llama_dir / "build" / "stale-object.o"
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
        ]
    ) == 0

    assert observed == [("llama", False, False), ("utopic", False, False)]


def test_setup_help_describes_force_clean_rebuild(capsys):
    with pytest.raises(SystemExit) as exc_info:
        installer.setup(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "Remove cached binaries and build directories" in help_text
    assert "rebuilding." in help_text
