from pathlib import Path

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
