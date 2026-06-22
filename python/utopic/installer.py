import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from . import __version__


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGED_NATIVE_DIR = PACKAGE_DIR / "native"
UTOPIC_NATIVE_REPO = "https://github.com/adavyas/utopic.git"
UTOPIC_NATIVE_REF = "92ca14f12fe45f78d605511bc4e7e21c3ed9bebd"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_REF = "refs/pull/24423/head"
BIN_NAMES = ("utopic", "utopic_server", "utopic_mcp", "utopic_acp")
INSTALL_METADATA_NAME = "install.json"
INSTALL_METADATA_SCHEMA_VERSION = 1
INSTALL_METADATA_MATCH_KEYS = (
    "schema_version",
    "package_version",
    "llama_repo",
    "llama_ref",
    "native_repo",
    "native_ref",
    "llama_dir",
    "native_dir",
    "system",
    "machine",
)
REQUIRED_LLAMA_SYMBOLS = (
    "llama_diffusion_set_sc",
    "llama_diffusion_device_sample",
    "llama_diffusion_set_phase",
)
LLAMA_CMAKE_FLAGS = (
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_BUILD_APP=OFF",
)
BACKENDS = ("auto", "cpu", "cuda", "metal")


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    reason: str
    device: str
    cuda_architectures: Optional[str] = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _environment_build_jobs(parser: argparse.ArgumentParser) -> Optional[int]:
    value = os.environ.get("UTOPIC_BUILD_JOBS")
    if not value:
        return None
    try:
        return _positive_int(value)
    except (argparse.ArgumentTypeError, ValueError):
        parser.error("UTOPIC_BUILD_JOBS must be a positive integer")
    return None


def cache_root() -> Path:
    configured = os.environ.get("UTOPIC_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "utopic"


def source_root() -> Path:
    return cache_root() / "src"


def build_root() -> Path:
    return cache_root() / "build"


def bin_dir() -> Path:
    configured = os.environ.get("UTOPIC_BIN_DIR")
    if configured:
        return Path(configured).expanduser()
    return cache_root() / "bin"


def install_metadata_path() -> Path:
    return bin_dir() / INSTALL_METADATA_NAME


def default_native_dir() -> Path:
    configured = os.environ.get("UTOPIC_NATIVE_DIR")
    if configured:
        return Path(configured).expanduser()
    if PACKAGED_NATIVE_DIR.exists():
        return PACKAGED_NATIVE_DIR
    return source_root() / "Utopic"


def default_llama_dir() -> Path:
    configured = os.environ.get("UTOPIC_LLAMACPP_DIR")
    if configured:
        return Path(configured).expanduser()
    return source_root() / "llama.cpp"


def _run(
    command: Sequence[object],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"+ {printable}")
    if dry_run:
        return
    subprocess.run(
        [str(part) for part in command],
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        check=True,
    )


def _clone_or_checkout(repo: str, ref: str, dest: Path, *, dry_run: bool, reset: bool = False) -> None:
    dest_exists = dest.exists()
    if dest.exists() and not (dest / ".git").exists():
        print(f"+ remove invalid source checkout {dest}")
        if not dry_run:
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest_exists = False

    if dest_exists:
        _run(["git", "fetch", "--all", "--tags"], cwd=dest, dry_run=dry_run)
    else:
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo, dest], dry_run=dry_run)

    checkout_ref = ref
    if ref.startswith("refs/"):
        _run(["git", "fetch", "origin", ref], cwd=dest, dry_run=dry_run)
        checkout_ref = "FETCH_HEAD"

    _run(["git", "checkout", checkout_ref], cwd=dest, dry_run=dry_run)
    if reset:
        _run(["git", "reset", "--hard", checkout_ref], cwd=dest, dry_run=dry_run)


def _cuda_compiler_candidates(cuda_architectures: Optional[str] = None) -> list[Path]:
    candidates: list[Path] = []

    def append(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    configured = os.environ.get("CUDACXX")
    if configured:
        append(Path(configured).expanduser())

    arch_parts = (cuda_architectures or "").replace(",", ";").split(";")
    if any(part.strip().startswith("12") for part in arch_parts):
        append(Path("/usr/local/cuda-13.0/bin/nvcc"))
        append(Path("/usr/local/cuda-13/bin/nvcc"))

    found = shutil.which("nvcc")
    if found:
        append(Path(found))

    for candidate in (
        Path("/usr/local/cuda/bin/nvcc"),
        Path("/usr/local/cuda-12.4/bin/nvcc"),
        Path("/usr/local/cuda-12.3/bin/nvcc"),
        Path("/usr/local/cuda-12.2/bin/nvcc"),
        Path("/usr/local/cuda-12.1/bin/nvcc"),
        Path("/usr/local/cuda-12.0/bin/nvcc"),
    ):
        append(candidate)
    return candidates


def _find_cuda_compiler(cuda_architectures: Optional[str] = None) -> Optional[Path]:
    for candidate in _cuda_compiler_candidates(cuda_architectures):
        if candidate.exists():
            return candidate
    return None


def _detect_cuda_architectures() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    arches: list[str] = []
    for line in completed.stdout.splitlines():
        cap = line.strip()
        if not cap:
            continue
        arch = cap.replace(".", "")
        if arch and arch not in arches:
            arches.append(arch)
    return ";".join(arches) if arches else None


def _detect_metal_device() -> Optional[str]:
    if platform.system() != "Darwin":
        return None

    compiler = shutil.which("clang++") or shutil.which("clang")
    if compiler is None:
        return None

    source = """
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("device=null\\n");
            return 2;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            printf("device=%s\\nqueue=null\\n", [[device name] UTF8String]);
            return 3;
        }
        printf("device=%s\\nqueue=ok\\n", [[device name] UTF8String]);
        return 0;
    }
}
""".lstrip()

    with tempfile.TemporaryDirectory(prefix="utopic-metal-probe-") as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "metal_probe.mm"
        exe = tmp_dir / "metal_probe"
        src.write_text(source, encoding="utf-8")
        compile_result = subprocess.run(
            [compiler, "-ObjC++", src, "-framework", "Foundation", "-framework", "Metal", "-o", exe],
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            return None

        run_result = subprocess.run([exe], capture_output=True, text=True)
        if run_result.returncode != 0:
            return None

    for line in run_result.stdout.splitlines():
        if line.startswith("device="):
            device = line.removeprefix("device=").strip()
            if device and device != "null":
                return device
    return None


def _resolve_backend(requested_backend: str, cuda_architectures: Optional[str]) -> BackendDecision:
    if requested_backend == "auto":
        metal_device = _detect_metal_device()
        if metal_device:
            return BackendDecision(
                backend="metal",
                reason="Metal device available",
                device=metal_device,
            )

        detected_cuda_architectures = cuda_architectures or _detect_cuda_architectures()
        cuda_compiler = _find_cuda_compiler(detected_cuda_architectures)
        if cuda_compiler:
            device = (
                f"CUDA arch {detected_cuda_architectures}"
                if detected_cuda_architectures
                else "NVIDIA CUDA"
            )
            return BackendDecision(
                backend="cuda",
                reason="NVIDIA CUDA compiler available",
                device=device,
                cuda_architectures=detected_cuda_architectures,
            )

        return BackendDecision(
            backend="cpu",
            reason="No usable Metal device or CUDA compiler found",
            device="CPU",
        )

    if requested_backend == "cuda":
        detected_cuda_architectures = cuda_architectures or _detect_cuda_architectures()
        device = (
            f"CUDA arch {detected_cuda_architectures}"
            if detected_cuda_architectures
            else "NVIDIA CUDA"
        )
        return BackendDecision(
            backend="cuda",
            reason="Requested by --backend cuda",
            device=device,
            cuda_architectures=detected_cuda_architectures,
        )

    if requested_backend == "metal":
        return BackendDecision(
            backend="metal",
            reason="Requested by --backend metal",
            device=_detect_metal_device() or "Metal",
        )

    return BackendDecision(
        backend="cpu",
        reason="Requested by --backend cpu",
        device="CPU",
    )


def _print_backend_decision(decision: BackendDecision, requested_backend: str) -> None:
    label = "Detected" if requested_backend == "auto" else "Selected"
    print(f"{label} backend: {decision.backend}")
    print(f"Device: {decision.device}")
    print(f"Reason: {decision.reason}")
    if decision.cuda_architectures:
        print(f"CUDA architectures: {decision.cuda_architectures}")


def _install_metadata(
    decision: BackendDecision,
    *,
    requested_backend: str,
    llama_dir: Path,
    native_dir: Path,
) -> dict[str, object]:
    return {
        "schema_version": INSTALL_METADATA_SCHEMA_VERSION,
        "package_version": __version__,
        "requested_backend": requested_backend,
        "backend": decision.backend,
        "cuda_architectures": decision.cuda_architectures,
        "llama_repo": os.environ.get("UTOPIC_LLAMA_REPO", LLAMA_REPO),
        "llama_ref": os.environ.get("UTOPIC_LLAMA_REF", LLAMA_REF),
        "native_repo": os.environ.get("UTOPIC_NATIVE_REPO", UTOPIC_NATIVE_REPO),
        "native_ref": os.environ.get("UTOPIC_NATIVE_REF", UTOPIC_NATIVE_REF),
        "llama_dir": str(_normalize_path(llama_dir)),
        "native_dir": str(_normalize_path(native_dir)),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _read_install_metadata() -> Optional[dict[str, object]]:
    path = install_metadata_path()
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _write_install_metadata(
    decision: BackendDecision,
    *,
    requested_backend: str,
    llama_dir: Path,
    native_dir: Path,
) -> None:
    path = install_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _install_metadata(
        decision,
        requested_backend=requested_backend,
        llama_dir=llama_dir,
        native_dir=native_dir,
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _binary_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def native_installation_is_current(binary_names: Sequence[str] = BIN_NAMES) -> bool:
    suffix = _binary_suffix()
    for name in binary_names:
        binary = bin_dir() / f"{name}{suffix}"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            return False

    metadata = _read_install_metadata()
    if metadata is None:
        return False

    requested_backend = os.environ.get("UTOPIC_BACKEND", "auto")
    cuda_architectures = os.environ.get("UTOPIC_CUDA_ARCHITECTURES")
    if requested_backend != "auto" and metadata.get("backend") != requested_backend:
        return False

    metadata_backend = metadata.get("backend")
    if not isinstance(metadata_backend, str):
        return False
    metadata_cuda_architectures = metadata.get("cuda_architectures")
    if metadata_cuda_architectures is not None and not isinstance(metadata_cuda_architectures, str):
        return False
    if metadata_backend == "cuda" and cuda_architectures:
        if metadata_cuda_architectures != cuda_architectures:
            return False

    expected = _install_metadata(
        BackendDecision(
            backend=metadata_backend,
            reason="installed metadata",
            device="installed metadata",
            cuda_architectures=metadata_cuda_architectures,
        ),
        requested_backend=str(metadata.get("requested_backend", "auto")),
        llama_dir=default_llama_dir(),
        native_dir=default_native_dir(),
    )
    return all(metadata.get(key) == expected.get(key) for key in INSTALL_METADATA_MATCH_KEYS)


def _build_command(build_dir: Path, *, jobs: Optional[int]) -> list[object]:
    command: list[object] = ["cmake", "--build", build_dir, "-j"]
    if jobs is not None:
        command.append(str(jobs))
    return command


def _build_llama(
    llama_dir: Path,
    *,
    backend: str,
    cuda_architectures: Optional[str],
    jobs: Optional[int],
    dry_run: bool,
) -> None:
    command = ["cmake", "-B", llama_dir / "build", "-S", llama_dir, *LLAMA_CMAKE_FLAGS]
    if backend == "cpu":
        command.extend(["-DGGML_CUDA=OFF", "-DGGML_METAL=OFF"])
    elif backend == "cuda":
        command.append("-DGGML_CUDA=ON")
        if cuda_architectures is None:
            cuda_architectures = _detect_cuda_architectures()
        cuda_compiler = _find_cuda_compiler(cuda_architectures)
        if cuda_compiler:
            command.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
        if cuda_architectures:
            command.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
    elif backend == "metal":
        command.extend(["-DGGML_METAL=ON", "-DGGML_CUDA=OFF"])
    _run(command, dry_run=dry_run)
    _run(_build_command(llama_dir / "build", jobs=jobs), dry_run=dry_run)


def _verify_llama_apis(llama_dir: Path) -> None:
    header = llama_dir / "include" / "llama.h"
    if not header.exists():
        raise RuntimeError(
            f"Utopic native dependency header was not found: {header}. "
            "Run `utopic setup --force` to refresh the package-managed sources."
        )
    text = header.read_text(encoding="utf-8")
    missing = [symbol for symbol in REQUIRED_LLAMA_SYMBOLS if symbol not in text]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "The Utopic native dependency is missing required diffusion APIs: "
            f"{names}. Run `utopic setup --force` to refresh the package-managed sources."
        )


def _native_cmake_source(native_dir: Path) -> Path:
    if (native_dir / "CMakeLists.txt").exists():
        return native_dir
    return native_dir / "native"


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _cached_cmake_source(build_dir: Path) -> Optional[Path]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            return _normalize_path(Path(line.split("=", 1)[1]))
    return None


def _prepare_cmake_build_dir(build_dir: Path, source_dir: Path, *, dry_run: bool) -> None:
    cached_source = _cached_cmake_source(build_dir)
    if cached_source is None or cached_source == _normalize_path(source_dir):
        return
    print(f"+ remove stale CMake build directory {build_dir}")
    if not dry_run:
        shutil.rmtree(build_dir)


def _remove_path(path: Path, *, dry_run: bool) -> None:
    print(f"+ remove {path}")
    if dry_run:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _build_utopic(native_dir: Path, llama_dir: Path, *, jobs: Optional[int], dry_run: bool) -> Path:
    out_dir = build_root() / "utopic"
    source_dir = _native_cmake_source(native_dir)
    _prepare_cmake_build_dir(out_dir, source_dir, dry_run=dry_run)

    _run(
        ["cmake", "-B", out_dir, "-S", source_dir, f"-DUTOPIC_LLAMACPP_DIR={llama_dir}"],
        dry_run=dry_run,
    )
    _run(_build_command(out_dir, jobs=jobs), dry_run=dry_run)
    return out_dir


def _install_binaries(build_dir: Path) -> None:
    dest_dir = bin_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""

    for name in BIN_NAMES:
        src = build_dir / f"{name}{suffix}"
        if not src.exists():
            raise RuntimeError(f"Expected Utopic build output was not found: {src}")
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        dest.chmod(0o755)


def setup(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="utopic setup",
        description="Build and cache Utopic from package-managed native sources.",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=os.environ.get("UTOPIC_BACKEND", "auto"),
        help="Native acceleration backend to build. Use cuda on NVIDIA hosts.",
    )
    parser.add_argument("--cuda", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cuda-architectures",
        default=os.environ.get("UTOPIC_CUDA_ARCHITECTURES"),
        help="CUDA architecture list for the Utopic native build, for example 89 on RTX 4090 hosts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove cached binaries and build directories before rebuilding.",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=None,
        help="Limit native build parallelism when disk or temporary space is constrained.",
    )
    parser.add_argument("--llama-dir", help=argparse.SUPPRESS)
    parser.add_argument("--native-dir", help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-llama-build",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.jobs is None:
        args.jobs = _environment_build_jobs(parser)

    dry_run = bool(args.dry_run)
    requested_backend = "cuda" if args.cuda else args.backend
    backend_decision = _resolve_backend(requested_backend, args.cuda_architectures)
    _print_backend_decision(backend_decision, requested_backend)
    llama_dir = Path(args.llama_dir).expanduser() if args.llama_dir else default_llama_dir()
    native_dir = Path(args.native_dir).expanduser() if args.native_dir else default_native_dir()

    if args.force:
        for cache_path in (bin_dir(), build_root(), llama_dir / "build"):
            if cache_path.exists():
                _remove_path(cache_path, dry_run=dry_run)

    if args.llama_dir or os.environ.get("UTOPIC_LLAMACPP_DIR"):
        print(f"Using maintainer-provided native dependency source at {llama_dir}")
    else:
        print(f"Managing native dependency source at {llama_dir}")
        _clone_or_checkout(
            os.environ.get("UTOPIC_LLAMA_REPO", LLAMA_REPO),
            os.environ.get("UTOPIC_LLAMA_REF", LLAMA_REF),
            llama_dir,
            dry_run=dry_run,
            reset=True,
        )

    if not dry_run:
        _verify_llama_apis(llama_dir)

    if not args.skip_llama_build:
        _build_llama(
            llama_dir,
            backend=backend_decision.backend,
            cuda_architectures=backend_decision.cuda_architectures,
            jobs=args.jobs,
            dry_run=dry_run,
        )

    if args.native_dir or os.environ.get("UTOPIC_NATIVE_DIR"):
        print(f"Using external Utopic source at {native_dir}")
    elif native_dir == PACKAGED_NATIVE_DIR:
        print(f"Using packaged Utopic source at {native_dir}")
    else:
        print(f"Managing Utopic source at {native_dir}")
        _clone_or_checkout(
            os.environ.get("UTOPIC_NATIVE_REPO", UTOPIC_NATIVE_REPO),
            os.environ.get("UTOPIC_NATIVE_REF", UTOPIC_NATIVE_REF),
            native_dir,
            dry_run=dry_run,
        )

    native_build_dir = _build_utopic(native_dir, llama_dir, jobs=args.jobs, dry_run=dry_run)
    if dry_run:
        print(f"Would install Utopic native binaries to {bin_dir()}")
        return 0

    _install_binaries(native_build_dir)
    _write_install_metadata(
        backend_decision,
        requested_backend=requested_backend,
        llama_dir=llama_dir,
        native_dir=native_dir,
    )
    print(f"Installed Utopic native binaries to {bin_dir()}")
    return 0
