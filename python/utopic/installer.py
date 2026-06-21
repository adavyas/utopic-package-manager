import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
UTOPIC_NATIVE_REPO = "https://github.com/adavyas/Utopic.git"
UTOPIC_NATIVE_REF = "dad769dd687feda089d7cd36d780cd9e6c979a3a"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_REF = "9b4dae81f48b96765b6e24539c229c6ec304fc6c"
BIN_NAMES = ("utopic", "utopic_server", "utopic_mcp", "utopic_acp")
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
BACKENDS = ("auto", "cpu", "cuda")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


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


def default_native_dir() -> Path:
    configured = os.environ.get("UTOPIC_NATIVE_DIR")
    if configured:
        return Path(configured).expanduser()
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
    if dest.exists():
        _run(["git", "fetch", "--all", "--tags"], cwd=dest, dry_run=dry_run)
    else:
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo, dest], dry_run=dry_run)
    _run(["git", "checkout", ref], cwd=dest, dry_run=dry_run)
    if reset:
        _run(["git", "reset", "--hard", ref], cwd=dest, dry_run=dry_run)


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


def _build_utopic(native_dir: Path, llama_dir: Path, *, jobs: Optional[int], dry_run: bool) -> Path:
    out_dir = build_root() / "utopic"

    _run(
        ["cmake", "-B", out_dir, "-S", native_dir / "native", f"-DUTOPIC_LLAMACPP_DIR={llama_dir}"],
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
    parser.add_argument("--force", action="store_true", help="Remove cached binaries before rebuilding.")
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=int(os.environ["UTOPIC_BUILD_JOBS"]) if os.environ.get("UTOPIC_BUILD_JOBS") else None,
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

    dry_run = bool(args.dry_run)
    backend = "cuda" if args.cuda else args.backend
    llama_dir = Path(args.llama_dir).expanduser() if args.llama_dir else default_llama_dir()
    native_dir = Path(args.native_dir).expanduser() if args.native_dir else default_native_dir()

    if args.force and bin_dir().exists():
        print(f"+ remove {bin_dir()}")
        if not dry_run:
            shutil.rmtree(bin_dir())

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
            backend=backend,
            cuda_architectures=args.cuda_architectures,
            jobs=args.jobs,
            dry_run=dry_run,
        )

    if args.native_dir or os.environ.get("UTOPIC_NATIVE_DIR"):
        print(f"Using external Utopic source at {native_dir}")
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
    print(f"Installed Utopic native binaries to {bin_dir()}")
    return 0
