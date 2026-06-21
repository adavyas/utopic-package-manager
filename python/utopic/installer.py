import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
UTOPIC_NATIVE_REPO = "https://github.com/adavyas/Utopic.git"
UTOPIC_NATIVE_REF = "6943cab5a80ac165bd6c4a14962c6d4b64cb6226"
LLAMA_REPO = "https://github.com/danielhanchen/llama.cpp.git"
LLAMA_REF = "ef5e2dcce81881ffad262576d073f25ca6c1ad50"
BIN_NAMES = ("utopic", "utopic_server", "utopic_mcp", "utopic_acp")
REQUIRED_LLAMA_SYMBOLS = (
    "llama_diffusion_set_sc",
    "llama_diffusion_device_sample",
    "llama_diffusion_set_phase",
    "llama_diffusion_set_block_decode",
)
LLAMA_CMAKE_FLAGS = (
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_TOOLS=OFF",
)


def llama_patch_path() -> Path:
    return PACKAGE_DIR / "patches" / "llama.cpp-utopic.patch"


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


def _apply_llama_patch(llama_dir: Path, *, dry_run: bool) -> None:
    patch = llama_patch_path()
    if not dry_run and not patch.exists():
        raise RuntimeError(f"Utopic llama.cpp compatibility patch was not found: {patch}")
    _run(["git", "apply", patch], cwd=llama_dir, dry_run=dry_run)


def _build_llama(llama_dir: Path, *, cuda: bool, dry_run: bool) -> None:
    command = ["cmake", "-B", llama_dir / "build", "-S", llama_dir, *LLAMA_CMAKE_FLAGS]
    if cuda:
        command.append("-DGGML_CUDA=ON")
    _run(command, dry_run=dry_run)
    _run(["cmake", "--build", llama_dir / "build", "-j"], dry_run=dry_run)


def _verify_llama_apis(llama_dir: Path) -> None:
    header = llama_dir / "include" / "llama.h"
    if not header.exists():
        raise RuntimeError(f"llama.cpp header was not found: {header}")
    text = header.read_text(encoding="utf-8")
    missing = [symbol for symbol in REQUIRED_LLAMA_SYMBOLS if symbol not in text]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "This llama.cpp checkout is missing Utopic diffusion APIs: "
            f"{names}. Use a Utopic-compatible checkout with --llama-dir, "
            "or set UTOPIC_LLAMA_REPO and UTOPIC_LLAMA_REF before running `utopic setup`."
        )


def _build_utopic(native_dir: Path, llama_dir: Path, *, dry_run: bool) -> Path:
    out_dir = build_root() / "utopic"
    env = os.environ.copy()
    env["UTOPIC_LLAMACPP_DIR"] = str(llama_dir)

    _run(
        ["cmake", "-B", out_dir, "-S", native_dir / "native"],
        env=env,
        dry_run=dry_run,
    )
    _run(["cmake", "--build", out_dir, "-j"], env=env, dry_run=dry_run)
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
        description="Build and cache the Utopic native runtime binaries.",
    )
    parser.add_argument("--cuda", action="store_true", help="Build llama.cpp with GGML_CUDA=ON.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--force", action="store_true", help="Remove cached binaries before rebuilding.")
    parser.add_argument("--llama-dir", help="Use an existing llama.cpp checkout instead of the managed cache.")
    parser.add_argument("--native-dir", help="Use an existing Utopic checkout instead of the managed cache.")
    parser.add_argument(
        "--skip-llama-build",
        action="store_true",
        help="Do not run the llama.cpp CMake build before building Utopic.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dry_run = bool(args.dry_run)
    llama_dir = Path(args.llama_dir).expanduser() if args.llama_dir else default_llama_dir()
    native_dir = Path(args.native_dir).expanduser() if args.native_dir else default_native_dir()

    if args.force and bin_dir().exists():
        print(f"+ remove {bin_dir()}")
        if not dry_run:
            shutil.rmtree(bin_dir())

    if args.llama_dir or os.environ.get("UTOPIC_LLAMACPP_DIR"):
        print(f"Using llama.cpp checkout at {llama_dir}")
    else:
        _clone_or_checkout(
            os.environ.get("UTOPIC_LLAMA_REPO", LLAMA_REPO),
            os.environ.get("UTOPIC_LLAMA_REF", LLAMA_REF),
            llama_dir,
            dry_run=dry_run,
            reset=True,
        )
        _apply_llama_patch(llama_dir, dry_run=dry_run)

    if not dry_run:
        _verify_llama_apis(llama_dir)

    if not args.skip_llama_build:
        _build_llama(llama_dir, cuda=bool(args.cuda), dry_run=dry_run)

    if args.native_dir or os.environ.get("UTOPIC_NATIVE_DIR"):
        print(f"Using Utopic checkout at {native_dir}")
    else:
        _clone_or_checkout(
            os.environ.get("UTOPIC_NATIVE_REPO", UTOPIC_NATIVE_REPO),
            os.environ.get("UTOPIC_NATIVE_REF", UTOPIC_NATIVE_REF),
            native_dir,
            dry_run=dry_run,
        )

    native_build_dir = _build_utopic(native_dir, llama_dir, dry_run=dry_run)
    if dry_run:
        print(f"Would install Utopic native binaries to {bin_dir()}")
        return 0

    _install_binaries(native_build_dir)
    print(f"Installed Utopic native binaries to {bin_dir()}")
    return 0
