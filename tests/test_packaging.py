import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from utopic import _native, cli, installer


ROOT = Path(__file__).resolve().parents[1]


def subprocess_completed(*, stdout: str):
    return mock.Mock(stdout=stdout)


class NativeLauncherTests(unittest.TestCase):
    def test_binary_path_resolves_cached_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            exe = bin_dir / "utopic"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"UTOPIC_BIN_DIR": str(bin_dir)}):
                self.assertEqual(_native.binary_path("utopic"), exe)

    def test_binary_path_uses_exe_suffix_on_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            exe = bin_dir / "utopic.exe"
            exe.write_text("", encoding="utf-8")

            with mock.patch.dict(os.environ, {"UTOPIC_BIN_DIR": str(bin_dir)}):
                with mock.patch.object(_native, "_binary_suffix", return_value=".exe"):
                    self.assertEqual(_native.binary_path("utopic"), exe)

    def test_binary_path_errors_when_binary_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_BIN_DIR": str(Path(tmp))}):
                with self.assertRaisesRegex(RuntimeError, "utopic setup"):
                    _native.binary_path("utopic")

    def test_main_execs_selected_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "utopic_server"
            exe.write_text("", encoding="utf-8")

            with mock.patch.object(_native, "binary_path", return_value=exe):
                with mock.patch("os.execv") as execv:
                    _native.main("utopic_server", ["--help"])

            execv.assert_called_once_with(str(exe), [str(exe), "--help"])


class CliTests(unittest.TestCase):
    def test_setup_subcommand_runs_installer(self):
        with mock.patch.object(installer, "setup", return_value=0) as setup:
            with mock.patch.object(_native, "main") as native:
                with mock.patch("sys.argv", ["utopic", "setup", "--dry-run"]):
                    with self.assertRaisesRegex(SystemExit, "0"):
                        cli.main()

        setup.assert_called_once_with(["--dry-run"])
        native.assert_not_called()

    def test_run_subcommand_strips_verb_before_native_exec(self):
        with mock.patch.object(_native, "main") as native:
            with mock.patch("sys.argv", ["utopic", "run", "-m", "model.gguf"]):
                cli.main()

        native.assert_called_once_with("utopic", ["-m", "model.gguf"])


class InstallerTests(unittest.TestCase):
    def test_cache_root_respects_utopic_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=False):
                self.assertEqual(installer.cache_root(), Path(tmp))
                self.assertEqual(installer.bin_dir(), Path(tmp) / "bin")

    def test_setup_dry_run_accepts_external_llama_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(installer, "_run") as run:
                with redirect_stdout(StringIO()):
                    result = installer.setup(["--dry-run", "--llama-dir", tmp])

        self.assertEqual(result, 0)
        self.assertTrue(run.called)
        self.assertTrue(all(call.kwargs["dry_run"] for call in run.call_args_list))

    def test_setup_help_keeps_llama_checkout_internal(self):
        out = StringIO()
        with redirect_stdout(out):
            with self.assertRaises(SystemExit):
                installer.setup(["--help"])

        help_text = out.getvalue()
        self.assertIn("utopic setup", help_text)
        self.assertNotIn("llama.cpp", help_text)
        self.assertNotIn("--llama-dir", help_text)
        self.assertNotIn("--skip-llama-build", help_text)

    def test_backend_cuda_adds_cuda_cmake_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_find_cuda_compiler", return_value=None):
                    with mock.patch.object(installer, "_detect_cuda_architectures", return_value=None):
                        with mock.patch.object(installer, "_run") as run:
                            with redirect_stdout(StringIO()):
                                result = installer.setup(["--dry-run", "--backend", "cuda"])

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            [
                "cmake",
                "-B",
                Path(tmp) / "src" / "llama.cpp" / "build",
                "-S",
                Path(tmp) / "src" / "llama.cpp",
                *installer.LLAMA_CMAKE_FLAGS,
                "-DGGML_CUDA=ON",
            ],
            commands,
        )

    def test_backend_cpu_disables_gpu_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_run") as run:
                    with redirect_stdout(StringIO()):
                        result = installer.setup(["--dry-run", "--backend", "cpu"])

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            [
                "cmake",
                "-B",
                Path(tmp) / "src" / "llama.cpp" / "build",
                "-S",
                Path(tmp) / "src" / "llama.cpp",
                *installer.LLAMA_CMAKE_FLAGS,
                "-DGGML_CUDA=OFF",
                "-DGGML_METAL=OFF",
            ],
            commands,
        )

    def test_backend_cuda_adds_discovered_nvcc_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = Path(tmp) / "cuda" / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("", encoding="utf-8")

            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_find_cuda_compiler", return_value=nvcc):
                    with mock.patch.object(installer, "_detect_cuda_architectures", return_value=None):
                        with mock.patch.object(installer, "_run") as run:
                            with redirect_stdout(StringIO()):
                                result = installer.setup(["--dry-run", "--backend", "cuda"])

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            [
                "cmake",
                "-B",
                Path(tmp) / "src" / "llama.cpp" / "build",
                "-S",
                Path(tmp) / "src" / "llama.cpp",
                *installer.LLAMA_CMAKE_FLAGS,
                "-DGGML_CUDA=ON",
                f"-DCMAKE_CUDA_COMPILER={nvcc}",
            ],
            commands,
        )

    def test_backend_cuda_adds_requested_cuda_architectures(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_find_cuda_compiler", return_value=None):
                    with mock.patch.object(installer, "_run") as run:
                        with redirect_stdout(StringIO()):
                            result = installer.setup(
                                ["--dry-run", "--backend", "cuda", "--cuda-architectures", "89"]
                            )

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            [
                "cmake",
                "-B",
                Path(tmp) / "src" / "llama.cpp" / "build",
                "-S",
                Path(tmp) / "src" / "llama.cpp",
                *installer.LLAMA_CMAKE_FLAGS,
                "-DGGML_CUDA=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=89",
            ],
            commands,
        )

    def test_backend_cuda_detects_cuda_architectures(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_find_cuda_compiler", return_value=None):
                    with mock.patch.object(installer, "_detect_cuda_architectures", return_value="89"):
                        with mock.patch.object(installer, "_run") as run:
                            with redirect_stdout(StringIO()):
                                result = installer.setup(["--dry-run", "--backend", "cuda"])

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            [
                "cmake",
                "-B",
                Path(tmp) / "src" / "llama.cpp" / "build",
                "-S",
                Path(tmp) / "src" / "llama.cpp",
                *installer.LLAMA_CMAKE_FLAGS,
                "-DGGML_CUDA=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=89",
            ],
            commands,
        )

    def test_detect_cuda_architectures_deduplicates_nvidia_smi_caps(self):
        completed = subprocess_completed(stdout="8.9\n8.9\n9.0\n")

        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(installer._detect_cuda_architectures(), "89;90")

    def test_cuda_121_prefers_cuda_13_compiler_candidate(self):
        candidates = installer._cuda_compiler_candidates("121")

        self.assertLess(
            candidates.index(Path("/usr/local/cuda-13.0/bin/nvcc")),
            candidates.index(Path("/usr/local/cuda/bin/nvcc")),
        )

    def test_jobs_limits_native_and_llama_build_parallelism(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_run") as run:
                    with redirect_stdout(StringIO()):
                        result = installer.setup(["--dry-run", "--jobs", "2"])

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["cmake", "--build", Path(tmp) / "src" / "llama.cpp" / "build", "-j", "2"], commands)
        self.assertIn(["cmake", "--build", Path(tmp) / "build" / "utopic", "-j", "2"], commands)

    def test_setup_passes_managed_dependency_as_internal_cmake_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_run") as run:
                    with redirect_stdout(StringIO()):
                        result = installer.setup(["--dry-run"])

        self.assertEqual(result, 0)
        configure_calls = [
            call for call in run.call_args_list if call.args[0][:2] == ["cmake", "-B"]
        ]
        utopic_configure = next(
            call for call in configure_calls if call.args[0][2] == Path(tmp) / "build" / "utopic"
        )
        self.assertIn(
            f"-DUTOPIC_LLAMACPP_DIR={Path(tmp) / 'src' / 'llama.cpp'}",
            utopic_configure.args[0],
        )
        self.assertNotIn("UTOPIC_LLAMACPP_DIR", utopic_configure.kwargs.get("env", {}))

    def test_setup_dry_run_fetches_managed_dependency_without_patch_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"UTOPIC_HOME": tmp}, clear=True):
                with mock.patch.object(installer, "_run") as run:
                    with redirect_stdout(StringIO()):
                        result = installer.setup(["--dry-run"])

        self.assertEqual(result, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(["git", "clone", installer.LLAMA_REPO, Path(tmp) / "src" / "llama.cpp"], commands)
        self.assertIn(["git", "checkout", installer.LLAMA_REF], commands)
        self.assertFalse(any(command[:2] == ["git", "apply"] for command in commands))

    def test_default_llama_source_is_official_stock_good_pin(self):
        self.assertEqual(installer.LLAMA_REPO, "https://github.com/ggml-org/llama.cpp.git")
        self.assertRegex(installer.LLAMA_REF, r"^[0-9a-f]{40}$")

    def test_default_utopic_source_uses_package_managed_dependency_contract(self):
        self.assertEqual(
            installer.UTOPIC_NATIVE_REF,
            "1fb77ec47577f3e6ed25f8a14afda3f86c35769d",
        )

    def test_llama_build_flags_disable_user_facing_llama_targets(self):
        self.assertIn("-DLLAMA_BUILD_EXAMPLES=OFF", installer.LLAMA_CMAKE_FLAGS)
        self.assertIn("-DLLAMA_BUILD_TESTS=OFF", installer.LLAMA_CMAKE_FLAGS)
        self.assertIn("-DLLAMA_BUILD_TOOLS=OFF", installer.LLAMA_CMAKE_FLAGS)
        self.assertIn("-DLLAMA_BUILD_SERVER=OFF", installer.LLAMA_CMAKE_FLAGS)
        self.assertIn("-DLLAMA_BUILD_APP=OFF", installer.LLAMA_CMAKE_FLAGS)

    def test_verify_llama_apis_reports_missing_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "include"
            include.mkdir()
            (include / "llama.h").write_text("llama_diffusion_set_sc\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "llama_diffusion_device_sample"):
                installer._verify_llama_apis(Path(tmp))

    def test_verify_llama_apis_keeps_escape_hatches_internal(self):
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "include"
            include.mkdir()
            (include / "llama.h").write_text("llama_diffusion_set_sc\n", encoding="utf-8")

            with self.assertRaises(RuntimeError) as raised:
                installer._verify_llama_apis(Path(tmp))

        message = str(raised.exception)
        self.assertIn("utopic setup --force", message)
        self.assertNotIn("--llama-dir", message)
        self.assertNotIn("UTOPIC_LLAMACPP_DIR", message)


class PackagingTests(unittest.TestCase):
    def test_pyproject_uses_pure_python_build_backend(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('build-backend = "setuptools.build_meta"', text)
        self.assertNotIn("scikit-build-core", text)
        self.assertNotIn("[tool.scikit-build]", text)
        self.assertNotIn("patches/*.patch", text)

    def test_setup_py_keeps_old_setuptools_metadata_usable(self):
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('name="utopic"', text)
        self.assertIn('version="0.1.0"', text)
        self.assertIn('"utopic=utopic.cli:main"', text)

    def test_no_package_manager_cmake_entrypoint(self):
        self.assertFalse((ROOT / "CMakeLists.txt").exists())

    def test_readme_keeps_user_setup_package_managed(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("utopic setup", text)
        self.assertIn("package-managed", text)
        self.assertIn("python3 -m venv", text)
        self.assertNotIn("compatibility overlay", text)
        self.assertNotIn("CUDACXX=/usr/local/cuda-13.0/bin/nvcc", text)
        self.assertNotIn("utopic setup --llama-dir", text)
        self.assertNotIn("UTOPIC_LLAMACPP_DIR", text)


if __name__ == "__main__":
    unittest.main()
