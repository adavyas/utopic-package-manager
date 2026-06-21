import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from utopic import _native, cli, installer


ROOT = Path(__file__).resolve().parents[1]


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

    def test_verify_llama_apis_reports_missing_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            include = Path(tmp) / "include"
            include.mkdir()
            (include / "llama.h").write_text("llama_diffusion_set_sc\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "llama_diffusion_set_block_decode"):
                installer._verify_llama_apis(Path(tmp))


class PackagingTests(unittest.TestCase):
    def test_pyproject_uses_pure_python_build_backend(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('build-backend = "setuptools.build_meta"', text)
        self.assertNotIn("scikit-build-core", text)
        self.assertNotIn("[tool.scikit-build]", text)

    def test_no_package_manager_cmake_entrypoint(self):
        self.assertFalse((ROOT / "CMakeLists.txt").exists())


if __name__ == "__main__":
    unittest.main()
