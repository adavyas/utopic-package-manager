import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utopic import _native


ROOT = Path(__file__).resolve().parents[1]


class NativeLauncherTests(unittest.TestCase):
    def test_binary_path_resolves_packaged_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            exe = bin_dir / "utopic"
            exe.write_text("#!/bin/sh\n", encoding="utf-8")

            with mock.patch.object(_native, "PACKAGE_DIR", root):
                self.assertEqual(_native.binary_path("utopic"), exe)

    def test_binary_path_uses_exe_suffix_on_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            exe = bin_dir / "utopic.exe"
            exe.write_text("", encoding="utf-8")

            with mock.patch.object(_native, "PACKAGE_DIR", root):
                with mock.patch.object(os, "name", "nt"):
                    self.assertEqual(_native.binary_path("utopic"), exe)

    def test_binary_path_errors_when_binary_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(_native, "PACKAGE_DIR", Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, "native binary was not installed"):
                    _native.binary_path("utopic")

    def test_main_execs_selected_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "utopic_server"
            exe.write_text("", encoding="utf-8")

            with mock.patch.object(_native, "binary_path", return_value=exe):
                with mock.patch("os.execv") as execv:
                    _native.main("utopic_server", ["--help"])

            execv.assert_called_once_with(str(exe), [str(exe), "--help"])


class CMakeFetchTests(unittest.TestCase):
    def test_cmake_fetches_pinned_native_runtime(self):
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("FetchContent_Declare", text)
        self.assertIn("https://github.com/adavyas/Utopic.git", text)
        self.assertIn("6943cab5a80ac165bd6c4a14962c6d4b64cb6226", text)
        self.assertRegex(text, re.compile(r"SOURCE_SUBDIR\s+native"))
        self.assertIn("install(TARGETS utopic utopic_server utopic_mcp utopic_acp", text)


if __name__ == "__main__":
    unittest.main()
