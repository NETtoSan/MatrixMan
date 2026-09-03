import os
import unittest
from unittest import mock
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman
from drivers.matrixman.backends.opengl import adapter


class DgpuPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.pop("MATRIXMAN_USE_DGPU", None)
        matrixman.config.reset()

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("MATRIXMAN_USE_DGPU", None)
        else:
            os.environ["MATRIXMAN_USE_DGPU"] = self.previous
        matrixman.config.reloadFromEnvironment()

    def test_default_and_python_override(self):
        self.assertFalse(matrixman.config.useDGPU)
        matrixman.config.useDGPU = True
        self.assertTrue(matrixman.config.useDGPU)
        self.assertTrue(matrixman.config.asDict()["useDGPU"])

    def test_environment_reload_and_reset(self):
        os.environ["MATRIXMAN_USE_DGPU"] = "1"
        matrixman.config.reloadFromEnvironment()
        self.assertTrue(matrixman.config.useDGPU)
        matrixman.config.reset()
        self.assertFalse(matrixman.config.useDGPU)
        matrixman.config.reloadFromEnvironment()
        self.assertTrue(matrixman.config.useDGPU)

    def test_renderer_policy_does_not_use_vendor_as_topology(self):
        self.assertEqual(adapter.classify_renderer("AMD", "Radeon RX 580"), "discrete")
        self.assertEqual(adapter.classify_renderer("Intel", "UHD Graphics 630"), "integrated")
        self.assertEqual(adapter.classify_renderer("NVIDIA", "unknown renderer"), "unknown")

    def test_unavailable_preference_is_reported(self):
        result = adapter.finalize_preference(
            {"gpu_preference": "discrete", "gpu_preference_honored": "unknown", "gpu_preference_reason": ""},
            "Intel",
            "UHD Graphics 630",
        )
        self.assertEqual(result["gpu_preference_honored"], "no")

    def test_windows_reports_sdl_limitation_without_faking_selection(self):
        with mock.patch.object(adapter.os, "name", "nt"):
            result = adapter.request_preference(True)
        self.assertEqual(result["gpu_preference"], "discrete")
        self.assertEqual(result["gpu_preference_honored"], "unknown")
        self.assertIn("SDL/WGL", result["gpu_preference_reason"])

    def test_linux_preserves_existing_prime_selection(self):
        with mock.patch.object(adapter.os, "name", "posix"), mock.patch.dict(
            adapter.os.environ, {"DRI_PRIME": "1"}, clear=False
        ):
            result = adapter.request_preference(True)
        self.assertEqual(result["gpu_preference_honored"], "requested-via-DRI_PRIME")

    def test_backend_preference_remains_independent(self):
        matrixman.config.backend = "auto"
        matrixman.config.useDGPU = True
        self.assertEqual(matrixman.config.backend, "auto")


if __name__ == "__main__":
    unittest.main()
