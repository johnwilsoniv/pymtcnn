"""Backend tests that need no OpenFace data (synthetic weights)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _synthetic  # noqa: E402

from pymtcnn import _convert  # noqa: E402


def _write_synthetic_onnx(folder):
    for net, name in _synthetic.FILE_NAMES.items():
        path = os.path.join(folder, name)
        _synthetic.write_dat(path, net)
        params = _convert._parameters(net, _convert.read_openface_cnn(path))
        with open(os.path.join(folder, net + ".onnx"), "wb") as f:
            f.write(_convert.build_onnx(net, params))


class _BrokenAcceleratedProvider:
    """Simulates ONNX Runtime's CoreML execution provider failing on this Mac.

    mode "load": creating a session with it fails (what onnxruntime 1.23.2 does on macOS 26);
    mode "run": the session is created but running it fails.
    """

    def __init__(self, onnx_backend, mode):
        self.real = onnx_backend.ort.InferenceSession
        self.mode = mode

    def __call__(self, path, sess_options=None, providers=None):
        accelerated = providers != ["CPUExecutionProvider"]
        if accelerated and self.mode == "load":
            raise RuntimeError("Failed to create MLModel (simulated)")
        session = self.real(path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        if not accelerated:
            return session

        class Broken:
            def get_providers(self):
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

            def run(self, *args, **kwargs):
                raise RuntimeError("CoreML execution failed (simulated)")

        return Broken()


class BackendTestCase(unittest.TestCase):
    def setUp(self):
        try:
            from pymtcnn.backends import onnx_backend
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        if onnx_backend.ort is None:
            self.skipTest("onnxruntime is not installed")
        self.onnx_backend = onnx_backend
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        _write_synthetic_onnx(self.tmp.name)

    def simulate(self, mode):
        """Make the accelerated provider fail, and pretend it is available."""
        for patcher in (
                mock.patch.object(self.onnx_backend.ort, "InferenceSession",
                                  _BrokenAcceleratedProvider(self.onnx_backend, mode)),
                mock.patch.object(self.onnx_backend.ort, "get_available_providers",
                                  lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])):
            patcher.start()
            self.addCleanup(patcher.stop)

    def without_coremltools(self):
        from pymtcnn.backends import coreml_backend
        patcher = mock.patch.object(coreml_backend, "ct", None)
        patcher.start()
        self.addCleanup(patcher.stop)


class OnnxProviderFallbackTest(BackendTestCase):
    def test_automatic_provider_falls_back_to_cpu(self):
        self.simulate("load")
        detector = self.onnx_backend.ONNXMTCNN(model_dir=self.tmp.name)
        self.assertEqual(detector.get_active_provider(), "CPUExecutionProvider")

    def test_explicit_provider_still_fails(self):
        self.simulate("load")
        with self.assertRaises(RuntimeError):
            self.onnx_backend.ONNXMTCNN(model_dir=self.tmp.name, provider="coreml")


class AutoBackendWithoutCoreMLToolsTest(BackendTestCase):
    """MTCNN() must pick a backend that loads and runs when coremltools is missing."""

    def detect(self):
        from pymtcnn import MTCNN
        detector = MTCNN(model_dir=self.tmp.name)
        image = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        bboxes, landmarks = detector.detect(image)
        self.assertEqual(bboxes.shape[1:], (4,))
        self.assertEqual(landmarks.shape[1:], (5, 2))
        return detector

    def test_accelerated_provider_cannot_load(self):
        self.without_coremltools()
        self.simulate("load")
        detector = self.detect()
        self.assertEqual(detector.backend_name, "ONNX + CPU")

    def test_accelerated_provider_loads_but_cannot_run(self):
        self.without_coremltools()
        self.simulate("run")
        detector = self.detect()
        self.assertEqual(detector.backend_name, "ONNX + CPU")
        self.assertEqual(detector.get_backend_info()["provider"], "CPUExecutionProvider")

    def test_real_providers(self):
        # Whatever this machine's onnxruntime offers, auto-selection must end up working.
        self.without_coremltools()
        self.detect()


if __name__ == "__main__":
    unittest.main()
