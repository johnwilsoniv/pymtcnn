"""Tests for the OpenFace .dat -> ONNX / Core ML converter.

Most tests use synthetic weights. The test that checks the output against
pymtcnn 1.1.5 needs OpenFace's real files: it uses the shared model cache when
the models are installed, or a folder with PNet.dat/RNet.dat/ONet.dat given in
the PYMTCNN_OPENFACE_DIR environment variable, and is skipped otherwise.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import _synthetic  # noqa: E402

from pymtcnn import _convert, models  # noqa: E402

NETS = ("pnet", "rnet", "onet")
INPUT_SHAPES = {"pnet": (1, 3, 31, 45), "rnet": (2, 3, 24, 24), "onet": (2, 3, 48, 48)}


def _synthetic_params(tmp, net, seed=0):
    path = os.path.join(tmp, _synthetic.FILE_NAMES[net])
    _synthetic.write_dat(path, net, seed)
    return _convert._parameters(net, _convert.read_openface_cnn(path))


class ProtobufWriterTest(unittest.TestCase):
    def test_varints(self):
        self.assertEqual(_convert._varint(0), b"\x00")
        self.assertEqual(_convert._varint(300), b"\xac\x02")
        self.assertEqual(_convert._varint(-1), b"\xff" * 9 + b"\x01")


class ReaderTest(unittest.TestCase):
    def test_reads_synthetic_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for net in NETS:
                params = _synthetic_params(tmp, net)
                for key, shape in _convert._ARCH[net]["param_shapes"].items():
                    self.assertEqual(params[key].shape, shape, key)
                    self.assertEqual(params[key].dtype, np.float32)

    def test_rejects_truncated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "PNet.dat")
            data = _synthetic.write_dat(path, "pnet")
            with open(path, "wb") as f:
                f.write(data[:-10])
            with self.assertRaises(_convert.ConversionError):
                _convert.read_openface_cnn(path)

    def test_rejects_wrong_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "RNet.dat")
            _synthetic.write_dat(path, "rnet")
            with self.assertRaises(_convert.ConversionError):
                _convert._parameters("pnet", _convert.read_openface_cnn(path))


class OnnxOutputTest(unittest.TestCase):
    """The ONNX graphs compute the MTCNN networks (checked with random weights)."""

    def test_onnx_matches_reference(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime is not installed")
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            for net in NETS:
                params = _synthetic_params(tmp, net)
                session = ort.InferenceSession(_convert.build_onnx(net, params),
                                               providers=["CPUExecutionProvider"])
                x = rng.uniform(-1, 1, INPUT_SHAPES[net]).astype(np.float32)
                got = session.run(None, {"input": x})[0]
                want = _synthetic.reference_forward(net, params, x)
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4, err_msg=net)


class CoreMLOutputTest(unittest.TestCase):
    """The Core ML packages load and compute the MTCNN networks (macOS only)."""

    def test_coreml_matches_reference(self):
        if sys.platform != "darwin":
            self.skipTest("Core ML runs on macOS only")
        try:
            import coremltools as ct
        except ImportError:
            self.skipTest("coremltools is not installed")
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            for net in NETS:
                params = _synthetic_params(tmp, net)
                model, weights, manifest = _convert.build_coreml(net, params)
                package = os.path.join(tmp, net + "_fp32.mlpackage")
                _convert._write(os.path.join(package, "Data", "com.apple.CoreML", "model.mlmodel"), model)
                _convert._write(os.path.join(package, "Data", "com.apple.CoreML", "weights", "weight.bin"),
                                weights)
                _convert._write(os.path.join(package, "Manifest.json"), manifest)
                mlmodel = ct.models.MLModel(package, compute_units=ct.ComputeUnit.CPU_ONLY)
                output = mlmodel.get_spec().description.output[0].name
                # Core ML pads partial ceil-mode pooling windows differently from ONNX at the
                # right/bottom edge, so use a PNet size without partial windows.
                shape = (1, 3, 32, 46) if net == "pnet" else INPUT_SHAPES[net]
                x = rng.uniform(-1, 1, shape).astype(np.float32)
                got = mlmodel.predict({"input": x})[output]
                want = _synthetic.reference_forward(net, params, x)
                np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3, err_msg=net)


def _real_openface_dir():
    folder = os.environ.get("PYMTCNN_OPENFACE_DIR")
    if folder:
        return folder
    root = models.models_root()
    entries = models.load_manifest()["files"]
    if all(models._original_ok(root, e) for e in entries):
        return str(models._original_path(root, entries[0]).parent)
    return None


class RealWeightsTest(unittest.TestCase):
    def test_conversion_is_byte_identical_to_1_1_5(self):
        folder = _real_openface_dir()
        if folder is None:
            self.skipTest("OpenFace's MTCNN files are not available (run pymtcnn-download-models "
                          "or set PYMTCNN_OPENFACE_DIR)")
        manifest = {os.path.basename(e["path"]): e for e in models.load_manifest()["files"]}
        originals = {}
        for net, name in _convert.DAT_FILES.items():
            path = os.path.join(folder, name)
            self.assertEqual(models._sha256_of(path), manifest[name]["sha256"], name)
            originals[net] = path
        with tempfile.TemporaryDirectory() as tmp:
            _convert.convert(originals, tmp)  # raises unless every file matches EXPECTED_SHA256
            for rel in _convert.EXPECTED_SHA256:
                self.assertTrue(os.path.isfile(os.path.join(tmp, *rel.split("/"))), rel)


if __name__ == "__main__":
    unittest.main()
