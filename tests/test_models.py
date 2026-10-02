"""Tests for model installation: cache location, license gate, download, verification, locking.

Downloads are served by a local HTTP server with synthetic weight files, so these
tests use no network and no OpenFace data.
"""

import hashlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import _synthetic  # noqa: E402

from pymtcnn import _convert, download_models, models  # noqa: E402


class _Server:
    """Local HTTP server. ``routes`` maps a path to a list of (status, body) answers."""

    def __init__(self, routes):
        self.routes = routes
        self.hits = {}
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                server.hits[self.path] = server.hits.get(self.path, 0) + 1
                answers = server.routes.get(self.path, [(404, b"")])
                status, body = answers.pop(0) if len(answers) > 1 else answers[0]
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class InstallTestCase(unittest.TestCase):
    """Sets up synthetic OpenFace files, a local server and an empty cache folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "cache"
        self.root = self.cache / "2.2.0"
        src = Path(self.tmp.name) / "src"
        src.mkdir()
        self.data = {}
        for seed, (net, name) in enumerate(_synthetic.FILE_NAMES.items()):
            self.data[name] = _synthetic.write_dat(src / name, net, seed)
        # Expected hashes and sizes of the conversion of these synthetic files.
        out = Path(self.tmp.name) / "expected"
        with mock.patch.object(_convert, "verify", lambda d: None):
            _convert.convert({net: str(src / name) for net, name in _synthetic.FILE_NAMES.items()}, str(out))
        expected = {rel: models._sha256_of(out / rel) for rel in _convert.EXPECTED_SHA256}
        sizes = {rel: (out / rel).stat().st_size for rel in _convert.EXPECTED_SIZE}
        self.server = _Server({"/" + name: [(200, body)] for name, body in self.data.items()})
        self.addCleanup(self.server.close)
        base = "lib/local/LandmarkDetector/model/mtcnn_detector/"
        self.manifest = {"files": [
            {"path": base + name, "urls": [self.server.url + "/" + name],
             "sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
            for name, body in self.data.items()]}
        for patcher in (mock.patch.object(models, "load_manifest", lambda: self.manifest),
                        mock.patch.dict(_convert.EXPECTED_SHA256, expected),
                        mock.patch.dict(_convert.EXPECTED_SIZE, sizes),
                        mock.patch.object(models.time, "sleep", lambda s: None),
                        mock.patch.dict(os.environ, {}, clear=False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop(models.ENV_ACCEPT_LICENSE, None)
        os.environ.pop(models.ENV_MODELS_DIR, None)

    def files_in_cache(self):
        return sorted(str(p.relative_to(self.cache)) for p in self.cache.rglob("*") if p.is_file())


class LocationTest(unittest.TestCase):
    def test_cache_dir_argument_and_environment(self):
        with mock.patch.dict(os.environ, {models.ENV_MODELS_DIR: " /tmp/of-models "}):
            self.assertEqual(models.models_root(), Path("/tmp/of-models/2.2.0"))
            self.assertEqual(models.models_root("/data/x"), Path("/data/x/2.2.0"))
            self.assertEqual(models.models_dir("/data/x"), Path("/data/x/2.2.0/derived/pymtcnn/1"))

    def test_platform_defaults(self):
        home = Path.home()
        env = {k: v for k, v in os.environ.items() if k not in (models.ENV_MODELS_DIR, "XDG_DATA_HOME")}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(models.sys, "platform", "darwin"):
                self.assertEqual(models.models_root(),
                                 home / "Library" / "Application Support" / "OpenFaceModels" / "2.2.0")
            if os.name == "nt":
                with mock.patch.dict(os.environ, {"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}):
                    self.assertEqual(models.models_root(),
                                     Path(r"C:\Users\x\AppData\Local\OpenFaceModels\2.2.0"))
            else:
                with mock.patch.object(models.sys, "platform", "linux"):
                    self.assertEqual(models.models_root(),
                                     home / ".local" / "share" / "OpenFaceModels" / "2.2.0")
                    with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/xdg"}):
                        self.assertEqual(models.models_root(), Path("/xdg/OpenFaceModels/2.2.0"))
                    with mock.patch.dict(os.environ, {"XDG_DATA_HOME": "relative/ignored"}):
                        self.assertEqual(models.models_root(),
                                         home / ".local" / "share" / "OpenFaceModels" / "2.2.0")

    def test_manifest_lists_official_openface_files(self):
        manifest = models.load_manifest()
        names = sorted(e["path"].rsplit("/", 1)[-1] for e in manifest["files"])
        self.assertEqual(names, ["ONet.dat", "PNet.dat", "RNet.dat"])
        for entry in manifest["files"]:
            self.assertEqual(entry["urls"], [
                "https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/OpenFace_2.2.0/" + entry["path"]])
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["size"], 0)

    def test_exception_classes(self):
        self.assertTrue(issubclass(models.ModelsNotInstalledError, FileNotFoundError))
        self.assertTrue(issubclass(models.ModelDownloadError, RuntimeError))
        self.assertFalse(issubclass(models.ModelDownloadError, models.ModelsNotInstalledError))

    def test_only_the_exact_value_1_accepts_the_license(self):
        for value, accepted in (("1", True), (" 1 ", True), ("true", False), ("yes", False), ("0", False)):
            with mock.patch.dict(os.environ, {models.ENV_ACCEPT_LICENSE: value}):
                self.assertEqual(models.license_accepted(), accepted, value)
        self.assertTrue(models.license_accepted(True))


class LicenseGateTest(InstallTestCase):
    def test_no_download_without_consent(self):
        with self.assertRaises(models.ModelsNotInstalledError) as ctx:
            models.ensure_models(cache_dir=self.cache)
        self.assertIsInstance(ctx.exception, FileNotFoundError)
        message = str(ctx.exception)
        self.assertIn("pymtcnn-download-models", message)
        self.assertIn("-m pymtcnn.download_models", message)
        self.assertIn(models.LICENSE_URL, message)
        self.assertEqual(self.server.hits, {})
        self.assertFalse(self.cache.exists())

    def test_constructors_do_not_download(self):
        import pymtcnn
        with mock.patch.dict(os.environ, {models.ENV_MODELS_DIR: str(self.cache)}):
            with self.assertRaises(pymtcnn.ModelsNotInstalledError):
                pymtcnn.MTCNN()
        self.assertEqual(self.server.hits, {})

    def test_environment_variable_counts_as_consent(self):
        with mock.patch.dict(os.environ, {models.ENV_ACCEPT_LICENSE: "yes"}):
            with self.assertRaises(models.ModelsNotInstalledError):
                models.ensure_models(cache_dir=self.cache)
        with mock.patch.dict(os.environ, {models.ENV_ACCEPT_LICENSE: "1"}):
            target = models.ensure_models(cache_dir=self.cache)
        self.assertTrue((target / "pnet.onnx").is_file())


class InstallTest(InstallTestCase):
    def test_download_verify_convert_and_reuse(self):
        events = []
        target = models.ensure_models(True, cache_dir=self.cache, progress=lambda d, t, n: events.append((d, t, n)))
        self.assertEqual(target, self.root / "derived" / "pymtcnn" / _convert.CONVERTER_VERSION)
        for rel in _convert.EXPECTED_SHA256:
            self.assertTrue((target / rel).is_file(), rel)
        stamp = json.loads((target / models.STAMP_FILE).read_text())
        self.assertEqual((stamp["package"], stamp["converter_version"], stamp["openface_version"]),
                         ("pymtcnn", _convert.CONVERTER_VERSION, "2.2.0"))
        self.assertEqual(stamp["files"], _convert.EXPECTED_SHA256)
        for entry in self.manifest["files"]:
            self.assertEqual((self.root / "originals" / entry["path"]).read_bytes(),
                             self.data[entry["path"].rsplit("/", 1)[-1]])
        total = sum(e["size"] for e in self.manifest["files"])
        self.assertEqual(events[0], (0, total, ""))
        self.assertEqual(events[-1], (total, total, ""))
        self.assertEqual({n for _, _, n in events if n}, {e["path"] for e in self.manifest["files"]})
        self.assertEqual([d for d, _, _ in events], sorted(d for d, _, _ in events))
        self.assertEqual(set(self.server.hits.values()), {1})
        self.assertFalse([f for f in self.files_in_cache() if ".part-" in f or "/.tmp-" in f])
        # Second call: nothing is downloaded again.
        self.assertTrue(models.models_ready(self.cache))
        self.assertEqual(models.ensure_models(cache_dir=self.cache), target)
        self.assertEqual(set(self.server.hits.values()), {1})

    def test_converts_from_cached_originals_without_license(self):
        target = models.ensure_models(True, cache_dir=self.cache)
        (target / "onet.onnx").write_bytes(b"damaged")
        self.assertFalse(models.models_ready(self.cache))
        self.assertEqual(models.ensure_models(cache_dir=self.cache), target)  # no consent needed
        self.assertEqual(models._sha256_of(target / "onet.onnx"), _convert.EXPECTED_SHA256["onet.onnx"])
        self.assertEqual(set(self.server.hits.values()), {1})

    def test_corrupted_download_is_never_used(self):
        self.server.routes["/RNet.dat"] = [(200, self.data["RNet.dat"][:-1] + b"X")]
        with self.assertRaises(models.ModelDownloadError) as ctx:
            models.ensure_models(True, cache_dir=self.cache)
        self.assertNotIsInstance(ctx.exception, models.ModelsNotInstalledError)
        self.assertIn("RNet.dat", str(ctx.exception))
        files = self.files_in_cache()
        self.assertFalse([f for f in files if f.endswith("RNet.dat") or ".part-" in f])
        self.assertFalse([f for f in files if "/derived/" in f])
        self.assertEqual(self.server.hits["/RNet.dat"], models._RETRIES)

    def test_retries_temporary_errors(self):
        self.server.routes["/ONet.dat"] = [(503, b""), (500, b""), (200, self.data["ONet.dat"])]
        models.ensure_models(True, cache_dir=self.cache)
        self.assertEqual(self.server.hits["/ONet.dat"], 3)

    def test_missing_file_is_not_retried(self):
        self.server.routes["/PNet.dat"] = [(404, b"")]
        with self.assertRaises(models.ModelDownloadError):
            models.ensure_models(True, cache_dir=self.cache)
        self.assertEqual(self.server.hits["/PNet.dat"], 1)

    def test_concurrent_installs_download_once(self):
        results, errors = [], []

        def worker():
            try:
                results.append(models.ensure_models(True, cache_dir=self.cache))
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(set(self.server.hits.values()), {1})

    def test_lock_file_location(self):
        models.ensure_models(True, cache_dir=self.cache)
        self.assertTrue((self.root / ".lock").is_file())


class CommandLineTest(InstallTestCase):
    def run_cli(self, args, answer=None):
        out = io.StringIO()
        stdin = io.StringIO(answer) if answer is not None else io.StringIO("")
        with redirect_stdout(out), mock.patch.object(sys, "stdin", stdin):
            code = download_models.main(args + ["--cache-dir", str(self.cache)])
        return code, out.getvalue()

    def test_declining_downloads_nothing(self):
        code, text = self.run_cli([], answer="no\n")
        self.assertEqual(code, 1)
        self.assertIn(models.LICENSE_URL, text)
        self.assertEqual(self.server.hits, {})

    def test_typing_yes_installs(self):
        code, text = self.run_cli([], answer="yes\n")
        self.assertEqual(code, 0, text)
        self.assertIn("Done.", text)
        self.assertIn("downloading PNet.dat", text)  # progress bar fed (done, total, name)

    def test_accept_license_flag_and_rerun(self):
        code, text = self.run_cli(["--accept-license"])
        self.assertEqual(code, 0, text)
        code, text = self.run_cli(["--accept-license"])
        self.assertEqual(code, 0)
        self.assertIn("ready in", text)
        self.assertEqual(set(self.server.hits.values()), {1})

    def test_no_terminal_input(self):
        code, text = self.run_cli([])
        self.assertEqual(code, 1)
        self.assertIn("--accept-license", text)
        self.assertEqual(self.server.hits, {})


if __name__ == "__main__":
    unittest.main()
