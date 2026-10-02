# Changelog

## 1.2.0

### The models are now downloaded from OpenFace

pymtcnn's face-detection models come from OpenFace 2.2.0. OpenFace is licensed for
academic or non-profit noncommercial research only, and its license does not allow
redistributing the models, so pymtcnn no longer includes them. Earlier versions
bundled them; please upgrade.

After installing pymtcnn, run once:

```bash
pymtcnn-download-models
```

It shows a short summary of the OpenFace license, asks you to type `YES`, downloads
OpenFace's original `PNet.dat`, `RNet.dat` and `ONet.dat` (about 2.2 MB) from OpenFace's
official GitHub repository, checks each file's SHA-256 checksum, and converts them on
your computer into the ONNX and Core ML models pymtcnn uses. The converted models are
byte-for-byte identical to the ones shipped in 1.1.5, so detections are unchanged.

- New: `pymtcnn-download-models` command (also `python -m pymtcnn.download_models`),
  with `--accept-license` for scripts.
- New: `pymtcnn.models.ensure_models(accept_license=False, *, cache_dir=None, progress=None)`
  installs the models and returns their folder. `progress(done_bytes, total_bytes, name)`
  reports the download. It raises `pymtcnn.ModelsNotInstalledError` (a `FileNotFoundError`)
  with installation instructions when the files are missing and the license was not
  accepted, and `pymtcnn.models.ModelDownloadError` (a `RuntimeError`) when a download
  or a check fails.
- `MTCNN()`, `CoreMLMTCNN()` and `ONNXMTCNN()` find the installed models automatically.
  They never download without consent: only `OPENFACE_MODELS_ACCEPT_LICENSE=1` counts as
  accepting the license.
- Models are stored in a folder shared with pyclnf and pyfaceau
  (`OpenFaceModels/2.2.0` in your user data folder; override with `OPENFACE_MODELS_DIR`).
  Files another package already downloaded are reused without asking again.
- `PurePythonMTCNN` reads OpenFace's `.dat` files from that shared folder.
- `MTCNN()` with automatic backend selection only picks a backend that loads its models
  and runs on this computer: CoreML (macOS), then ONNX with automatic provider selection,
  then ONNX on the CPU. On a Mac that has onnxruntime but not coremltools, ONNX Runtime's
  CoreML provider can fail (seen with onnxruntime 1.23.2 and 1.30.0 on macOS 26); pymtcnn
  1.1.5 then stopped with an error, 1.2.0 uses the CPU. `ONNXMTCNN()` without an explicit
  provider also falls back to the CPU when an accelerated provider cannot load the models.
- Removed the bundled ONNX and Core ML model files and the Git LFS setup.

## 1.1.5 and earlier

Bundled the ONNX and Core ML models in the package.
