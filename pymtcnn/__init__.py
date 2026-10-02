"""
PyMTCNN - High-Performance Cross-Platform MTCNN Face Detection

Cross-platform MTCNN implementation with automatic backend selection:
- CoreML (Apple Neural Engine) on macOS: 34.26 FPS
- ONNX + CUDA on NVIDIA GPUs: 50+ FPS
- ONNX + CPU fallback: 5-10 FPS

Example usage:
    from pymtcnn import MTCNN

    # Auto-select best backend
    detector = MTCNN()

    # Single-frame detection
    bboxes, landmarks = detector.detect(frame)

    # Batch processing (cross-frame batching)
    results = detector.detect_batch(frames)

    # Or use specific backends:
    from pymtcnn import CoreMLMTCNN, ONNXMTCNN

    coreml_detector = CoreMLMTCNN()  # Force CoreML
    onnx_detector = ONNXMTCNN()      # Force ONNX

Installation:
    - macOS (CoreML): pip install pymtcnn[coreml]
    - NVIDIA GPU: pip install pymtcnn[onnx-gpu]
    - CPU only: pip install pymtcnn[onnx]
    Then install the models once (OpenFace license; asks you to type YES):
        pymtcnn-download-models

Models:
    The MTCNN weights come from OpenFace 2.2.0 (academic or non-profit
    noncommercial research use only) and are not included in this package.
    pymtcnn-download-models (or pymtcnn.models.ensure_models) downloads
    OpenFace's original files from OpenFace's GitHub repository, verifies them
    and converts them locally. See pymtcnn.models.

Performance:
    - CoreML: 31.88 FPS (single-frame), 34.26 FPS (batch)
    - ONNX+CUDA: 50+ FPS on RTX GPUs
    - ONNX+CPU: 5-10 FPS
    - Accuracy: 95% IoU vs C++ OpenFace baseline

License:
    pymtcnn code: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
    OpenFace models: OpenFace license (see pymtcnn.models.LICENSE_URL)
"""

__version__ = "1.2.0"
__author__ = "SplitFace"
__license__ = "CC BY-NC 4.0"

# Primary API: Unified MTCNN with auto-backend selection
from .detector import MTCNN

# Model installation (OpenFace weights are downloaded and converted on first use)
from .models import ensure_models, ModelsNotInstalledError

# Backend-specific classes (optional, for advanced users)
try:
    from .backends.coreml_backend import CoreMLMTCNN
except ImportError:
    CoreMLMTCNN = None

try:
    from .backends.onnx_backend import ONNXMTCNN
except ImportError:
    ONNXMTCNN = None

# Export all available classes
__all__ = ["MTCNN", "ensure_models", "ModelsNotInstalledError"]
if CoreMLMTCNN is not None:
    __all__.append("CoreMLMTCNN")
if ONNXMTCNN is not None:
    __all__.append("ONNXMTCNN")
