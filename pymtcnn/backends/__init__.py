"""
PyMTCNN Backend Implementations

Supports multiple hardware acceleration backends:
- CoreML: Apple Neural Engine (macOS/iOS)
- ONNX: CUDA (NVIDIA), DirectML (Windows), CPU (all platforms)
"""

# Lazy imports to avoid failing on platforms where some backends aren't available
# CoreML is only available on macOS, ONNX requires onnxruntime

__all__ = ["CoreMLMTCNN", "ONNXMTCNN"]


def __getattr__(name):
    """Lazy import backends to avoid import errors on unsupported platforms."""
    if name == "CoreMLMTCNN":
        from .coreml_backend import CoreMLMTCNN
        return CoreMLMTCNN
    elif name == "ONNXMTCNN":
        from .onnx_backend import ONNXMTCNN
        return ONNXMTCNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
