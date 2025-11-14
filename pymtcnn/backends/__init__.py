"""
PyMTCNN Backend Implementations

Supports multiple hardware acceleration backends:
- CoreML: Apple Neural Engine (macOS/iOS)
- ONNX: CUDA (NVIDIA), DirectML (Windows), CPU (all platforms)
"""

from .coreml_backend import CoreMLMTCNN
from .onnx_backend import ONNXMTCNN

__all__ = ["CoreMLMTCNN", "ONNXMTCNN"]
