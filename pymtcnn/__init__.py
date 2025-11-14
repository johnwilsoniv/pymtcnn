"""
PyMTCNN - High-Performance MTCNN Face Detection for Apple Silicon

A pure Python + CoreML implementation of MTCNN (Multi-task Cascaded Convolutional Networks)
optimized for Apple Neural Engine, achieving 34.26 FPS on M-series chips.

Example usage:
    from pymtcnn import CoreMLMTCNN

    # Initialize detector
    detector = CoreMLMTCNN(
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709
    )

    # Single-frame detection
    bboxes, landmarks = detector.detect(frame)

    # Batch processing (cross-frame batching)
    results = detector.detect_batch(frames)

Performance:
    - Single-frame: 31.88 FPS (31.4 ms/frame)
    - Batch (4 frames): 34.26 FPS (29.2 ms/frame)
    - Accuracy: 95% IoU vs C++ OpenFace baseline

License:
    CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
"""

__version__ = "1.0.0"
__author__ = "SplitFace"
__license__ = "CC BY-NC 4.0"

from .detector import CoreMLMTCNN

__all__ = ["CoreMLMTCNN"]
