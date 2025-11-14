# PyMTCNN

High-performance MTCNN face detection optimized for Apple Neural Engine, achieving **34.26 FPS** on Apple Silicon.

## Overview

PyMTCNN is a pure Python implementation of MTCNN (Multi-task Cascaded Convolutional Networks) that leverages CoreML and Apple's Neural Engine for hardware-accelerated face detection. It achieves **175.7x speedup** over baseline Python implementations while maintaining 95% IoU accuracy.

### Key Features

- **High Performance**: 34.26 FPS with batch processing on Apple Silicon
- **Accurate**: 95% IoU agreement with C++ OpenFace baseline
- **Easy to Use**: Simple, clean Python API
- **Hardware Accelerated**: Leverages Apple Neural Engine (ANE)
- **Flexible**: Single-frame or batch processing modes
- **Production Ready**: Optimized for real-time video analysis

### Performance

| Method | FPS | ms/frame | Use Case |
|--------|-----|----------|----------|
| `detect()` | 31.88 | 31.4 | Single-frame real-time |
| `detect_batch(4)` | 34.26 | 29.2 | Batch video processing |

**Speedup**: 175.7x faster than baseline Python implementation

## Requirements

- **macOS**: macOS 13.0 or later
- **Hardware**: Apple Silicon (M1, M2, M3) recommended
- **Python**: 3.8 or later

## Installation

### From Source

```bash
git clone https://github.com/your-org/PyMTCNN.git
cd PyMTCNN
pip install -e .
```

### From PyPI (Coming Soon)

```bash
pip install pymtcnn
```

## Quick Start

### Single Frame Detection

```python
import cv2
from pymtcnn import CoreMLMTCNN

# Initialize detector
detector = CoreMLMTCNN()

# Load image
img = cv2.imread("image.jpg")

# Detect faces
bboxes, landmarks = detector.detect(img)

# Process results
print(f"Detected {len(bboxes)} faces")
for i, bbox in enumerate(bboxes):
    x, y, w, h, conf = bbox
    print(f"Face {i+1}: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f} (confidence: {conf:.3f})")
```

### Batch Video Processing

```python
import cv2
from pymtcnn import CoreMLMTCNN

# Initialize detector
detector = CoreMLMTCNN()

# Load video frames
cap = cv2.VideoCapture("video.mp4")
frames = []
for _ in range(4):  # Process 4 frames at a time
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

# Batch detection (cross-frame batching for maximum throughput)
results = detector.detect_batch(frames)

# Process results
for i, (bboxes, landmarks) in enumerate(results):
    print(f"Frame {i+1}: {len(bboxes)} faces detected")
```

## API Reference

### `CoreMLMTCNN`

Main face detector class.

#### Constructor

```python
CoreMLMTCNN(
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    coreml_dir=None,
    verbose=False
)
```

**Parameters:**

- `min_face_size` (int): Minimum face size in pixels. Default: 60
- `thresholds` (list): Detection thresholds for [PNet, RNet, ONet]. Default: [0.6, 0.7, 0.7]
- `factor` (float): Image pyramid scale factor. Default: 0.709
- `coreml_dir` (str): Path to CoreML models directory. Default: bundled models
- `verbose` (bool): Enable verbose logging. Default: False

#### Methods

##### `detect(image)`

Detect faces in a single image using within-frame batching.

**Parameters:**
- `image` (numpy.ndarray): Input image (BGR format, H×W×3)

**Returns:**
- `bboxes` (numpy.ndarray): Bounding boxes (N×5), format: [x, y, w, h, confidence]
- `landmarks` (numpy.ndarray): Facial landmarks (N×5×2), 5 points per face: left eye, right eye, nose, left mouth, right mouth

**Performance:** 31.88 FPS (31.4 ms/frame)

##### `detect_batch(frames)`

Detect faces in multiple frames using cross-frame batching.

**Parameters:**
- `frames` (list): List of images (each BGR format, H×W×3)

**Returns:**
- `results` (list): List of (bboxes, landmarks) tuples, one per frame

**Performance:** 34.26 FPS (29.2 ms/frame) with batch_size=4

**Recommended batch size:** 4 frames for optimal throughput

## Performance Guide

### When to Use Each Method

- **`detect()`**: Use for real-time per-frame processing, webcam feeds, or when you need lowest latency
- **`detect_batch()`**: Use for offline batch video processing, maximum throughput, or when processing multiple frames simultaneously

### Optimization Tips

1. **Batch Size**: Use 4 frames for optimal throughput
   - Larger batches (8, 16) are slower due to overhead

2. **Frame Resolution**: Performance tested on 1920×1080
   - Lower resolution → faster processing
   - Higher resolution → more candidates, may require batch splitting

3. **Min Face Size**: Increase `min_face_size` for better performance
   - Default: 60 pixels
   - 80-100 pixels: 1.2-1.5x faster (may miss smaller faces)

## Examples

See the `examples/` directory for complete examples:

- `single_frame_detection.py`: Basic single-frame face detection
- `batch_processing.py`: Batch video processing
- `s1_integration_example.py`: Integration with S1 video pipeline

## Accuracy

PyMTCNN maintains high accuracy while achieving exceptional performance:

- **Mean IoU**: 95% vs C++ OpenFace baseline
- **Detection Agreement**: 100% (same faces detected)
- **Validation**: Tested on 30 frames from real-world patient videos

## Architecture

PyMTCNN uses a three-stage cascade architecture:

1. **PNet** (Proposal Network): Fast candidate generation using image pyramid
2. **RNet** (Refinement Network): Candidate refinement with batching
3. **ONet** (Output Network): Final bbox regression and landmark prediction

All networks are converted to CoreML FP32 format with flexible batch dimensions (1-50) for optimal ANE utilization.

## Optimization Journey

PyMTCNN achieved a **175.7x speedup** through multiple optimization phases:

| Phase | Implementation | FPS | Speedup | Status |
|-------|---------------|-----|---------|--------|
| Baseline | Pure Python CNN | 0.195 | 1.0x | ✅ |
| Phase 1 | Vectorized NumPy | 0.910 | 4.7x | ✅ |
| Phase 2 | ONNX Runtime CPU | 5.870 | 30.1x | ✅ |
| Phase 3 | CoreML + ANE | 13.56 | 69.5x | ✅ |
| Phase 4 | Within-Frame Batching | 31.88 | 163.5x | ✅ |
| Phase 5 | Cross-Frame Batching | **34.26** | **175.7x** | ✅ |

See `docs/OPTIMIZATION_JOURNEY.md` for the complete story.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

**You are free to:**
- Share: Copy and redistribute the material
- Adapt: Remix, transform, and build upon the material

**Under the following terms:**
- Attribution: You must give appropriate credit
- NonCommercial: You may not use the material for commercial purposes

See [LICENSE](LICENSE) for full terms.

## Citation

If you use PyMTCNN in your research, please cite:

```bibtex
@software{pymtcnn2025,
  title={PyMTCNN: High-Performance MTCNN Face Detection for Apple Silicon},
  author={SplitFace},
  year={2025},
  url={https://github.com/your-org/PyMTCNN}
}
```

## Acknowledgments

- Original MTCNN paper: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- C++ OpenFace implementation: Tadas Baltrušaitis et al.
- Apple Neural Engine optimization insights from the CoreML community

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/your-org/PyMTCNN).
