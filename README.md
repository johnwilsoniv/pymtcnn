# PyMTCNN

High-performance **cross-platform** MTCNN face detection with CUDA and Apple Neural Engine support.

## Overview

PyMTCNN is a pure Python implementation of MTCNN (Multi-task Cascaded Convolutional Networks) with multi-backend support for optimal performance across different hardware platforms. It achieves **175.7x speedup** over baseline Python implementations while maintaining **pixel-perfect accuracy** with the C++ OpenFace reference implementation.

### Key Features

- **Cross-Platform**: Works on Windows, Linux, and macOS
- **Multi-Backend**: Auto-selects best backend (CoreML, CUDA, or CPU)
- **High Performance**:
  - Apple Silicon (CoreML): 34+ FPS
  - NVIDIA GPUs (CUDA): 50+ FPS
  - CPU fallback: 5-10 FPS
- **Pixel-Perfect Accuracy**: < 0.001 px error vs C++ OpenFace reference
- **Easy to Use**: Simple, unified Python API
- **Hardware Accelerated**: Leverages Apple Neural Engine or NVIDIA CUDA
- **Flexible**: Single-frame or batch processing modes
- **Production Ready**: Optimized for real-time video analysis

### Performance

| Backend | Hardware | FPS | ms/frame |
|---------|----------|-----|----------|
| CoreML | Apple M1/M2/M3 | 34.26 | 29.2 |
| ONNX+CUDA | NVIDIA RTX 3090 | 50+ | <20 |
| ONNX+CPU | Intel/AMD CPU | 5-10 | 100-200 |

**Speedup**: 175.7x faster than baseline Python implementation

## Requirements

- **Python**: 3.8 or later
- **OS**: macOS, Windows, or Linux
- **Hardware** (one of):
  - Apple Silicon (M1, M2, M3) for CoreML
  - NVIDIA GPU with CUDA for GPU acceleration
  - Any CPU for CPU fallback

## Installation

### From PyPI (Recommended)

Choose the installation that matches your hardware:

#### macOS with Apple Silicon
```bash
pip install pymtcnn[coreml]
```

#### NVIDIA GPU (CUDA)
```bash
pip install pymtcnn[onnx-gpu]
```

#### CPU only
```bash
pip install pymtcnn[onnx]
```

#### All backends (development)
```bash
pip install pymtcnn[all]
```

### From Source

```bash
git clone https://github.com/johnwilsoniv/pymtcnn.git
cd pymtcnn
pip install -e .[coreml]  # or [onnx-gpu] or [onnx]
```

## Quick Start

### Auto-Backend Selection (Recommended)

PyMTCNN automatically selects the best available backend:

```python
import cv2
from pymtcnn import MTCNN

# Auto-select best backend (CoreML on Mac, CUDA on NVIDIA, CPU fallback)
detector = MTCNN(verbose=True)  # Shows which backend was selected

# Load image
img = cv2.imread("image.jpg")

# Detect faces
bboxes, landmarks = detector.detect(img)

# Process results
print(f"Detected {len(bboxes)} faces")
for i, bbox in enumerate(bboxes):
    x, y, w, h = bbox[:4]
    print(f"Face {i+1}: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f}")
```

### Force Specific Backend

```python
from pymtcnn import MTCNN

# Force CoreML (Apple Neural Engine)
detector = MTCNN(backend='coreml')

# Force ONNX (auto-selects CUDA if available)
detector = MTCNN(backend='onnx')

# Force CUDA (NVIDIA GPU)
detector = MTCNN(backend='cuda')

# Force CPU
detector = MTCNN(backend='cpu')
```

### Visualize Detections

```python
import cv2
from pymtcnn import MTCNN

detector = MTCNN()
img = cv2.imread("image.jpg")
bboxes, landmarks = detector.detect(img)

# Draw bounding boxes
for bbox in bboxes:
    x, y, w, h = bbox[:4].astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw landmarks
for lm in landmarks:
    for point in lm:
        cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

cv2.imwrite("output.jpg", img)
```

### Batch Video Processing

```python
import cv2
from pymtcnn import MTCNN

# Initialize detector
detector = MTCNN()

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

### Advanced: Direct Backend Access

For advanced users who need backend-specific features:

```python
from pymtcnn import CoreMLMTCNN, ONNXMTCNN

# Use CoreML directly
coreml_detector = CoreMLMTCNN(verbose=True)

# Use ONNX directly with specific provider
onnx_detector = ONNXMTCNN(provider='cuda', verbose=True)
```

## API Reference

### `MTCNN`

Unified face detector class with automatic backend selection.

#### Constructor

```python
MTCNN(
    backend='auto',      # 'auto', 'coreml', 'onnx', 'cuda', 'cpu'
    model_dir=None,      # Custom model directory (default: bundled)
    verbose=False        # Print backend selection info
)
```

**Parameters:**

- `backend` (str): Backend to use. Options:
  - `'auto'`: Auto-select best available (default)
  - `'coreml'`: Force CoreML (Apple Neural Engine)
  - `'onnx'`: Force ONNX (auto-selects CUDA/CPU)
  - `'cuda'`: Force ONNX with CUDA
  - `'cpu'`: Force ONNX with CPU
- `model_dir` (str): Path to models directory. Default: bundled models
- `verbose` (bool): Print initialization info. Default: False

#### Methods

##### `detect(image)`

Detect faces in a single image.

**Parameters:**
- `image` (numpy.ndarray): Input image (BGR format, H×W×3)

**Returns:**
- `bboxes` (numpy.ndarray): Bounding boxes (N×4), format: [x, y, width, height]
- `landmarks` (numpy.ndarray): Facial landmarks (N×5×2), 5 points per face: left eye, right eye, nose, left mouth, right mouth

##### `detect_batch(frames)`

Detect faces in multiple frames.

**Parameters:**
- `frames` (list): List of images (each BGR format, H×W×3)

**Returns:**
- `results` (list): List of (bboxes, landmarks) tuples, one per frame

##### `get_backend_info()`

Get information about the active backend.

**Returns:**
- `info` (dict): Dictionary with 'backend' and 'provider' keys

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

PyMTCNN achieves **pixel-perfect accuracy** compared to the C++ OpenFace reference implementation:

- **Bounding Box Error**: < 0.001 px (effectively zero)
- **Landmark Error**: < 0.001 px (effectively zero)
- **Detection Agreement**: 100% (identical faces detected)
- **Validation**: Tested on multiple real-world images

Both bounding boxes and 5-point facial landmarks (eyes, nose, mouth corners) match the C++ implementation exactly.

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
