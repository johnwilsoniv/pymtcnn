# pymtcnn

MTCNN face detection with CoreML (Apple Silicon) and CUDA support.

## Installation

```bash
pip install pymtcnn
```

## Usage

```python
from pymtcnn import MTCNN

detector = MTCNN()  # auto-selects best backend
boxes, landmarks = detector.detect(image)
```

## What it does

- Detects faces and 5-point facial landmarks
- Auto-selects backend: CoreML on Mac, CUDA on NVIDIA, CPU fallback
- ~34 FPS on Apple Silicon, ~50 FPS on CUDA

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

CC BY-NC 4.0 — free for non-commercial use with attribution.
