# pymtcnn

MTCNN face detection with CoreML (Apple Silicon) and CUDA support.

## Installation

Installing pymtcnn takes two steps.

**1. Install the package.** Pick the line for your computer:

```bash
pip install "pymtcnn[coreml]"     # Mac
pip install "pymtcnn[onnx]"       # Windows or Linux
pip install "pymtcnn[onnx-gpu]"   # computer with an NVIDIA graphics card
```

**2. Download the face-detection models (once).** Run:

```bash
pymtcnn-download-models
```

The command shows a short summary of the OpenFace license and asks you to type `YES`.
It then downloads OpenFace's original model files (about 2.2 MB) from OpenFace's
official GitHub repository, checks that they are complete and unchanged, and
prepares them for pymtcnn. If your terminal says the command is not found, run
`python -m pymtcnn.download_models` instead.

If you skip step 2, pymtcnn stops with a message that tells you to run this command.
It never downloads anything without your consent.

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

## The models and their license

pymtcnn uses the MTCNN face-detection models from
[OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace). Carnegie Mellon
University licenses OpenFace for **academic or non-profit noncommercial research only**.
The license does not allow sharing the models with others, so pymtcnn does not include
them: each user downloads them directly from OpenFace and accepts the license.
In short:

- Use the models only for your own noncommercial research.
- Do not share, sell or redistribute them, or give others access to them.
- Commercial use needs a separate license from Carnegie Mellon University.

Full license: https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt

**Where the files go.** The models are stored in a folder that pyclnf and pyfaceau
also use, so you only download them once:

- Mac: `~/Library/Application Support/OpenFaceModels/2.2.0`
- Windows: `%LOCALAPPDATA%\OpenFaceModels\2.2.0`
- Linux: `~/.local/share/OpenFaceModels/2.2.0`

To use another folder, set the `OPENFACE_MODELS_DIR` environment variable
(the `2.2.0` folder is created inside it).

**Automated setups** (servers, scripts): to accept the license without being asked, run
`pymtcnn-download-models --accept-license` or set `OPENFACE_MODELS_ACCEPT_LICENSE=1`.
From Python:

```python
import pymtcnn.models
pymtcnn.models.ensure_models(accept_license=True)
```

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

pymtcnn's code: CC BY-NC 4.0 — free for non-commercial use with attribution.
The OpenFace models are covered by the OpenFace license above.
