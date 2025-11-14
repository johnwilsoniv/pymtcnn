"""
PyMTCNN - High-Performance MTCNN Face Detection

Unified detector with automatic backend selection:
- CoreML (Apple Neural Engine) on macOS - 34 FPS
- ONNX + CUDA on NVIDIA GPUs - 50+ FPS
- ONNX + CPU fallback - 5-10 FPS

Usage:
    from pymtcnn import MTCNN

    # Auto-select best backend
    detector = MTCNN()

    # Or specify backend explicitly
    detector = MTCNN(backend='coreml')  # Force CoreML
    detector = MTCNN(backend='onnx')    # Force ONNX
    detector = MTCNN(backend='cuda')    # Force ONNX with CUDA
"""

import platform
import warnings


class MTCNN:
    """
    Unified MTCNN face detector with automatic backend selection.

    Automatically selects the best available backend:
    1. CoreML (if on macOS and coremltools available)
    2. ONNX with CUDA (if NVIDIA GPU available)
    3. ONNX with CPU (fallback)
    """

    def __init__(self, backend=None, model_dir=None, verbose=False, **kwargs):
        """
        Initialize MTCNN detector with automatic backend selection.

        Args:
            backend: Backend to use ('auto', 'coreml', 'onnx', 'cuda', 'cpu')
                    Default: 'auto' (automatically selects best available)
            model_dir: Directory containing models (default: bundled models)
            verbose: Print initialization and backend selection info
            **kwargs: Additional backend-specific arguments
        """
        self.verbose = verbose
        self.backend_name = None
        self._detector = None

        # Normalize backend name
        if backend is None:
            backend = 'auto'
        backend = backend.lower()

        # Auto-select or validate backend
        if backend == 'auto':
            self._auto_select_backend(model_dir, **kwargs)
        elif backend == 'coreml':
            self._init_coreml_backend(model_dir, **kwargs)
        elif backend in ['onnx', 'cuda', 'cpu']:
            provider = None if backend == 'onnx' else backend
            self._init_onnx_backend(model_dir, provider=provider, **kwargs)
        else:
            raise ValueError(
                f"Invalid backend '{backend}'. "
                f"Must be one of: auto, coreml, onnx, cuda, cpu"
            )

        if self.verbose:
            print(f"✓ PyMTCNN initialized with {self.backend_name} backend")

    def _auto_select_backend(self, model_dir, **kwargs):
        """Automatically select the best available backend."""

        # Priority 1: CoreML on macOS (fastest on Apple Silicon)
        if platform.system() == 'Darwin':
            try:
                self._init_coreml_backend(model_dir, **kwargs)
                if self.verbose:
                    print("Auto-selected: CoreML (Apple Neural Engine)")
                return
            except ImportError:
                if self.verbose:
                    print("CoreML not available (coremltools not installed)")
            except Exception as e:
                if self.verbose:
                    print(f"CoreML initialization failed: {e}")

        # Priority 2: ONNX (with automatic CUDA/CPU selection)
        try:
            self._init_onnx_backend(model_dir, provider=None, **kwargs)
            if self.verbose:
                provider = self._detector.get_active_provider()
                if 'CUDA' in provider:
                    print("Auto-selected: ONNX with CUDA")
                elif 'CoreML' in provider:
                    print("Auto-selected: ONNX with CoreML Execution Provider")
                else:
                    print("Auto-selected: ONNX with CPU")
            return
        except ImportError:
            raise RuntimeError(
                "No compatible backend found. Please install:\n"
                "  - macOS: pip install pymtcnn[coreml]\n"
                "  - NVIDIA GPU: pip install pymtcnn[onnx-gpu]\n"
                "  - CPU: pip install pymtcnn[onnx]"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize any backend: {e}")

    def _init_coreml_backend(self, model_dir, **kwargs):
        """Initialize CoreML backend."""
        try:
            from .backends.coreml_backend import CoreMLMTCNN
            self._detector = CoreMLMTCNN(
                coreml_dir=model_dir,
                verbose=self.verbose,
                **kwargs
            )
            self.backend_name = "CoreML"
        except ImportError as e:
            raise ImportError(
                "CoreML backend requires coremltools. Install with:\n"
                "  pip install pymtcnn[coreml]"
            ) from e

    def _init_onnx_backend(self, model_dir, provider=None, **kwargs):
        """Initialize ONNX backend."""
        try:
            from .backends.onnx_backend import ONNXMTCNN
            self._detector = ONNXMTCNN(
                model_dir=model_dir,
                provider=provider,
                verbose=self.verbose,
                **kwargs
            )
            active_provider = self._detector.get_active_provider()

            if 'CUDA' in active_provider:
                self.backend_name = "ONNX + CUDA"
            elif 'CoreML' in active_provider:
                self.backend_name = "ONNX + CoreML"
            else:
                self.backend_name = "ONNX + CPU"

        except ImportError as e:
            raise ImportError(
                "ONNX backend requires onnxruntime. Install with:\n"
                "  GPU: pip install pymtcnn[onnx-gpu]\n"
                "  CPU: pip install pymtcnn[onnx]"
            ) from e

    def detect(self, img):
        """
        Detect faces in a single image.

        Args:
            img: Input image (H, W, 3) in BGR format (OpenCV format)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
        """
        return self._detector.detect(img)

    def detect_batch(self, frames):
        """
        Detect faces across multiple frames (cross-frame batching).

        Args:
            frames: List of images, each (H, W, 3) in BGR format

        Returns:
            List of (bboxes, landmarks) tuples, one per frame
        """
        if hasattr(self._detector, 'detect_batch'):
            return self._detector.detect_batch(frames)
        else:
            # Fallback: process individually
            return [self.detect(frame) for frame in frames]

    def get_backend_info(self):
        """Get information about the active backend."""
        info = {
            'backend': self.backend_name,
        }

        if hasattr(self._detector, 'get_active_provider'):
            info['provider'] = self._detector.get_active_provider()

        return info

    def __repr__(self):
        return f"MTCNN(backend={self.backend_name})"


# Backward compatibility: Export CoreML and ONNX backends directly
try:
    from .backends.coreml_backend import CoreMLMTCNN
except ImportError:
    CoreMLMTCNN = None

try:
    from .backends.onnx_backend import ONNXMTCNN
except ImportError:
    ONNXMTCNN = None
