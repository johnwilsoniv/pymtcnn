#!/usr/bin/env python3
"""
ONNX MTCNN Backend

Cross-platform face detection using ONNX Runtime.
Supports CUDA, CoreML EP, and CPU execution providers.
"""

import numpy as np
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..base import MTCNNBase


class ONNXMTCNN(MTCNNBase):
    """
    ONNX Runtime MTCNN face detector.

    Supports multiple execution providers:
    - CUDA (NVIDIA GPUs)
    - CoreML (macOS)
    - CPU (fallback)

    Input: BGR images (OpenCV format) - automatically converted to RGB internally.
    """

    def __init__(self, model_dir=None, provider=None, verbose=False):
        """
        Initialize ONNX MTCNN detector.

        Args:
            model_dir: Directory containing .onnx models
            provider: Execution provider ('cuda', 'coreml', 'cpu', or None for auto)
            verbose: Print initialization info
        """
        super().__init__()

        if ort is None:
            raise ImportError(
                "ONNX backend requires onnxruntime. Install with:\n"
                "  pip install onnxruntime  # CPU\n"
                "  pip install onnxruntime-gpu  # CUDA"
            )

        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models"
            )

        if verbose:
            print(f"Loading ONNX models from: {model_dir}")

        # Configure execution providers
        providers = self._get_providers(provider, verbose)

        # Create ONNX sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.pnet = ort.InferenceSession(
            os.path.join(model_dir, "pnet.onnx"),
            sess_options=sess_options,
            providers=providers
        )
        self.rnet = ort.InferenceSession(
            os.path.join(model_dir, "rnet.onnx"),
            sess_options=sess_options,
            providers=providers
        )
        self.onet = ort.InferenceSession(
            os.path.join(model_dir, "onet.onnx"),
            sess_options=sess_options,
            providers=providers
        )

        # Store active provider
        self._active_provider = self.pnet.get_providers()[0]

        if verbose:
            print(f"ONNX models loaded with provider: {self._active_provider}")

    def _get_providers(self, provider, verbose):
        """Get list of execution providers based on preference."""
        available = ort.get_available_providers()

        if provider == 'cuda':
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                raise RuntimeError("CUDA provider requested but not available")

        elif provider == 'coreml':
            if 'CoreMLExecutionProvider' in available:
                return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            else:
                raise RuntimeError("CoreML provider requested but not available")

        elif provider == 'cpu':
            return ['CPUExecutionProvider']

        else:
            # Auto-select best available
            providers = []
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            if 'CoreMLExecutionProvider' in available:
                providers.append('CoreMLExecutionProvider')
            providers.append('CPUExecutionProvider')

            if verbose:
                print(f"Available providers: {available}")
                print(f"Using providers: {providers}")

            return providers

    # ==================== Network Execution ====================

    def _run_pnet(self, img_data):
        """Run PNet on preprocessed image."""
        input_data = img_data[np.newaxis, :, :, :].astype(np.float32)
        result = self.pnet.run(None, {'input': input_data})
        return result[0]

    def _run_rnet_batch(self, img_data_list):
        """
        Run RNet on batch of crops.

        Args:
            img_data_list: List of preprocessed crops (C, H, W)

        Returns:
            (N, 6) array of RNet outputs
        """
        outputs = []
        for img_data in img_data_list:
            img_batch = img_data[np.newaxis, :, :, :].astype(np.float32)
            result = self.rnet.run(None, {'input': img_batch})
            outputs.append(result[0][0])
        return np.vstack(outputs)

    def _run_onet_batch(self, img_data_list):
        """
        Run ONet on batch of crops.

        Args:
            img_data_list: List of preprocessed crops (C, H, W)

        Returns:
            (N, 16) array of ONet outputs
        """
        outputs = []
        for img_data in img_data_list:
            img_batch = img_data[np.newaxis, :, :, :].astype(np.float32)
            result = self.onet.run(None, {'input': img_batch})
            outputs.append(result[0][0])
        return np.vstack(outputs)

    # ==================== Public Methods ====================

    def get_active_provider(self):
        """Return the active execution provider."""
        return self._active_provider
