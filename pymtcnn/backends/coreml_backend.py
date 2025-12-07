#!/usr/bin/env python3
"""
CoreML MTCNN Backend

High-performance face detection using Apple Neural Engine via CoreML.
Achieves ~34 FPS on Apple Silicon.
"""

import numpy as np
import os

try:
    import coremltools as ct
except ImportError:
    ct = None

from ..base import MTCNNBase


class CoreMLMTCNN(MTCNNBase):
    """
    CoreML-accelerated MTCNN face detector.

    Uses Apple Neural Engine for fast inference on macOS.
    Input: BGR images (OpenCV format) - automatically converted to RGB internally.
    """

    def __init__(self, coreml_dir=None, verbose=False):
        """
        Initialize CoreML MTCNN detector.

        Args:
            coreml_dir: Directory containing .mlpackage models
            verbose: Print initialization info
        """
        super().__init__()

        if ct is None:
            raise ImportError(
                "CoreML backend requires coremltools. Install with:\n"
                "  pip install coremltools"
            )

        if coreml_dir is None:
            coreml_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "models"
            )

        if verbose:
            print(f"Loading CoreML models from: {coreml_dir}")

        # Load CoreML models
        self.pnet = ct.models.MLModel(
            os.path.join(coreml_dir, "pnet_fp32.mlpackage"),
            compute_units=ct.ComputeUnit.ALL
        )
        self.rnet = ct.models.MLModel(
            os.path.join(coreml_dir, "rnet_fp32.mlpackage"),
            compute_units=ct.ComputeUnit.ALL
        )
        self.onet = ct.models.MLModel(
            os.path.join(coreml_dir, "onet_fp32.mlpackage"),
            compute_units=ct.ComputeUnit.ALL
        )

        # Get output names from model specs
        self._pnet_output = self.pnet.get_spec().description.output[0].name
        self._rnet_output = self.rnet.get_spec().description.output[0].name
        self._onet_output = self.onet.get_spec().description.output[0].name

        if verbose:
            print("CoreML models loaded successfully")

    # ==================== Network Execution ====================

    def _run_pnet(self, img_data):
        """Run PNet on preprocessed image."""
        input_data = img_data[np.newaxis, :, :, :].astype(np.float32)
        result = self.pnet.predict({'input': input_data})
        return result[self._pnet_output]

    def _run_rnet_batch(self, img_data_list, max_batch_size=50):
        """
        Run RNet on batch of crops with batching support.

        Args:
            img_data_list: List of preprocessed crops (C, H, W)
            max_batch_size: Maximum batch size for CoreML

        Returns:
            (N, 6) array of RNet outputs
        """
        outputs = []

        for i in range(0, len(img_data_list), max_batch_size):
            batch = img_data_list[i:i + max_batch_size]
            batch_array = np.stack(batch, axis=0).astype(np.float32)
            result = self.rnet.predict({'input': batch_array})
            outputs.append(result[self._rnet_output])

        return np.vstack(outputs)

    def _run_onet_batch(self, img_data_list, max_batch_size=50):
        """
        Run ONet on batch of crops with batching support.

        Args:
            img_data_list: List of preprocessed crops (C, H, W)
            max_batch_size: Maximum batch size for CoreML

        Returns:
            (N, 16) array of ONet outputs
        """
        outputs = []

        for i in range(0, len(img_data_list), max_batch_size):
            batch = img_data_list[i:i + max_batch_size]
            batch_array = np.stack(batch, axis=0).astype(np.float32)
            result = self.onet.predict({'input': batch_array})
            outputs.append(result[self._onet_output])

        return np.vstack(outputs)

    # ==================== Public Methods ====================

    def get_active_provider(self):
        """Return the active compute provider."""
        return "CoreML (Apple Neural Engine)"
