"""
ONNX MTCNN Detector for Cross-Platform Inference

High-performance face detection using ONNX Runtime.
Supports CUDA, CoreML EP, and CPU backends.

Performance:
- CUDA: 50+ FPS on RTX GPUs
- CPU: 5-10 FPS
- CoreML EP: 20-25 FPS (macOS)
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path

from ..base import PurePythonMTCNN_Optimized


class ONNXMTCNN(PurePythonMTCNN_Optimized):
    """
    ONNX-accelerated MTCNN using ONNX Runtime.

    Inherits all pipeline logic from PurePythonMTCNN_Optimized,
    only replaces the CNN forward passes with ONNX inference.
    """

    def __init__(self, model_dir=None, provider=None, verbose=False):
        """
        Initialize ONNX MTCNN detector.

        Args:
            model_dir: Directory containing ONNX models (default: models/)
            provider: Execution provider ('cuda', 'coreml', 'cpu', or None for auto)
            verbose: Print loading messages (default: False)
        """
        # Don't call super().__init__() - we'll load ONNX models instead

        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models"
        else:
            model_dir = Path(model_dir)

        if verbose:
            print(f"Loading ONNX models from {model_dir}...")

        # Determine execution providers
        providers = self._get_providers(provider, verbose)

        # Load ONNX models
        self.pnet = ort.InferenceSession(
            str(model_dir / "pnet.onnx"),
            providers=providers
        )
        self.rnet = ort.InferenceSession(
            str(model_dir / "rnet.onnx"),
            providers=providers
        )
        self.onet = ort.InferenceSession(
            str(model_dir / "onet.onnx"),
            providers=providers
        )

        if verbose:
            active_provider = self.get_active_provider()
            print(f"✓ ONNX MTCNN initialized with {active_provider}")

        # MTCNN parameters (must match C++ baseline)
        self.thresholds = [0.6, 0.7, 0.7]  # PNet, RNet, ONet thresholds
        self.min_face_size = 60
        self.factor = 0.709

    def _get_providers(self, provider, verbose):
        """Determine ONNX Runtime execution providers."""
        available = ort.get_available_providers()

        if provider is None:
            # Auto-select: CUDA > CoreML > CPU
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if verbose:
                    print("Auto-selected: CUDA")
            elif 'CoreMLExecutionProvider' in available:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
                if verbose:
                    print("Auto-selected: CoreML Execution Provider")
            else:
                providers = ['CPUExecutionProvider']
                if verbose:
                    print("Auto-selected: CPU")
        elif provider.lower() == 'cuda':
            if 'CUDAExecutionProvider' not in available:
                raise RuntimeError("CUDA provider requested but not available")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif provider.lower() == 'coreml':
            if 'CoreMLExecutionProvider' not in available:
                raise RuntimeError("CoreML provider requested but not available")
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        elif provider.lower() == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return providers

    def get_active_provider(self):
        """Get the active execution provider."""
        return self.pnet.get_providers()[0]

    def _run_pnet(self, img_data):
        """
        Run PNet using ONNX model.

        Args:
            img_data: Input shape (C, H, W)

        Returns:
            Output shape (1, 6, H', W') - batch dimension added for compatibility
        """
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Run ONNX inference
        output = self.pnet.run(None, {'input': img_batch})[0]

        # Output shape: (1, 6, H', W')
        # Pure Python MTCNN expects (1, 6, H', W')
        return output

    def _run_rnet(self, img_data):
        """
        Run RNet using ONNX model.

        Args:
            img_data: Input shape (C, H, W) - should be (3, 24, 24)

        Returns:
            Output shape (6,) - [cls_not_face, cls_face, dx, dy, dw, dh]
        """
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Run ONNX inference
        output = self.rnet.run(None, {'input': img_batch})[0]

        # Output shape: (1, 6) → (6,)
        return output[0]

    def _run_onet(self, img_data):
        """
        Run ONet using ONNX model.

        Args:
            img_data: Input shape (C, H, W) - should be (3, 48, 48)

        Returns:
            Output shape (16,) - [cls_not_face, cls_face, dx, dy, dw, dh, lm1_x, lm1_y, ...]
        """
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Run ONNX inference
        output = self.onet.run(None, {'input': img_batch})[0]

        # Output shape: (1, 16) → (16,)
        return output[0]

    def detect_with_debug(self, img):
        """
        Detect faces in image with debug info capturing stage-by-stage outputs.

        This method is identical to CoreML's detect_with_debug() to ensure
        identical pipeline behavior.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
            debug_info: Dict with stage-by-stage outputs
        """
        import cv2
        import time
        debug_info = {}

        img_h, img_w = img.shape[:2]
        img_float = img.astype(np.float32)

        # Build image pyramid
        min_size = self.min_face_size
        m = 12.0 / min_size
        min_l = min(img_h, img_w) * m

        scales = []
        scale = m
        while min_l >= 12:
            scales.append(scale)
            scale *= self.factor
            min_l *= self.factor

        # Stage 1: PNet
        t0 = time.time()
        total_boxes = []

        for scale in scales:
            hs = int(np.ceil(img_h * scale))
            ws = int(np.ceil(img_w * scale))

            img_scaled = cv2.resize(img_float, (ws, hs), interpolation=cv2.INTER_LINEAR)
            img_data = self._preprocess(img_scaled, flip_bgr_to_rgb=True)

            output = self._run_pnet(img_data)
            output = output[0].transpose(1, 2, 0)

            logit_not_face = output[:, :, 0]
            logit_face = output[:, :, 1]
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            score_map = np.stack([1.0 - prob_face, prob_face], axis=2)
            reg_map = output[:, :, 2:6]

            boxes = self._generate_bboxes(score_map, reg_map, scale, self.thresholds[0])

            if boxes.shape[0] > 0:
                keep = self._nms(boxes, 0.5, 'Union')
                boxes = boxes[keep]
                total_boxes.append(boxes)

        if len(total_boxes) > 0:
            total_boxes = np.vstack(total_boxes)
            keep = self._nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[keep]
            total_boxes = self._apply_bbox_regression(total_boxes)
        else:
            total_boxes = np.empty((0, 9))

        pnet_time = (time.time() - t0) * 1000

        # Convert to [x, y, w, h] format for debug
        pnet_boxes_debug = np.zeros((total_boxes.shape[0], 4))
        if total_boxes.shape[0] > 0:
            pnet_boxes_debug[:, 0] = total_boxes[:, 0]
            pnet_boxes_debug[:, 1] = total_boxes[:, 1]
            pnet_boxes_debug[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
            pnet_boxes_debug[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        debug_info['pnet'] = {
            'num_boxes': total_boxes.shape[0],
            'boxes': pnet_boxes_debug.copy(),
            'time_ms': pnet_time
        }

        if total_boxes.shape[0] == 0:
            debug_info['rnet'] = {'num_boxes': 0, 'boxes': np.array([]), 'time_ms': 0.0}
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': 0.0}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        # Stage 2: RNet
        t0 = time.time()
        total_boxes = self._square_bbox(total_boxes)

        rnet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            bbox_x = total_boxes[i, 0]
            bbox_y = total_boxes[i, 1]
            bbox_w = total_boxes[i, 2] - total_boxes[i, 0]
            bbox_h = total_boxes[i, 3] - total_boxes[i, 1]

            width_target = int(bbox_w + 1)
            height_target = int(bbox_h + 1)

            start_x_in = max(int(bbox_x - 1), 0)
            start_y_in = max(int(bbox_y - 1), 0)
            end_x_in = min(int(bbox_x + width_target - 1), img_w)
            end_y_in = min(int(bbox_y + height_target - 1), img_h)

            start_x_out = max(int(-bbox_x + 1), 0)
            start_y_out = max(int(-bbox_y + 1), 0)
            end_x_out = min(int(width_target - (bbox_x + bbox_w - img_w)), width_target)
            end_y_out = min(int(height_target - (bbox_y + bbox_h - img_h)), height_target)

            tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)
            tmp[start_y_out:end_y_out, start_x_out:end_x_out] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            face = cv2.resize(tmp, (24, 24))
            rnet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
            valid_indices.append(i)

        if len(rnet_input) == 0:
            rnet_time = (time.time() - t0) * 1000
            debug_info['rnet'] = {'num_boxes': 0, 'boxes': np.array([]), 'time_ms': rnet_time}
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': 0.0}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time + rnet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        total_boxes = total_boxes[valid_indices]

        # Run RNet
        rnet_outputs = []
        for face_data in rnet_input:
            output = self._run_rnet(face_data)
            rnet_outputs.append(output)

        output = np.vstack(rnet_outputs)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        keep = scores > self.thresholds[1]

        if not keep.any():
            rnet_time = (time.time() - t0) * 1000
            debug_info['rnet'] = {'num_boxes': 0, 'boxes': np.array([]), 'time_ms': rnet_time}
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': 0.0}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time + rnet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = reg[keep]

        if total_boxes.shape[0] == 0:
            rnet_time = (time.time() - t0) * 1000
            debug_info['rnet'] = {'num_boxes': 0, 'boxes': np.array([]), 'time_ms': rnet_time}
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': 0.0}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time + rnet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        w = total_boxes[:, 2] - total_boxes[:, 0]
        h = total_boxes[:, 3] - total_boxes[:, 1]
        x1 = total_boxes[:, 0].copy()
        y1 = total_boxes[:, 1].copy()

        total_boxes[:, 0] = x1 + reg[:, 0] * w
        total_boxes[:, 1] = y1 + reg[:, 1] * h
        total_boxes[:, 2] = x1 + w + w * reg[:, 2]
        total_boxes[:, 3] = y1 + h + h * reg[:, 3]
        total_boxes[:, 4] = scores

        rnet_time = (time.time() - t0) * 1000

        # Convert to [x, y, w, h] format for debug
        rnet_boxes_debug = np.zeros((total_boxes.shape[0], 4))
        rnet_boxes_debug[:, 0] = total_boxes[:, 0]
        rnet_boxes_debug[:, 1] = total_boxes[:, 1]
        rnet_boxes_debug[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        rnet_boxes_debug[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        debug_info['rnet'] = {
            'num_boxes': total_boxes.shape[0],
            'boxes': rnet_boxes_debug.copy(),
            'time_ms': rnet_time
        }

        # Stage 3: ONet
        t0 = time.time()
        total_boxes = self._square_bbox(total_boxes)

        onet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            x1 = int(max(0, total_boxes[i, 0]))
            y1 = int(max(0, total_boxes[i, 1]))
            x2 = int(min(img_w, total_boxes[i, 2]))
            y2 = int(min(img_h, total_boxes[i, 3]))

            if x2 <= x1 or y2 <= y1:
                continue

            face = img_float[y1:y2, x1:x2]
            face = cv2.resize(face, (48, 48))
            onet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
            valid_indices.append(i)

        if len(onet_input) == 0:
            onet_time = (time.time() - t0) * 1000
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': onet_time}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time + rnet_time + onet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        total_boxes = total_boxes[valid_indices]

        # Run ONet
        onet_outputs = []
        for face_data in onet_input:
            output = self._run_onet(face_data)
            onet_outputs.append(output)

        output = np.vstack(onet_outputs)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        keep = scores > self.thresholds[2]

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]
        landmarks = output[keep, 6:16]

        if total_boxes.shape[0] == 0:
            onet_time = (time.time() - t0) * 1000
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': onet_time}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time + rnet_time + onet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        x1 = total_boxes[:, 0].copy()
        y1 = total_boxes[:, 1].copy()

        total_boxes[:, 0] = x1 + reg[:, 0] * w
        total_boxes[:, 1] = y1 + reg[:, 1] * h
        total_boxes[:, 2] = x1 + w + w * reg[:, 2]
        total_boxes[:, 3] = y1 + h + h * reg[:, 3]
        total_boxes[:, 4] = scores

        # ONet outputs landmarks as [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4] (x-first format)
        landmarks = np.stack([landmarks[:, 0:5], landmarks[:, 5:10]], axis=2)

        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        # NO CALIBRATION - output raw bbox matching C++ raw output
        # Calibration is applied downstream in the pipeline

        # Denormalize landmarks using RAW (uncalibrated) bbox dimensions
        # Landmarks are normalized [0,1] relative to the regressed bbox
        for k in range(total_boxes.shape[0]):
            w = total_boxes[k, 2] - total_boxes[k, 0]
            h = total_boxes[k, 3] - total_boxes[k, 1]
            for i in range(5):
                landmarks[k, i, 0] = total_boxes[k, 0] + landmarks[k, i, 0] * w
                landmarks[k, i, 1] = total_boxes[k, 1] + landmarks[k, i, 1] * h

        onet_time = (time.time() - t0) * 1000

        # Convert to [x, y, w, h] format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        debug_info['onet'] = {
            'num_boxes': bboxes.shape[0],
            'boxes': bboxes.copy(),
            'landmarks': landmarks.copy(),
            'time_ms': onet_time
        }

        debug_info['final'] = {
            'num_boxes': bboxes.shape[0],
            'boxes': bboxes.copy(),
            'landmarks': landmarks.copy(),
            'total_time_ms': pnet_time + rnet_time + onet_time
        }

        return bboxes, landmarks, debug_info
