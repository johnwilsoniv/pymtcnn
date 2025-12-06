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
import time
import cv2

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

    # ==================== ONNX Inference Methods ====================

    def _run_pnet(self, img_data):
        """Run PNet using ONNX model. Input: (C, H, W), Output: (1, 6, H', W')"""
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)
        return self.pnet.run(None, {'input': img_batch})[0]

    def _run_rnet(self, img_data):
        """Run RNet using ONNX model. Input: (C, H, W), Output: (6,)"""
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)
        return self.rnet.run(None, {'input': img_batch})[0][0]

    def _run_onet(self, img_data):
        """Run ONet using ONNX model. Input: (C, H, W), Output: (16,)"""
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)
        return self.onet.run(None, {'input': img_batch})[0][0]

    # ==================== Shared Pipeline Stages ====================

    def _extract_crop(self, img_float, box, target_size):
        """
        Extract and resize crop from image matching C++ MTCNN extraction.

        C++ uses (x-1, y-1) start and (w+1, h+1) buffer for proper edge handling.

        Args:
            img_float: Float32 image (H, W, 3)
            box: Bounding box [x1, y1, x2, y2, score, ...]
            target_size: Output size (e.g., 24 for RNet, 48 for ONet)

        Returns:
            Preprocessed crop (C, H, W) or None if invalid
        """
        img_h, img_w = img_float.shape[:2]

        box_x = box[0]
        box_y = box[1]
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]

        width_target = int(box_w + 1)
        height_target = int(box_h + 1)

        # C++ uses x-1, y-1 as extraction start
        start_x_in = max(int(box_x - 1), 0)
        start_y_in = max(int(box_y - 1), 0)
        end_x_in = min(int(box_x + width_target - 1), img_w)
        end_y_in = min(int(box_y + height_target - 1), img_h)

        # Output buffer offsets (for edge cases)
        start_x_out = max(int(-box_x + 1), 0)
        start_y_out = max(int(-box_y + 1), 0)

        if end_x_in <= start_x_in or end_y_in <= start_y_in:
            return None

        # Create zero-padded buffer of size (w+1, h+1)
        tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)

        # Copy image region to buffer
        copy_h = end_y_in - start_y_in
        copy_w = end_x_in - start_x_in
        tmp[start_y_out:start_y_out+copy_h, start_x_out:start_x_out+copy_w] = \
            img_float[start_y_in:end_y_in, start_x_in:end_x_in]

        # Resize and preprocess
        face = cv2.resize(tmp, (target_size, target_size))
        return self._preprocess(face, flip_bgr_to_rgb=True)

    def _pnet_stage(self, img_float):
        """
        Run PNet stage: pyramid processing, box generation, NMS.

        Args:
            img_float: Float32 image (H, W, 3)

        Returns:
            total_boxes: (N, 9) array after PNet or None if no detections
        """
        img_h, img_w = img_float.shape[:2]

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

        if len(total_boxes) == 0:
            return None

        total_boxes = np.vstack(total_boxes)

        # NMS across scales
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        if total_boxes.shape[0] == 0:
            return None

        # Apply PNet bbox regression
        total_boxes = self._apply_bbox_regression(total_boxes)
        return total_boxes

    def _rnet_stage(self, img_float, total_boxes):
        """
        Run RNet stage: crop extraction, scoring, filtering, NMS, regression.

        Args:
            img_float: Float32 image (H, W, 3)
            total_boxes: (N, 9) boxes from PNet

        Returns:
            total_boxes: (N, 9) array after RNet or None if no detections
        """
        total_boxes = self._square_bbox(total_boxes)

        rnet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            crop = self._extract_crop(img_float, total_boxes[i], 24)
            if crop is not None:
                rnet_input.append(crop)
                valid_indices.append(i)

        if len(rnet_input) == 0:
            return None

        total_boxes = total_boxes[valid_indices]

        # Run RNet (sequential for ONNX)
        rnet_outputs = [self._run_rnet(face_data) for face_data in rnet_input]
        output = np.vstack(rnet_outputs)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # Filter by threshold
        keep = scores > self.thresholds[1]
        if not keep.any():
            return None

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        # NMS - update scores BEFORE NMS (critical for correct box selection)
        total_boxes[:, 4] = scores
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = reg[keep]

        if total_boxes.shape[0] == 0:
            return None

        # Apply RNet regression
        w = total_boxes[:, 2] - total_boxes[:, 0]
        h = total_boxes[:, 3] - total_boxes[:, 1]
        x1 = total_boxes[:, 0].copy()
        y1 = total_boxes[:, 1].copy()

        total_boxes[:, 0] = x1 + reg[:, 0] * w
        total_boxes[:, 1] = y1 + reg[:, 1] * h
        total_boxes[:, 2] = x1 + w + w * reg[:, 2]
        total_boxes[:, 3] = y1 + h + h * reg[:, 3]
        total_boxes[:, 4] = scores

        return total_boxes

    def _onet_stage(self, img_float, total_boxes):
        """
        Run ONet stage: crop extraction, scoring, filtering, regression, landmarks.

        Args:
            img_float: Float32 image (H, W, 3)
            total_boxes: (N, 9) boxes from RNet

        Returns:
            bboxes: (N, 4) array [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
            Or (None, None) if no detections
        """
        total_boxes = self._square_bbox(total_boxes)

        onet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            crop = self._extract_crop(img_float, total_boxes[i], 48)
            if crop is not None:
                onet_input.append(crop)
                valid_indices.append(i)

        if len(onet_input) == 0:
            return None, None

        total_boxes = total_boxes[valid_indices]

        # Run ONet (sequential for ONNX)
        onet_outputs = [self._run_onet(face_data) for face_data in onet_input]
        output = np.vstack(onet_outputs)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # Filter by threshold
        keep = scores > self.thresholds[2]
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]
        landmarks = output[keep, 6:16]

        if total_boxes.shape[0] == 0:
            return None, None

        # Apply ONet regression (with +1)
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        x1 = total_boxes[:, 0].copy()
        y1 = total_boxes[:, 1].copy()

        total_boxes[:, 0] = x1 + reg[:, 0] * w
        total_boxes[:, 1] = y1 + reg[:, 1] * h
        total_boxes[:, 2] = x1 + w + w * reg[:, 2]
        total_boxes[:, 3] = y1 + h + h * reg[:, 3]
        total_boxes[:, 4] = scores

        # Reshape landmarks: [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4] -> (N, 5, 2)
        landmarks = np.stack([landmarks[:, 0:5], landmarks[:, 5:10]], axis=2)

        # Final NMS
        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        # Denormalize landmarks using raw bbox dimensions
        w = (total_boxes[:, 2] - total_boxes[:, 0]).reshape(-1, 1)
        h = (total_boxes[:, 3] - total_boxes[:, 1]).reshape(-1, 1)
        x1 = total_boxes[:, 0].reshape(-1, 1)
        y1 = total_boxes[:, 1].reshape(-1, 1)
        landmarks[:, :, 0] = x1 + landmarks[:, :, 0] * w
        landmarks[:, :, 1] = y1 + landmarks[:, :, 1] * h

        # Convert to (x, y, width, height) format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        return bboxes, landmarks

    # ==================== Public Detection Methods ====================

    def detect(self, img):
        """
        Detect faces in image using MTCNN cascade.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
        """
        img_float = img.astype(np.float32)

        # Stage 1: PNet
        total_boxes = self._pnet_stage(img_float)
        if total_boxes is None:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Stage 2: RNet
        total_boxes = self._rnet_stage(img_float, total_boxes)
        if total_boxes is None:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Stage 3: ONet
        bboxes, landmarks = self._onet_stage(img_float, total_boxes)
        if bboxes is None:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        return bboxes, landmarks

    def detect_with_debug(self, img):
        """
        Detect faces with debug info capturing stage-by-stage outputs.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
            debug_info: Dict with stage-by-stage outputs
        """
        debug_info = {}
        img_float = img.astype(np.float32)

        # Stage 1: PNet
        t0 = time.time()
        total_boxes = self._pnet_stage(img_float)
        pnet_time = (time.time() - t0) * 1000

        if total_boxes is None:
            debug_info['pnet'] = {'num_boxes': 0, 'time_ms': pnet_time}
            debug_info['rnet'] = {'num_boxes': 0, 'time_ms': 0.0}
            debug_info['onet'] = {'num_boxes': 0, 'time_ms': 0.0}
            debug_info['final'] = {'num_boxes': 0, 'total_time_ms': pnet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        debug_info['pnet'] = {
            'num_boxes': total_boxes.shape[0],
            'boxes': total_boxes[:, :4].copy(),
            'time_ms': pnet_time
        }

        # Stage 2: RNet
        t0 = time.time()
        total_boxes = self._rnet_stage(img_float, total_boxes)
        rnet_time = (time.time() - t0) * 1000

        if total_boxes is None:
            debug_info['rnet'] = {'num_boxes': 0, 'time_ms': rnet_time}
            debug_info['onet'] = {'num_boxes': 0, 'time_ms': 0.0}
            debug_info['final'] = {'num_boxes': 0, 'total_time_ms': pnet_time + rnet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        # Convert to [x, y, w, h] for debug
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
        bboxes, landmarks = self._onet_stage(img_float, total_boxes)
        onet_time = (time.time() - t0) * 1000

        if bboxes is None:
            debug_info['onet'] = {'num_boxes': 0, 'time_ms': onet_time}
            debug_info['final'] = {'num_boxes': 0, 'total_time_ms': pnet_time + rnet_time + onet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

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
