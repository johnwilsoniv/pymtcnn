#!/usr/bin/env python3
"""
MTCNN Base Classes

Provides shared functionality for all MTCNN backends:
- MTCNNBase: Abstract base with shared pipeline methods
- PurePythonMTCNN: Concrete implementation using pure Python CNN
"""

import numpy as np
import cv2
import os
from abc import ABC, abstractmethod


class MTCNNBase(ABC):
    """
    Abstract base class for MTCNN implementations.

    Provides shared methods for bbox processing, NMS, and preprocessing.
    Subclasses must implement _run_pnet, _run_rnet, _run_onet.
    """

    def __init__(self):
        # MTCNN parameters (matching C++)
        self.thresholds = [0.6, 0.7, 0.7]
        self.min_face_size = 60
        self.factor = 0.709

    # ==================== Abstract Methods (implement in subclass) ====================

    @abstractmethod
    def _run_pnet(self, img_data):
        """Run PNet on preprocessed image data. Returns (1, C, H, W) output."""
        pass

    @abstractmethod
    def _run_rnet_batch(self, img_data_list):
        """Run RNet on batch of crops. Returns (N, 6) output."""
        pass

    @abstractmethod
    def _run_onet_batch(self, img_data_list):
        """Run ONet on batch of crops. Returns (N, 16) output."""
        pass

    # ==================== Shared Preprocessing ====================

    def _preprocess(self, img: np.ndarray, flip_bgr_to_rgb: bool = True) -> np.ndarray:
        """
        Preprocess image for network input.

        Args:
            img: Input image (H, W, 3)
            flip_bgr_to_rgb: Whether to flip BGR to RGB

        Returns:
            Preprocessed image (C, H, W) normalized to [-1, 1]
        """
        img_norm = (img.astype(np.float32) - 127.5) * 0.0078125
        img_chw = np.transpose(img_norm, (2, 0, 1))

        if flip_bgr_to_rgb:
            img_chw = img_chw[[2, 1, 0], :, :]

        return img_chw

    # ==================== Shared Crop Extraction ====================

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

    # ==================== Shared Pipeline Stages ====================

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

        # Run RNet
        output = self._run_rnet_batch(rnet_input)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # Filter by threshold
        keep = scores > self.thresholds[1]
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        if total_boxes.shape[0] == 0:
            return None

        # Update scores for NMS
        total_boxes[:, 4] = scores

        # NMS
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
        Run ONet stage: crop extraction, scoring, landmarks, NMS.

        Args:
            img_float: Float32 image (H, W, 3)
            total_boxes: (N, 9) boxes from RNet

        Returns:
            bboxes: (N, 4) as [x, y, w, h]
            landmarks: (N, 5, 2) facial landmarks
        """
        total_boxes = self._square_bbox(total_boxes)

        # Save squared box dimensions BEFORE filtering - needed for landmark denormalization
        # ONet landmarks are normalized relative to the 48x48 crop from this squared box
        squared_boxes = total_boxes.copy()

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
        squared_boxes = squared_boxes[valid_indices]

        # Run ONet
        output = self._run_onet_batch(onet_input)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # Filter by threshold
        keep = scores > self.thresholds[2]
        total_boxes = total_boxes[keep]
        squared_boxes = squared_boxes[keep]
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
        squared_boxes = squared_boxes[keep]
        landmarks = landmarks[keep]

        # Apply final calibration to match C++ MTCNN output
        # This converts raw ONet bbox to the tight face bbox that C++ outputs
        w = total_boxes[:, 2] - total_boxes[:, 0]
        h = total_boxes[:, 3] - total_boxes[:, 1]
        new_x1 = total_boxes[:, 0] + w * -0.0075
        new_y1 = total_boxes[:, 1] + h * 0.2459
        new_width = w * 1.0323
        new_height = h * 0.7751
        total_boxes[:, 0] = new_x1
        total_boxes[:, 1] = new_y1
        total_boxes[:, 2] = new_x1 + new_width
        total_boxes[:, 3] = new_y1 + new_height

        # Denormalize landmarks using SQUARED box (the ONet input box)
        # ONet outputs normalized landmarks relative to the 48x48 crop from squared box
        sq_x = squared_boxes[:, 0].reshape(-1, 1)
        sq_y = squared_boxes[:, 1].reshape(-1, 1)
        sq_w = (squared_boxes[:, 2] - squared_boxes[:, 0]).reshape(-1, 1)
        sq_h = (squared_boxes[:, 3] - squared_boxes[:, 1]).reshape(-1, 1)
        landmarks[:, :, 0] = sq_x + landmarks[:, :, 0] * sq_w
        landmarks[:, :, 1] = sq_y + landmarks[:, :, 1] * sq_h

        # Convert to (x, y, width, height) format - return RAW bbox (after regression)
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        return bboxes, landmarks

    # ==================== Shared Utility Methods ====================

    def _generate_bboxes(self, score_map, reg_map, scale, threshold):
        """Generate bounding boxes from PNet output (C++ matching)"""
        stride = 2
        cellsize = 12

        t_index = np.where(score_map[:, :, 1] >= threshold)

        if t_index[0].size == 0:
            return np.array([]).reshape(0, 9)

        dx1, dy1, dx2, dy2 = [reg_map[t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = score_map[t_index[0], t_index[1], 1]

        boundingbox = np.vstack([
            np.floor((stride * t_index[1] + 1) / scale).astype(int),
            np.floor((stride * t_index[0] + 1) / scale).astype(int),
            np.floor((stride * t_index[1] + cellsize) / scale).astype(int),
            np.floor((stride * t_index[0] + cellsize) / scale).astype(int),
            score,
            reg
        ])

        return boundingbox.T

    def _nms(self, boxes, threshold, method):
        """Non-Maximum Suppression (C++ matching)"""
        if boxes.shape[0] == 0:
            return np.array([])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1) * (y2 - y1)
        sorted_s = np.argsort(s)

        pick = []
        while sorted_s.shape[0] > 0:
            i = sorted_s[-1]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[sorted_s[:-1]])
            yy1 = np.maximum(y1[i], y1[sorted_s[:-1]])
            xx2 = np.minimum(x2[i], x2[sorted_s[:-1]])
            yy2 = np.minimum(y2[i], y2[sorted_s[:-1]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if method == 'Min':
                o = inter / np.minimum(area[i], area[sorted_s[:-1]])
            else:
                o = inter / (area[i] + area[sorted_s[:-1]] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)[0]]

        return pick

    def _apply_bbox_regression(self, bboxes, add1=False):
        """Apply bbox regression (C++ matching)"""
        result = bboxes.copy()

        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]

        if add1:
            w = w + 1
            h = h + 1

        result[:, 0] = bboxes[:, 0] + bboxes[:, 5] * w
        result[:, 1] = bboxes[:, 1] + bboxes[:, 6] * h
        result[:, 2] = bboxes[:, 0] + w + w * bboxes[:, 7]
        result[:, 3] = bboxes[:, 1] + h + h * bboxes[:, 8]

        return result

    def _square_bbox(self, bboxes):
        """Convert bboxes to squares (C++ matching)"""
        square_bboxes = bboxes.copy()
        h = bboxes[:, 3] - bboxes[:, 1]
        w = bboxes[:, 2] - bboxes[:, 0]
        max_side = np.maximum(h, w)

        new_x1 = np.trunc(bboxes[:, 0] + w * 0.5 - max_side * 0.5).astype(int)
        new_y1 = np.trunc(bboxes[:, 1] + h * 0.5 - max_side * 0.5).astype(int)
        max_side_int = np.trunc(max_side).astype(int)

        square_bboxes[:, 0] = new_x1
        square_bboxes[:, 1] = new_y1
        square_bboxes[:, 2] = new_x1 + max_side_int
        square_bboxes[:, 3] = new_y1 + max_side_int
        return square_bboxes

    # ==================== Main Detection Method ====================

    def detect(self, img: np.ndarray):
        """
        Detect faces in image.

        Args:
            img: Input image (H, W, 3) in BGR format

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


class PurePythonMTCNN(MTCNNBase):
    """
    Pure Python MTCNN implementation using optimized CNN loader.

    This is a fallback when CoreML/ONNX are not available.
    Uses the cpp_cnn_loader to load C++ .dat model files.
    """

    def __init__(self, model_dir=None):
        super().__init__()

        # Lazy import to avoid dependency when using CoreML/ONNX backends
        try:
            from .cpp_cnn_loader_optimized import CPPCNN
        except ImportError:
            raise ImportError(
                "cpp_cnn_loader_optimized is required for PurePythonMTCNN. "
                "Use CoreMLMTCNN or ONNXMTCNN instead."
            )

        if model_dir is None:
            model_dir = os.path.expanduser(
                "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
                "face_detection/mtcnn/convert_to_cpp/"
            )

        # Load Pure Python CNNs
        self.pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
        self.rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))
        self.onet = CPPCNN(os.path.join(model_dir, "ONet.dat"))

    def _run_pnet(self, img_data):
        """Run PNet using Pure Python CNN"""
        outputs = self.pnet(img_data)
        output = outputs[-1]
        return output[np.newaxis, :, :, :]

    def _run_rnet_batch(self, img_data_list):
        """Run RNet on batch (sequential for Pure Python)"""
        outputs = []
        for img_data in img_data_list:
            output = self.rnet(img_data)
            outputs.append(output[-1])
        return np.vstack(outputs)

    def _run_onet_batch(self, img_data_list):
        """Run ONet on batch (sequential for Pure Python)"""
        outputs = []
        for img_data in img_data_list:
            output = self.onet(img_data)
            outputs.append(output[-1])
        return np.vstack(outputs)


# Backward compatibility alias
PurePythonMTCNN_Optimized = PurePythonMTCNN
