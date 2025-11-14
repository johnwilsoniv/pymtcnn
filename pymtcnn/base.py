#!/usr/bin/env python3
"""
OPTIMIZED Pure Python MTCNN - Production Version

Performance improvements:
- Uses optimized CNN loader (vectorized im2col, PReLU, MaxPool)
- Removed ALL debug file I/O operations
- Removed verbose debug logging
- Optimized NumPy operations

All accuracy-critical logic PRESERVED:
- Exact PReLU, MaxPool rounding, BGR→RGB flipping
- C++ bbox calibration coefficients
- All threshold and NMS logic
"""

import numpy as np
import cv2
import os


class PurePythonMTCNN_Optimized:
    """Optimized Pure Python CNN MTCNN"""

    def __init__(self, model_dir=None):
        # Lazy import to avoid dependency when using CoreML/ONNX backends
        try:
            from .cpp_cnn_loader_optimized import CPPCNN
        except ImportError:
            raise ImportError(
                "cpp_cnn_loader_optimized is required for base MTCNN. "
                "Use CoreMLMTCNN or ONNXMTCNN instead."
            )

        if model_dir is None:
            model_dir = os.path.expanduser(
                "~/repo/fea_tool/external_libs/openFace/OpenFace/matlab_version/"
                "face_detection/mtcnn/convert_to_cpp/"
            )

        # Load optimized Pure Python CNNs
        self.pnet = CPPCNN(os.path.join(model_dir, "PNet.dat"))
        self.rnet = CPPCNN(os.path.join(model_dir, "RNet.dat"))
        self.onet = CPPCNN(os.path.join(model_dir, "ONet.dat"))

        # MTCNN parameters (matching C++)
        self.thresholds = [0.6, 0.7, 0.7]
        self.min_face_size = 60
        self.factor = 0.709

    def _preprocess(self, img: np.ndarray, flip_bgr_to_rgb: bool = True) -> np.ndarray:
        """Preprocess image - PRESERVES accuracy-critical BGR→RGB flip"""
        img_norm = (img.astype(np.float32) - 127.5) * 0.0078125
        img_chw = np.transpose(img_norm, (2, 0, 1))

        if flip_bgr_to_rgb:
            img_chw = img_chw[[2, 1, 0], :, :]

        return img_chw

    def _run_pnet(self, img_data):
        """Run PNet using optimized CNN"""
        outputs = self.pnet(img_data)
        output = outputs[-1]
        return output[np.newaxis, :, :, :]

    def _run_rnet(self, img_data):
        """Run RNet using optimized CNN"""
        outputs = self.rnet(img_data)
        return outputs[-1]

    def _run_onet(self, img_data):
        """Run ONet using optimized CNN"""
        outputs = self.onet(img_data)
        return outputs[-1]

    def detect(self, img: np.ndarray):
        """
        Detect faces using optimized Pure Python CNN MTCNN

        Args:
            img: Input image (H, W, 3) in BGR format

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
        """
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
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = np.vstack(total_boxes)

        # NMS across scales
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        # Apply PNet bbox regression
        total_boxes = self._apply_bbox_regression(total_boxes)

        # Stage 2: RNet
        total_boxes = self._square_bbox(total_boxes)

        rnet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            bbox_x = total_boxes[i, 0]
            bbox_y = total_boxes[i, 1]
            bbox_w = total_boxes[i, 2] - total_boxes[i, 0]
            bbox_h = total_boxes[i, 3] - total_boxes[i, 1]

            # C++ RNet cropping with +1 padding
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
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run RNet
        rnet_outputs = []
        for face_data in rnet_input:
            output = self._run_rnet(face_data)
            rnet_outputs.append(output)

        output = np.vstack(rnet_outputs)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # Filter by threshold
        keep = scores > self.thresholds[1]

        if not keep.any():
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]

        # NMS
        keep = self._nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = reg[keep]

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

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

        # Stage 3: ONet
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
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run ONet
        onet_outputs = []
        for face_data in onet_input:
            output = self._run_onet(face_data)
            onet_outputs.append(output)

        output = np.vstack(onet_outputs)
        scores = 1.0 / (1.0 + np.exp(output[:, 0] - output[:, 1]))

        # Filter by threshold
        keep = scores > self.thresholds[2]

        total_boxes = total_boxes[keep]
        scores = scores[keep]
        reg = output[keep, 2:6]
        landmarks = output[keep, 6:16]

        if total_boxes.shape[0] == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

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

        # Denormalize landmarks
        for i in range(5):
            landmarks[:, 2*i] = total_boxes[:, 0] + landmarks[:, 2*i] * w
            landmarks[:, 2*i+1] = total_boxes[:, 1] + landmarks[:, 2*i+1] * h

        landmarks = landmarks.reshape(-1, 5, 2)

        # Final NMS
        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        # Apply final calibration (CRITICAL for accuracy!)
        for k in range(total_boxes.shape[0]):
            w = total_boxes[k, 2] - total_boxes[k, 0]
            h = total_boxes[k, 3] - total_boxes[k, 1]
            new_x1 = total_boxes[k, 0] + w * -0.0075
            new_y1 = total_boxes[k, 1] + h * 0.2459
            new_width = w * 1.0323
            new_height = h * 0.7751
            total_boxes[k, 0] = new_x1
            total_boxes[k, 1] = new_y1
            total_boxes[k, 2] = new_x1 + new_width
            total_boxes[k, 3] = new_y1 + new_height

        # Convert to (x, y, width, height) format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        return bboxes, landmarks

    def _generate_bboxes(self, score_map, reg_map, scale, threshold):
        """Generate bounding boxes from PNet output (C++ matching)"""
        stride = 2
        cellsize = 12

        t_index = np.where(score_map[:, :, 1] >= threshold)

        if t_index[0].size == 0:
            return np.array([])

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

        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, 0:4]
            dx1, dy1, dx2, dy2 = bboxes[i, 5:9]

            w = x2 - x1
            h = y2 - y1

            if add1:
                w = w + 1
                h = h + 1

            new_x1 = x1 + dx1 * w
            new_y1 = y1 + dy1 * h
            new_x2 = x1 + w + w * dx2
            new_y2 = y1 + h + h * dy2

            result[i, 0] = new_x1
            result[i, 1] = new_y1
            result[i, 2] = new_x2
            result[i, 3] = new_y2

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
