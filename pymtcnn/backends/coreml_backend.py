"""
CoreML MTCNN Detector for Apple Silicon / Neural Engine

High-performance face detection using Apple Neural Engine (ANE).
Achieves 34.26 FPS with cross-frame batching on Apple Silicon.

Performance:
- detect(): 31.88 FPS (within-frame batching)
- detect_batch(): 34.26 FPS (cross-frame batching, batch_size=4)
"""

import numpy as np
import coremltools as ct
from pathlib import Path
import time

from ..base import PurePythonMTCNN_Optimized


class CoreMLMTCNN(PurePythonMTCNN_Optimized):
    """
    CoreML-accelerated MTCNN using Apple Neural Engine.

    Inherits all pipeline logic from PurePythonMTCNN_Optimized,
    only replaces the CNN forward passes with CoreML inference.
    """

    def __init__(self, coreml_dir=None, verbose=False):
        """
        Initialize CoreML MTCNN detector.

        Args:
            coreml_dir: Directory containing CoreML models (default: models/)
            verbose: Print loading messages (default: False)
        """
        # Don't call super().__init__() - we'll load CoreML models instead

        if coreml_dir is None:
            coreml_dir = Path(__file__).parent.parent / "models"
        else:
            coreml_dir = Path(coreml_dir)

        if verbose:
            print(f"Loading CoreML models from {coreml_dir}...")

        # Load CoreML models
        self.pnet_model = ct.models.MLModel(str(coreml_dir / "pnet_fp32.mlpackage"))
        self.rnet_model = ct.models.MLModel(str(coreml_dir / "rnet_fp32.mlpackage"))
        self.onet_model = ct.models.MLModel(str(coreml_dir / "onet_fp32.mlpackage"))

        # Get input/output names from CoreML models
        self.pnet_input_name = self.pnet_model.get_spec().description.input[0].name
        self.pnet_output_name = self.pnet_model.get_spec().description.output[0].name

        self.rnet_input_name = self.rnet_model.get_spec().description.input[0].name
        self.rnet_output_name = self.rnet_model.get_spec().description.output[0].name

        self.onet_input_name = self.onet_model.get_spec().description.input[0].name
        self.onet_output_name = self.onet_model.get_spec().description.output[0].name

        if verbose:
            print(f"  PNet: {self.pnet_input_name} → {self.pnet_output_name}")
            print(f"  RNet: {self.rnet_input_name} → {self.rnet_output_name}")
            print(f"  ONet: {self.onet_input_name} → {self.onet_output_name}")
            print("✓ CoreML MTCNN initialized")

        # MTCNN parameters
        self.thresholds = [0.6, 0.7, 0.7]  # PNet, RNet, ONet thresholds
        self.min_face_size = 60
        self.factor = 0.709

    def _run_pnet(self, img_data):
        """
        Run PNet using CoreML model.

        Args:
            img_data: Input shape (C, H, W)

        Returns:
            Output shape (1, 6, H', W') - batch dimension added for compatibility
        """
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Run CoreML inference
        result = self.pnet_model.predict({self.pnet_input_name: img_batch})
        output = result[self.pnet_output_name]

        # Output should be (1, 6, H', W')
        # Pure Python MTCNN expects (1, 6, H', W')
        return output

    def _run_rnet(self, img_data):
        """
        Run RNet using CoreML model (single input - legacy compatibility).

        Args:
            img_data: Input shape (C, H, W) - should be (3, 24, 24)

        Returns:
            Output shape (6,) - classification + bbox regression
        """
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Run CoreML inference
        result = self.rnet_model.predict({self.rnet_input_name: img_batch})
        output = result[self.rnet_output_name]

        # Remove batch dimension: (1, 6) → (6,)
        return output.squeeze(0)

    def _run_rnet_batch(self, img_data_list, max_batch_size=50):
        """
        Run RNet using CoreML model with batching (within-frame batching).

        Automatically splits large batches into chunks of max_batch_size.

        Args:
            img_data_list: List of images, each shape (C, H, W) - should be (3, 24, 24)
            max_batch_size: Maximum batch size (default: 50)

        Returns:
            Output shape (N, 6) - classification + bbox regression for N inputs
        """
        if len(img_data_list) == 0:
            return np.empty((0, 6))

        # If batch is within limit, process in one go
        if len(img_data_list) <= max_batch_size:
            img_batch = np.stack(img_data_list, axis=0).astype(np.float32)
            result = self.rnet_model.predict({self.rnet_input_name: img_batch})
            return result[self.rnet_output_name]

        # Otherwise, split into smaller batches
        outputs = []
        for i in range(0, len(img_data_list), max_batch_size):
            batch_chunk = img_data_list[i:i + max_batch_size]
            img_batch = np.stack(batch_chunk, axis=0).astype(np.float32)
            result = self.rnet_model.predict({self.rnet_input_name: img_batch})
            outputs.append(result[self.rnet_output_name])

        # Concatenate all outputs
        return np.vstack(outputs)

    def _run_onet(self, img_data):
        """
        Run ONet using CoreML model (single input - legacy compatibility).

        Args:
            img_data: Input shape (C, H, W) - should be (3, 48, 48)

        Returns:
            Output shape (16,) - classification + bbox + landmarks
        """
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)

        # Run CoreML inference
        result = self.onet_model.predict({self.onet_input_name: img_batch})
        output = result[self.onet_output_name]

        # Remove batch dimension: (1, 16) → (16,)
        return output.squeeze(0)

    def _run_onet_batch(self, img_data_list, max_batch_size=50):
        """
        Run ONet using CoreML model with batching (within-frame batching).

        Automatically splits large batches into chunks of max_batch_size.

        Args:
            img_data_list: List of images, each shape (C, H, W) - should be (3, 48, 48)
            max_batch_size: Maximum batch size (default: 50)

        Returns:
            Output shape (N, 16) - classification + bbox + landmarks for N inputs
        """
        if len(img_data_list) == 0:
            return np.empty((0, 16))

        # If batch is within limit, process in one go
        if len(img_data_list) <= max_batch_size:
            img_batch = np.stack(img_data_list, axis=0).astype(np.float32)
            result = self.onet_model.predict({self.onet_input_name: img_batch})
            return result[self.onet_output_name]

        # Otherwise, split into smaller batches
        outputs = []
        for i in range(0, len(img_data_list), max_batch_size):
            batch_chunk = img_data_list[i:i + max_batch_size]
            img_batch = np.stack(batch_chunk, axis=0).astype(np.float32)
            result = self.onet_model.predict({self.onet_input_name: img_batch})
            outputs.append(result[self.onet_output_name])

        # Concatenate all outputs
        return np.vstack(outputs)

    def detect(self, img):
        """
        Detect faces in image using MTCNN with batched RNet/ONet inference.

        This method overrides the parent's detect() to use batch processing
        for RNet and ONet stages, providing 2-3x speedup over sequential processing.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
        """
        import cv2  # Import here to match parent

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

        # Stage 1: PNet (no batching - image pyramid requires sequential processing)
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

        # Stage 2: RNet (WITH BATCHING)
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

        # Run RNet WITH BATCHING (single CoreML call instead of loop)
        output = self._run_rnet_batch(rnet_input)

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

        # Stage 3: ONet (WITH BATCHING)
        total_boxes = self._square_bbox(total_boxes)

        onet_input = []
        valid_indices = []

        for i in range(total_boxes.shape[0]):
            # C++ matching extraction: use (x-1, y-1) start and (w+1, h+1) buffer
            box_x = total_boxes[i, 0]
            box_y = total_boxes[i, 1]
            box_w = total_boxes[i, 2] - total_boxes[i, 0]
            box_h = total_boxes[i, 3] - total_boxes[i, 1]

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
                continue

            # Create zero-padded buffer of size (w+1, h+1)
            tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)

            # Copy image region to buffer
            copy_h = end_y_in - start_y_in
            copy_w = end_x_in - start_x_in
            tmp[start_y_out:start_y_out+copy_h, start_x_out:start_x_out+copy_w] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            # Resize to 48x48 and preprocess
            face = cv2.resize(tmp, (48, 48))
            onet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
            valid_indices.append(i)

        if len(onet_input) == 0:
            return np.empty((0, 4)), np.empty((0, 5, 2))

        total_boxes = total_boxes[valid_indices]

        # Run ONet WITH BATCHING (single CoreML call instead of loop)
        output = self._run_onet_batch(onet_input)

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

        # ONet outputs landmarks as [x0,x1,x2,x3,x4, y0,y1,y2,y3,y4] (x-first format)
        # Reshape correctly to (N, 5, 2) where [:, i, 0] = x_i, [:, i, 1] = y_i
        landmarks = np.stack([landmarks[:, 0:5], landmarks[:, 5:10]], axis=2)

        # Final NMS
        keep = self._nms(total_boxes, 0.7, 'Min')
        total_boxes = total_boxes[keep]
        landmarks = landmarks[keep]

        # Apply final calibration (CRITICAL for accuracy!) - VECTORIZED
        # This adjusts the bbox to be tight around facial landmarks
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

        # Denormalize landmarks using CALIBRATED bbox dimensions - VECTORIZED
        # This ensures landmarks and bbox are properly aligned
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

    def detect_with_debug(self, img):
        """
        Detect faces in image with debug info capturing stage-by-stage outputs.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
            debug_info: Dict with stage-by-stage outputs
        """
        import cv2
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
        output = self._run_rnet_batch(rnet_input)
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
            # C++ matching extraction: use (x-1, y-1) start and (w+1, h+1) buffer
            box_x = total_boxes[i, 0]
            box_y = total_boxes[i, 1]
            box_w = total_boxes[i, 2] - total_boxes[i, 0]
            box_h = total_boxes[i, 3] - total_boxes[i, 1]

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
                continue

            # Create zero-padded buffer of size (w+1, h+1)
            tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)

            # Copy image region to buffer
            copy_h = end_y_in - start_y_in
            copy_w = end_x_in - start_x_in
            tmp[start_y_out:start_y_out+copy_h, start_x_out:start_x_out+copy_w] = \
                img_float[start_y_in:end_y_in, start_x_in:end_x_in]

            # Resize to 48x48 and preprocess
            face = cv2.resize(tmp, (48, 48))
            onet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
            valid_indices.append(i)

        if len(onet_input) == 0:
            onet_time = (time.time() - t0) * 1000
            debug_info['onet'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'time_ms': onet_time}
            debug_info['final'] = {'num_boxes': 0, 'boxes': np.array([]), 'landmarks': np.array([]), 'total_time_ms': pnet_time + rnet_time + onet_time}
            return np.empty((0, 4)), np.empty((0, 5, 2)), debug_info

        total_boxes = total_boxes[valid_indices]
        output = self._run_onet_batch(onet_input)
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

        # CAPTURE: Raw normalized landmarks and bbox BEFORE calibration (for C++ comparison)
        landmarks_raw_normalized = landmarks.copy()  # [0-1] range, ONet output
        bbox_before_calibration = total_boxes[:, :4].copy()  # [x1, y1, x2, y2] BEFORE calibration

        # Apply calibration - VECTORIZED
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

        # Denormalize landmarks using CALIBRATED bbox dimensions - VECTORIZED
        w = (total_boxes[:, 2] - total_boxes[:, 0]).reshape(-1, 1)
        h = (total_boxes[:, 3] - total_boxes[:, 1]).reshape(-1, 1)
        x1 = total_boxes[:, 0].reshape(-1, 1)
        y1 = total_boxes[:, 1].reshape(-1, 1)
        landmarks[:, :, 0] = x1 + landmarks[:, :, 0] * w
        landmarks[:, :, 1] = y1 + landmarks[:, :, 1] * h

        onet_time = (time.time() - t0) * 1000

        # Convert to [x, y, w, h] format
        bboxes = np.zeros((total_boxes.shape[0], 4))
        bboxes[:, 0] = total_boxes[:, 0]
        bboxes[:, 1] = total_boxes[:, 1]
        bboxes[:, 2] = total_boxes[:, 2] - total_boxes[:, 0]
        bboxes[:, 3] = total_boxes[:, 3] - total_boxes[:, 1]

        # Convert uncalibrated bbox to [x, y, w, h] format for debug
        bbox_uncal = np.zeros((bbox_before_calibration.shape[0], 4))
        bbox_uncal[:, 0] = bbox_before_calibration[:, 0]
        bbox_uncal[:, 1] = bbox_before_calibration[:, 1]
        bbox_uncal[:, 2] = bbox_before_calibration[:, 2] - bbox_before_calibration[:, 0]
        bbox_uncal[:, 3] = bbox_before_calibration[:, 3] - bbox_before_calibration[:, 1]

        debug_info['onet'] = {
            'num_boxes': bboxes.shape[0],
            'boxes': bboxes.copy(),
            'landmarks': landmarks.copy(),
            'landmarks_raw_normalized': landmarks_raw_normalized.copy(),  # Raw ONet output [0-1]
            'bbox_before_calibration': bbox_uncal.copy(),  # For C++ comparison
            'time_ms': onet_time
        }

        debug_info['final'] = {
            'num_boxes': bboxes.shape[0],
            'boxes': bboxes.copy(),
            'landmarks': landmarks.copy(),
            'total_time_ms': pnet_time + rnet_time + onet_time
        }

        return bboxes, landmarks, debug_info

    def detect_batch(self, frames):
        """
        Detect faces in multiple frames with cross-frame batching.

        This method processes multiple frames together, batching RNet/ONet candidates
        across all frames for maximum ANE utilization. Expected 1.2-1.5x speedup
        over per-frame batching.

        Args:
            frames: List of BGR images, each (H, W, 3)

        Returns:
            List of (bboxes, landmarks) tuples, one per frame
            - bboxes: (N, 4) array of [x, y, w, h]
            - landmarks: (N, 5, 2) array of facial landmarks
        """
        import cv2

        if len(frames) == 0:
            return []

        num_frames = len(frames)

        # Stage 1: Run PNet on each frame separately (image pyramid can't be batched)
        print(f"  Stage 1: PNet on {num_frames} frames...")
        pnet_results = []  # List of (total_boxes, img_float, img_shape)

        for frame_idx, img in enumerate(frames):
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

            # PNet stage
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

            pnet_results.append((total_boxes, img_float, (img_h, img_w)))

        # Stage 2: Mega-batch RNet across all frames
        print(f"  Stage 2: RNet mega-batching...")
        all_rnet_input = []
        rnet_frame_indices = []  # Track which frame each candidate belongs to
        rnet_box_indices = []    # Track which box index within the frame

        for frame_idx, (total_boxes, img_float, img_shape) in enumerate(pnet_results):
            img_h, img_w = img_shape
            total_boxes = self._square_bbox(total_boxes)

            for box_idx in range(total_boxes.shape[0]):
                bbox_x = total_boxes[box_idx, 0]
                bbox_y = total_boxes[box_idx, 1]
                bbox_w = total_boxes[box_idx, 2] - total_boxes[box_idx, 0]
                bbox_h = total_boxes[box_idx, 3] - total_boxes[box_idx, 1]

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
                all_rnet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
                rnet_frame_indices.append(frame_idx)
                rnet_box_indices.append(box_idx)

        # Mega-batch RNet inference
        if len(all_rnet_input) > 0:
            print(f"    RNet: Processing {len(all_rnet_input)} candidates from {num_frames} frames")
            rnet_output = self._run_rnet_batch(all_rnet_input)
            scores = 1.0 / (1.0 + np.exp(rnet_output[:, 0] - rnet_output[:, 1]))
        else:
            rnet_output = np.empty((0, 6))
            scores = np.array([])

        # Regroup RNet results by frame
        rnet_results = []  # List of (total_boxes, scores, reg) per frame
        for frame_idx in range(num_frames):
            # Find candidates for this frame
            frame_mask = np.array(rnet_frame_indices) == frame_idx
            if not frame_mask.any():
                rnet_results.append((np.empty((0, 9)), np.array([]), np.empty((0, 4))))
                continue

            frame_scores = scores[frame_mask]
            frame_output = rnet_output[frame_mask]
            frame_boxes = pnet_results[frame_idx][0][np.array(rnet_box_indices)[frame_mask]]

            # Filter by threshold
            keep = frame_scores > self.thresholds[1]
            if not keep.any():
                rnet_results.append((np.empty((0, 9)), np.array([]), np.empty((0, 4))))
                continue

            frame_boxes = frame_boxes[keep]
            frame_scores = frame_scores[keep]
            frame_reg = frame_output[keep, 2:6]

            # NMS
            keep = self._nms(frame_boxes, 0.7, 'Union')
            frame_boxes = frame_boxes[keep]
            frame_scores = frame_scores[keep]
            frame_reg = frame_reg[keep]

            # Apply RNet regression
            w = frame_boxes[:, 2] - frame_boxes[:, 0]
            h = frame_boxes[:, 3] - frame_boxes[:, 1]
            x1 = frame_boxes[:, 0].copy()
            y1 = frame_boxes[:, 1].copy()

            frame_boxes[:, 0] = x1 + frame_reg[:, 0] * w
            frame_boxes[:, 1] = y1 + frame_reg[:, 1] * h
            frame_boxes[:, 2] = x1 + w + w * frame_reg[:, 2]
            frame_boxes[:, 3] = y1 + h + h * frame_reg[:, 3]
            frame_boxes[:, 4] = frame_scores

            rnet_results.append((frame_boxes, frame_scores, frame_reg))

        # Stage 3: Mega-batch ONet across all frames
        print(f"  Stage 3: ONet mega-batching...")
        all_onet_input = []
        onet_frame_indices = []
        onet_box_indices = []

        for frame_idx, (total_boxes, img_float, img_shape) in enumerate(pnet_results):
            img_h, img_w = img_shape
            rnet_boxes = rnet_results[frame_idx][0]

            if rnet_boxes.shape[0] == 0:
                continue

            total_boxes = self._square_bbox(rnet_boxes)

            for box_idx in range(total_boxes.shape[0]):
                # C++ matching extraction: use (x-1, y-1) start and (w+1, h+1) buffer
                box_x = total_boxes[box_idx, 0]
                box_y = total_boxes[box_idx, 1]
                box_w = total_boxes[box_idx, 2] - total_boxes[box_idx, 0]
                box_h = total_boxes[box_idx, 3] - total_boxes[box_idx, 1]

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
                    continue

                # Create zero-padded buffer of size (w+1, h+1)
                tmp = np.zeros((height_target, width_target, 3), dtype=np.float32)

                # Copy image region to buffer
                copy_h = end_y_in - start_y_in
                copy_w = end_x_in - start_x_in
                tmp[start_y_out:start_y_out+copy_h, start_x_out:start_x_out+copy_w] = \
                    img_float[start_y_in:end_y_in, start_x_in:end_x_in]

                # Resize to 48x48 and preprocess
                face = cv2.resize(tmp, (48, 48))
                all_onet_input.append(self._preprocess(face, flip_bgr_to_rgb=True))
                onet_frame_indices.append(frame_idx)
                onet_box_indices.append(box_idx)

        # Mega-batch ONet inference
        if len(all_onet_input) > 0:
            print(f"    ONet: Processing {len(all_onet_input)} candidates from {num_frames} frames")
            onet_output = self._run_onet_batch(all_onet_input)
            scores = 1.0 / (1.0 + np.exp(onet_output[:, 0] - onet_output[:, 1]))
        else:
            onet_output = np.empty((0, 16))
            scores = np.array([])

        # Regroup ONet results by frame and finalize
        final_results = []
        for frame_idx in range(num_frames):
            # Find candidates for this frame
            frame_mask = np.array(onet_frame_indices) == frame_idx
            if not frame_mask.any():
                final_results.append((np.empty((0, 4)), np.empty((0, 5, 2))))
                continue

            frame_scores = scores[frame_mask]
            frame_output = onet_output[frame_mask]

            # Get the squared boxes for this frame
            rnet_boxes = rnet_results[frame_idx][0]
            total_boxes = self._square_bbox(rnet_boxes)
            total_boxes = total_boxes[np.array(onet_box_indices)[frame_mask]]

            # Filter by threshold
            keep = frame_scores > self.thresholds[2]
            total_boxes = total_boxes[keep]
            frame_scores = frame_scores[keep]
            reg = frame_output[keep, 2:6]
            landmarks = frame_output[keep, 6:16]

            if total_boxes.shape[0] == 0:
                final_results.append((np.empty((0, 4)), np.empty((0, 5, 2))))
                continue

            # Apply ONet regression (with +1)
            w = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1
            x1 = total_boxes[:, 0].copy()
            y1 = total_boxes[:, 1].copy()

            total_boxes[:, 0] = x1 + reg[:, 0] * w
            total_boxes[:, 1] = y1 + reg[:, 1] * h
            total_boxes[:, 2] = x1 + w + w * reg[:, 2]
            total_boxes[:, 3] = y1 + h + h * reg[:, 3]
            total_boxes[:, 4] = frame_scores

            # NOTE: Landmarks are denormalized AFTER calibration (see below)

            landmarks = landmarks.reshape(-1, 5, 2)

            # Final NMS
            keep = self._nms(total_boxes, 0.7, 'Min')
            total_boxes = total_boxes[keep]
            landmarks = landmarks[keep]

            # Apply final calibration - VECTORIZED
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

            # Denormalize landmarks using CALIBRATED bbox dimensions - VECTORIZED
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

            final_results.append((bboxes, landmarks))

        return final_results
