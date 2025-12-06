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
import cv2

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

    # ==================== CoreML Inference Methods ====================

    def _run_pnet(self, img_data):
        """Run PNet using CoreML model. Input: (C, H, W), Output: (1, 6, H', W')"""
        img_batch = np.expand_dims(img_data, axis=0).astype(np.float32)
        result = self.pnet_model.predict({self.pnet_input_name: img_batch})
        return result[self.pnet_output_name]

    def _run_rnet_batch(self, img_data_list, max_batch_size=50):
        """Run RNet on batch. Input: list of (C, H, W), Output: (N, 6)"""
        if len(img_data_list) == 0:
            return np.empty((0, 6))

        if len(img_data_list) <= max_batch_size:
            img_batch = np.stack(img_data_list, axis=0).astype(np.float32)
            result = self.rnet_model.predict({self.rnet_input_name: img_batch})
            return result[self.rnet_output_name]

        outputs = []
        for i in range(0, len(img_data_list), max_batch_size):
            batch_chunk = img_data_list[i:i + max_batch_size]
            img_batch = np.stack(batch_chunk, axis=0).astype(np.float32)
            result = self.rnet_model.predict({self.rnet_input_name: img_batch})
            outputs.append(result[self.rnet_output_name])
        return np.vstack(outputs)

    def _run_onet_batch(self, img_data_list, max_batch_size=50):
        """Run ONet on batch. Input: list of (C, H, W), Output: (N, 16)"""
        if len(img_data_list) == 0:
            return np.empty((0, 16))

        if len(img_data_list) <= max_batch_size:
            img_batch = np.stack(img_data_list, axis=0).astype(np.float32)
            result = self.onet_model.predict({self.onet_input_name: img_batch})
            return result[self.onet_output_name]

        outputs = []
        for i in range(0, len(img_data_list), max_batch_size):
            batch_chunk = img_data_list[i:i + max_batch_size]
            img_batch = np.stack(batch_chunk, axis=0).astype(np.float32)
            result = self.onet_model.predict({self.onet_input_name: img_batch})
            outputs.append(result[self.onet_output_name])
        return np.vstack(outputs)

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

        # Run RNet
        output = self._run_rnet_batch(rnet_input)
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

        # Run ONet
        output = self._run_onet_batch(onet_input)
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

    def detect_batch(self, frames):
        """
        Detect faces in multiple frames with cross-frame batching.

        This method processes multiple frames together, batching RNet/ONet candidates
        across all frames for maximum ANE utilization.

        Args:
            frames: List of BGR images, each (H, W, 3)

        Returns:
            List of (bboxes, landmarks) tuples, one per frame
        """
        if len(frames) == 0:
            return []

        # Stage 1: Run PNet on all frames
        pnet_results = []
        for frame in frames:
            img_float = frame.astype(np.float32)
            total_boxes = self._pnet_stage(img_float)
            pnet_results.append((total_boxes, img_float))

        # Stage 2: Mega-batch RNet across all frames
        all_rnet_input = []
        rnet_frame_indices = []
        rnet_box_indices = []

        for frame_idx, (total_boxes, img_float) in enumerate(pnet_results):
            if total_boxes is None:
                continue

            squared_boxes = self._square_bbox(total_boxes.copy())
            for box_idx in range(squared_boxes.shape[0]):
                crop = self._extract_crop(img_float, squared_boxes[box_idx], 24)
                if crop is not None:
                    all_rnet_input.append(crop)
                    rnet_frame_indices.append(frame_idx)
                    rnet_box_indices.append(box_idx)

        # Run mega-batch RNet
        if len(all_rnet_input) > 0:
            rnet_output = self._run_rnet_batch(all_rnet_input)
            scores = 1.0 / (1.0 + np.exp(rnet_output[:, 0] - rnet_output[:, 1]))
        else:
            rnet_output = np.empty((0, 6))
            scores = np.array([])

        # Process RNet results per frame
        rnet_results = [None] * len(frames)
        for frame_idx in range(len(frames)):
            frame_mask = np.array(rnet_frame_indices) == frame_idx
            if not frame_mask.any():
                continue

            total_boxes = pnet_results[frame_idx][0]
            if total_boxes is None:
                continue

            frame_scores = scores[frame_mask]
            frame_output = rnet_output[frame_mask]
            frame_box_indices = np.array(rnet_box_indices)[frame_mask]
            frame_boxes = total_boxes[frame_box_indices]

            # Filter by threshold
            keep = frame_scores > self.thresholds[1]
            if not keep.any():
                continue

            frame_boxes = frame_boxes[keep]
            frame_scores = frame_scores[keep]
            frame_reg = frame_output[keep, 2:6]

            # NMS - update scores BEFORE NMS
            frame_boxes[:, 4] = frame_scores
            keep = self._nms(frame_boxes, 0.7, 'Union')
            frame_boxes = frame_boxes[keep]
            frame_scores = frame_scores[keep]
            frame_reg = frame_reg[keep]

            if frame_boxes.shape[0] == 0:
                continue

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

            rnet_results[frame_idx] = frame_boxes

        # Stage 3: Mega-batch ONet across all frames
        all_onet_input = []
        onet_frame_indices = []
        onet_box_indices = []

        for frame_idx, (_, img_float) in enumerate(pnet_results):
            rnet_boxes = rnet_results[frame_idx]
            if rnet_boxes is None:
                continue

            squared_boxes = self._square_bbox(rnet_boxes.copy())
            for box_idx in range(squared_boxes.shape[0]):
                crop = self._extract_crop(img_float, squared_boxes[box_idx], 48)
                if crop is not None:
                    all_onet_input.append(crop)
                    onet_frame_indices.append(frame_idx)
                    onet_box_indices.append(box_idx)

        # Run mega-batch ONet
        if len(all_onet_input) > 0:
            onet_output = self._run_onet_batch(all_onet_input)
            scores = 1.0 / (1.0 + np.exp(onet_output[:, 0] - onet_output[:, 1]))
        else:
            onet_output = np.empty((0, 16))
            scores = np.array([])

        # Process ONet results per frame
        final_results = []
        for frame_idx in range(len(frames)):
            frame_mask = np.array(onet_frame_indices) == frame_idx
            if not frame_mask.any():
                final_results.append((np.empty((0, 4)), np.empty((0, 5, 2))))
                continue

            rnet_boxes = rnet_results[frame_idx]
            if rnet_boxes is None:
                final_results.append((np.empty((0, 4)), np.empty((0, 5, 2))))
                continue

            frame_scores = scores[frame_mask]
            frame_output = onet_output[frame_mask]
            frame_box_indices = np.array(onet_box_indices)[frame_mask]

            squared_boxes = self._square_bbox(rnet_boxes.copy())
            total_boxes = squared_boxes[frame_box_indices]

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

            # Reshape landmarks
            landmarks = np.stack([landmarks[:, 0:5], landmarks[:, 5:10]], axis=2)

            # Final NMS
            keep = self._nms(total_boxes, 0.7, 'Min')
            total_boxes = total_boxes[keep]
            landmarks = landmarks[keep]

            # Denormalize landmarks
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
