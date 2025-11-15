"""
ONNX MTCNN Detector with CUDA Support

Cross-platform face detection using ONNX Runtime.
Automatically selects best execution provider: CUDA > CoreML > CPU.

Based on the working ONNX implementation with C++ OpenFace weights.
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import platform
import time


class ONNXMTCNN:
    """
    ONNX Runtime-accelerated MTCNN with automatic hardware acceleration.

    Complete standalone implementation using ONNX models.

    Execution providers (auto-selected in priority order):
    1. CUDAExecutionProvider - NVIDIA GPUs with CUDA
    2. CoreMLExecutionProvider - Apple Neural Engine on macOS
    3. CPUExecutionProvider - CPU fallback (all platforms)
    """

    def __init__(self, model_dir=None, provider=None, verbose=False):
        """
        Initialize ONNX MTCNN detector.

        Args:
            model_dir: Directory containing ONNX models (default: models/)
            provider: Execution provider name or None for auto-detection
                     Options: 'cuda', 'coreml', 'cpu', None (auto)
            verbose: Print loading messages (default: False)
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models"
        else:
            model_dir = Path(model_dir)

        # Determine execution providers
        providers = self._get_execution_providers(provider, verbose)

        if verbose:
            print(f"Loading ONNX models from {model_dir}...")
            print(f"Execution provider priority: {providers}")

        # Create ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load ONNX models
        try:
            self.pnet = ort.InferenceSession(
                str(model_dir / "pnet.onnx"),
                sess_options=sess_options,
                providers=providers
            )
            self.rnet = ort.InferenceSession(
                str(model_dir / "rnet.onnx"),
                sess_options=sess_options,
                providers=providers
            )
            self.onet = ort.InferenceSession(
                str(model_dir / "onet.onnx"),
                sess_options=sess_options,
                providers=providers
            )

            if verbose:
                print(f"✓ Models loaded successfully")
                print(f"Active provider: {self.pnet.get_providers()[0]}")

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX models: {e}")

        # Detection parameters (matching C++ defaults and pure Python optimized)
        self.min_face_size = 60  # Match pure Python optimized
        self.scale_factor = 0.709
        self.thresholds = [0.6, 0.7, 0.7]  # PNet, RNet, ONet
        self.nms_threshold = 0.7
        self.verbose = verbose

    def _get_execution_providers(self, provider, verbose):
        """
        Determine ONNX Runtime execution providers.

        Args:
            provider: User-specified provider ('cuda', 'coreml', 'cpu', None)
            verbose: Print provider selection info

        Returns:
            List of providers in priority order
        """
        available_providers = ort.get_available_providers()

        if verbose:
            print(f"Available ONNX Runtime providers: {available_providers}")

        # User explicitly specified a provider
        if provider is not None:
            provider_map = {
                'cuda': 'CUDAExecutionProvider',
                'coreml': 'CoreMLExecutionProvider',
                'cpu': 'CPUExecutionProvider',
            }

            if provider.lower() not in provider_map:
                raise ValueError(
                    f"Invalid provider '{provider}'. "
                    f"Must be one of: {list(provider_map.keys())}"
                )

            requested_provider = provider_map[provider.lower()]

            if requested_provider not in available_providers:
                raise RuntimeError(
                    f"Requested provider '{requested_provider}' is not available. "
                    f"Available providers: {available_providers}"
                )

            return [requested_provider]

        # Auto-detect: Priority order based on platform and availability
        providers = []

        # Priority 1: CUDA (if available and on non-macOS)
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')

        # Priority 2: CoreML (if available and on macOS)
        if 'CoreMLExecutionProvider' in available_providers and platform.system() == 'Darwin':
            providers.append('CoreMLExecutionProvider')

        # Priority 3: CPU (always available)
        providers.append('CPUExecutionProvider')

        return providers

    def preprocess_image(self, img):
        """Preprocess image: BGR -> RGB, normalize to [-1, 1]."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0

        return img_normalized

    def compute_scales(self, height, width):
        """Compute image pyramid scales."""
        min_dimension = min(height, width)
        min_net_size = 12

        # Calculate scale to make smallest face equal to min_net_size
        scale = min_net_size / self.min_face_size

        scales = []
        scaled_dim = min_dimension * scale

        while scaled_dim >= min_net_size:
            scales.append(scale)
            scale *= self.scale_factor
            scaled_dim = min_dimension * scale

        return scales

    def run_pnet(self, img):
        """Stage 1: Proposal Network (PNet)."""
        height, width = img.shape[:2]
        scales = self.compute_scales(height, width)

        all_boxes = []

        for scale in scales:
            # Resize image
            scaled_h = int(height * scale)
            scaled_w = int(width * scale)
            scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Prepare input: HWC -> CHW -> BCHW
            input_data = scaled_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # Run PNet
            output = self.pnet.run(None, {'input': input_data})[0][0]

            # Output shape: (6, H, W) = [cls_not_face, cls_face, bbox_dx, bbox_dy, bbox_dw, bbox_dh]
            logit_not_face = output[0, :, :]  # Not-face logit
            logit_face = output[1, :, :]      # Face logit
            bbox_map = output[2:6, :, :]      # Bbox regression

            # Convert logits to probabilities using softmax
            prob_face = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            # Find faces (threshold on probability)
            face_indices = np.where(prob_face > self.thresholds[0])

            if len(face_indices[0]) == 0:
                continue

            # Convert to bounding boxes
            for y, x in zip(face_indices[0], face_indices[1]):
                score = prob_face[y, x]

                # Map back to original image coordinates
                # PNet uses 12x12 receptive field with stride 2
                bbox_x = int((x * 2) / scale)
                bbox_y = int((y * 2) / scale)
                bbox_w = int(12 / scale)
                bbox_h = int(12 / scale)

                # Apply bbox regression
                dx = bbox_map[0, y, x]
                dy = bbox_map[1, y, x]
                dw = bbox_map[2, y, x]
                dh = bbox_map[3, y, x]

                bbox_x = int(bbox_x + dx * bbox_w)
                bbox_y = int(bbox_y + dy * bbox_h)
                bbox_w = int(bbox_w * np.exp(dw))
                bbox_h = int(bbox_h * np.exp(dh))

                all_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h, score])

        if len(all_boxes) == 0:
            return []

        # Non-maximum suppression
        boxes = np.array(all_boxes)
        keep = self.nms(boxes, self.nms_threshold)

        return boxes[keep]

    def run_rnet(self, img, boxes):
        """Stage 2: Refinement Network (RNet)."""
        if len(boxes) == 0:
            return []

        refined_boxes = []

        for box in boxes:
            x, y, w, h, _ = box

            # Extract and resize face patch to 24x24
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            face_patch = img[y1:y2, x1:x2]
            face_patch = cv2.resize(face_patch, (24, 24), interpolation=cv2.INTER_LINEAR)

            # Prepare input: HWC -> CHW -> BCHW
            input_data = face_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # Run RNet
            output = self.rnet.run(None, {'input': input_data})[0][0]

            # Output: [cls_not_face, cls_face, bbox_dx, bbox_dy, bbox_dw, bbox_dh]
            logit_not_face = output[0]
            logit_face = output[1]

            # Convert logits to probability using softmax
            score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            if score > self.thresholds[1]:
                # Apply bbox regression
                dx, dy, dw, dh = output[2:6]

                bbox_x = int(x + dx * w)
                bbox_y = int(y + dy * h)
                bbox_w = int(w * np.exp(dw))
                bbox_h = int(h * np.exp(dh))

                refined_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h, score])

        if len(refined_boxes) == 0:
            return []

        # Non-maximum suppression
        boxes = np.array(refined_boxes)
        keep = self.nms(boxes, self.nms_threshold)

        return boxes[keep]

    def run_onet(self, img, boxes):
        """Stage 3: Output Network (ONet)."""
        if len(boxes) == 0:
            return [], []

        final_boxes = []
        landmarks = []

        for box in boxes:
            x, y, w, h, _ = box

            # Extract and resize face patch to 48x48
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.shape[1], int(x + w)), min(img.shape[0], int(y + h))

            if x2 <= x1 or y2 <= y1:
                continue

            face_patch = img[y1:y2, x1:x2]
            face_patch = cv2.resize(face_patch, (48, 48), interpolation=cv2.INTER_LINEAR)

            # Prepare input: HWC -> CHW -> BCHW
            input_data = face_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

            # Run ONet
            output = self.onet.run(None, {'input': input_data})[0][0]

            # Output: [cls_not_face, cls_face, bbox_dx, bbox_dy, bbox_dw, bbox_dh,
            #          lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y, lm5_x, lm5_y]
            logit_not_face = output[0]
            logit_face = output[1]

            # Convert logits to probability using softmax
            score = 1.0 / (1.0 + np.exp(logit_not_face - logit_face))

            if score > self.thresholds[2]:
                # Apply bbox regression
                dx, dy, dw, dh = output[2:6]

                bbox_x = int(x + dx * w)
                bbox_y = int(y + dy * h)
                bbox_w = int(w * np.exp(dw))
                bbox_h = int(h * np.exp(dh))

                final_boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])

                # Extract landmarks (5 points)
                lm = output[6:16].reshape(5, 2)
                lm[:, 0] = x + lm[:, 0] * w  # x coordinates
                lm[:, 1] = y + lm[:, 1] * h  # y coordinates
                landmarks.append(lm)

        if len(final_boxes) == 0:
            return np.array([]), np.array([])

        # Final NMS
        # Re-add scores for NMS
        boxes_with_scores = []
        for i, box in enumerate(final_boxes):
            boxes_with_scores.append(box + [boxes[i][4]])  # Re-use original score

        boxes_array = np.array(boxes_with_scores)
        keep = self.nms(boxes_array, self.nms_threshold)

        final_boxes = np.array([final_boxes[i] for i in keep])
        landmarks = np.array([landmarks[i] for i in keep])

        return final_boxes, landmarks

    def nms(self, boxes, threshold):
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img):
        """
        Run full MTCNN cascade.

        Args:
            img: Input image (H, W, 3) in BGR format (OpenCV format)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
        """
        # Preprocess
        img_normalized = self.preprocess_image(img)

        # Stage 1: PNet
        pnet_boxes = self.run_pnet(img_normalized)

        # Stage 2: RNet
        rnet_boxes = self.run_rnet(img_normalized, pnet_boxes)

        # Stage 3: ONet
        onet_boxes, landmarks = self.run_onet(img_normalized, rnet_boxes)

        return onet_boxes, landmarks

    def detect_with_debug(self, img):
        """
        Run full MTCNN cascade with debug info capturing stage-by-stage outputs.

        Args:
            img: Input image (H, W, 3) in BGR format (OpenCV format)

        Returns:
            bboxes: (N, 4) array of [x, y, w, h]
            landmarks: (N, 5, 2) array of facial landmarks
            debug_info: Dict with stage-by-stage outputs
        """
        debug_info = {}

        # Preprocess
        img_normalized = self.preprocess_image(img)

        # Stage 1: PNet
        t0 = time.time()
        pnet_boxes = self.run_pnet(img_normalized)
        pnet_time = (time.time() - t0) * 1000

        debug_info['pnet'] = {
            'num_boxes': len(pnet_boxes),
            'boxes': pnet_boxes.copy() if len(pnet_boxes) > 0 else np.array([]),
            'time_ms': pnet_time
        }

        # Stage 2: RNet
        t0 = time.time()
        rnet_boxes = self.run_rnet(img_normalized, pnet_boxes)
        rnet_time = (time.time() - t0) * 1000

        debug_info['rnet'] = {
            'num_boxes': len(rnet_boxes),
            'boxes': rnet_boxes.copy() if len(rnet_boxes) > 0 else np.array([]),
            'time_ms': rnet_time
        }

        # Stage 3: ONet
        t0 = time.time()
        onet_boxes, landmarks = self.run_onet(img_normalized, rnet_boxes)
        onet_time = (time.time() - t0) * 1000

        debug_info['onet'] = {
            'num_boxes': len(onet_boxes),
            'boxes': onet_boxes.copy() if len(onet_boxes) > 0 else np.array([]),
            'landmarks': landmarks.copy() if len(landmarks) > 0 else np.array([]),
            'time_ms': onet_time
        }

        # Final output
        debug_info['final'] = {
            'num_boxes': len(onet_boxes),
            'boxes': onet_boxes.copy() if len(onet_boxes) > 0 else np.array([]),
            'landmarks': landmarks.copy() if len(landmarks) > 0 else np.array([]),
            'total_time_ms': pnet_time + rnet_time + onet_time
        }

        return onet_boxes, landmarks, debug_info

    def detect_batch(self, frames):
        """
        Detect faces across multiple frames.

        Args:
            frames: List of images, each (H, W, 3) in BGR format

        Returns:
            List of (bboxes, landmarks) tuples, one per frame
        """
        # Process frames individually for now
        # TODO: Implement true batch processing for ONNX
        return [self.detect(frame) for frame in frames]

    def get_active_provider(self):
        """Get the active execution provider being used."""
        return self.pnet.get_providers()[0]
