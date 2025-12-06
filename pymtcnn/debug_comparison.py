"""
Debug comparison module for PyMTCNN vs C++ MTCNN.

Provides parsers for C++ debug files and stage-by-stage comparison functions.
"""

import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =============================================================================
# C++ Debug File Parsers
# =============================================================================

def parse_cpp_pnet_boxes(filepath: str) -> np.ndarray:
    """
    Parse cpp_pnet_all_boxes.txt

    Format:
        C++ PNet: 238 boxes after cross-scale NMS
        Box 0: x1=277 y1=610 x2=888 y2=1221 w=611 h=611 score=1
        ...

    Returns: (N, 5) array with [x, y, w, h, score] (converted from x1,y1,x2,y2)
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('Box'):
                # Parse: Box N: x1=INT y1=INT x2=INT y2=INT w=INT h=INT score=FLOAT
                match = re.search(
                    r'x1=(-?\d+)\s+y1=(-?\d+)\s+x2=(-?\d+)\s+y2=(-?\d+)\s+w=(-?\d+)\s+h=(-?\d+)\s+score=([0-9.e+-]+)',
                    line
                )
                if match:
                    x1, y1, x2, y2 = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
                    w, h = int(match.group(5)), int(match.group(6))
                    score = float(match.group(7))
                    # Use x1, y1, w, h format
                    boxes.append([x1, y1, w, h, score])
    return np.array(boxes) if boxes else np.empty((0, 5))


def parse_cpp_rnet_output(filepath: str) -> np.ndarray:
    """
    Parse cpp_rnet_output.txt

    Format:
        C++ RNet: 18 boxes OUTPUT from RNet (BEFORE square for ONet)
        Box 0: x1=1414.16 y1=649.968 x2=1802.8 y2=1168.64 w=388.64 h=518.675 score=0.999498
        ...

    Returns: (N, 5) array with [x, y, w, h, score]
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('Box'):
                match = re.search(
                    r'x1=([0-9.e+-]+)\s+y1=([0-9.e+-]+)\s+x2=([0-9.e+-]+)\s+y2=([0-9.e+-]+)\s+w=([0-9.e+-]+)\s+h=([0-9.e+-]+)\s+score=([0-9.e+-]+)',
                    line
                )
                if match:
                    x1, y1 = float(match.group(1)), float(match.group(2))
                    w, h = float(match.group(5)), float(match.group(6))
                    score = float(match.group(7))
                    boxes.append([x1, y1, w, h, score])
    return np.array(boxes) if boxes else np.empty((0, 5))


def parse_cpp_before_onet(filepath: str) -> np.ndarray:
    """
    Parse cpp_before_onet.txt (squared boxes for ONet input)

    Format:
        C++ Boxes BEFORE ONet (after RNet square/rectify):
        Total boxes: 18
        Box 0: x=1349 y=649 w=518 h=518
            x2=1867 y2=1167
        ...

    Returns: (N, 4) array with [x, y, w, h]
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('Box'):
                match = re.search(r'x=(-?\d+)\s+y=(-?\d+)\s+w=(-?\d+)\s+h=(-?\d+)', line)
                if match:
                    x, y, w, h = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
                    boxes.append([x, y, w, h])
    return np.array(boxes) if boxes else np.empty((0, 4))


def parse_cpp_onet_boxes(filepath: str) -> np.ndarray:
    """
    Parse cpp_onet_boxes.txt (ONet output before calibration)

    Format:
        C++ ONet: 2 boxes after final NMS (BEFORE bbox correction)
        Box 0: x=340.759 y=651.005 w=379.609 h=480.769 score=1
        ...

    Returns: (N, 5) array with [x, y, w, h, score]
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('Box'):
                match = re.search(
                    r'x=([0-9.e+-]+)\s+y=([0-9.e+-]+)\s+w=([0-9.e+-]+)\s+h=([0-9.e+-]+)\s+score=([0-9.e+-]+)',
                    line
                )
                if match:
                    x, y, w, h = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    score = float(match.group(5))
                    boxes.append([x, y, w, h, score])
    return np.array(boxes) if boxes else np.empty((0, 5))


def parse_cpp_final_bbox(filepath: str) -> np.ndarray:
    """
    Parse cpp_mtcnn_final_bbox.txt

    Format:
        337.912 769.227 391.87 372.644 1
        1416.43 740.828 392.949 418.833 0.999992

    Returns: (N, 5) array with [x, y, w, h, score]
    """
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, w, h, score = map(float, parts[:5])
                boxes.append([x, y, w, h, score])
    return np.array(boxes) if boxes else np.empty((0, 5))


def parse_cpp_landmarks(filepath: str) -> np.ndarray:
    """
    Parse cpp_mtcnn_5pt_landmarks.txt

    Format:
        # C++ MTCNN 5-point landmarks
        # Format: face_idx left_eye_x left_eye_y right_eye_x right_eye_y ...
        0 0.308104 0.354031 0.631406 0.329447 0.470538 0.5327 0.359603 0.738768 0.655086 0.72018
        ...

    Returns: (N, 5, 2) array with normalized [0-1] landmarks
             Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    landmarks = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 11:
                # Skip face_idx, get 5 pairs of x,y
                lms = []
                for i in range(5):
                    x = float(parts[1 + i*2])
                    y = float(parts[2 + i*2])
                    lms.append([x, y])
                landmarks.append(lms)
    return np.array(landmarks) if landmarks else np.empty((0, 5, 2))


# =============================================================================
# Comparison Utilities
# =============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.
    """
    x1_1, y1_1, w1, h1 = box1[:4]
    x1_2, y1_2, w2, h2 = box2[:4]

    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def find_best_match(target_box: np.ndarray, candidates: np.ndarray) -> Tuple[int, float]:
    """
    Find best matching box using IoU.
    Returns: (index, iou_score) or (-1, 0) if no match
    """
    if len(candidates) == 0:
        return -1, 0.0

    best_idx = -1
    best_iou = 0.0

    for i, cand in enumerate(candidates):
        iou = compute_iou(target_box, cand)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx, best_iou


def compute_box_diff(box1: np.ndarray, box2: np.ndarray) -> Dict:
    """
    Compute detailed difference between two boxes [x, y, w, h, ...].
    """
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dw = abs(w1 - w2)
    dh = abs(h1 - h2)

    # Center distance
    cx1, cy1 = x1 + w1/2, y1 + h1/2
    cx2, cy2 = x2 + w2/2, y2 + h2/2
    center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    return {
        'dx': dx,
        'dy': dy,
        'dw': dw,
        'dh': dh,
        'center_dist': center_dist,
        'iou': compute_iou(box1, box2),
        'max_coord_diff': max(dx, dy)
    }


def compare_box_sets(cpp_boxes: np.ndarray, py_boxes: np.ndarray,
                     iou_threshold: float = 0.5) -> Dict:
    """
    Compare two sets of boxes, matching by IoU.
    """
    result = {
        'cpp_count': len(cpp_boxes),
        'py_count': len(py_boxes),
        'count_match': len(cpp_boxes) == len(py_boxes),
        'matched_pairs': [],
        'unmatched_cpp': [],
        'unmatched_py': list(range(len(py_boxes))),
        'max_error_px': 0.0,
        'mean_error_px': 0.0
    }

    if len(cpp_boxes) == 0 or len(py_boxes) == 0:
        return result

    errors = []
    matched_py = set()

    for cpp_idx, cpp_box in enumerate(cpp_boxes):
        best_idx, best_iou = find_best_match(cpp_box, py_boxes)

        if best_iou >= iou_threshold and best_idx not in matched_py:
            py_box = py_boxes[best_idx]
            diff = compute_box_diff(cpp_box, py_box)

            result['matched_pairs'].append({
                'cpp_idx': cpp_idx,
                'py_idx': best_idx,
                'cpp_box': cpp_box.tolist(),
                'py_box': py_box.tolist(),
                'iou': best_iou,
                **diff
            })

            errors.append(diff['center_dist'])
            matched_py.add(best_idx)
        else:
            result['unmatched_cpp'].append(cpp_idx)

    result['unmatched_py'] = [i for i in range(len(py_boxes)) if i not in matched_py]

    if errors:
        result['max_error_px'] = max(errors)
        result['mean_error_px'] = np.mean(errors)

    return result


# =============================================================================
# Main Comparison Function
# =============================================================================

def compare_mtcnn_stages(image: np.ndarray, detector,
                         cpp_debug_dir: str = "/tmp") -> Dict:
    """
    Run PyMTCNN and compare against C++ debug files at each stage.

    Args:
        image: BGR image array
        detector: MTCNN detector instance (with debug_mode=True)
        cpp_debug_dir: Directory containing C++ debug files

    Returns:
        Dict with comparison results for each stage
    """
    import cv2

    # Run PyMTCNN with debug mode
    bboxes, landmarks, debug_info = detector.detect(image, return_debug=True)

    result = {
        'summary': {
            'first_divergence_stage': None,
            'max_divergence_px': 0.0,
            'matched': True
        },
        'stages': {}
    }

    cpp_dir = Path(cpp_debug_dir)

    # --- PNet Comparison ---
    pnet_file = cpp_dir / "cpp_pnet_all_boxes.txt"
    if pnet_file.exists():
        cpp_pnet = parse_cpp_pnet_boxes(str(pnet_file))
        py_pnet = debug_info.get('pnet', {}).get('boxes', np.empty((0, 4)))

        # Add dummy score column to py_pnet for comparison
        if len(py_pnet) > 0 and py_pnet.shape[1] == 4:
            py_pnet = np.hstack([py_pnet, np.ones((len(py_pnet), 1))])

        pnet_cmp = compare_box_sets(cpp_pnet, py_pnet, iou_threshold=0.3)
        pnet_cmp['stage'] = 'pnet'
        result['stages']['pnet'] = pnet_cmp

        if pnet_cmp['max_error_px'] > 2.0:
            if result['summary']['first_divergence_stage'] is None:
                result['summary']['first_divergence_stage'] = 'pnet'
            result['summary']['matched'] = False

    # --- RNet Comparison ---
    rnet_file = cpp_dir / "cpp_rnet_output.txt"
    if rnet_file.exists():
        cpp_rnet = parse_cpp_rnet_output(str(rnet_file))
        py_rnet = debug_info.get('rnet', {}).get('boxes', np.empty((0, 4)))

        if len(py_rnet) > 0 and py_rnet.shape[1] == 4:
            py_rnet = np.hstack([py_rnet, np.ones((len(py_rnet), 1))])

        rnet_cmp = compare_box_sets(cpp_rnet, py_rnet, iou_threshold=0.5)
        rnet_cmp['stage'] = 'rnet'
        result['stages']['rnet'] = rnet_cmp

        if rnet_cmp['max_error_px'] > 2.0:
            if result['summary']['first_divergence_stage'] is None:
                result['summary']['first_divergence_stage'] = 'rnet'
            result['summary']['matched'] = False

    # --- ONet Comparison (before calibration) ---
    onet_file = cpp_dir / "cpp_onet_boxes.txt"
    if onet_file.exists():
        cpp_onet = parse_cpp_onet_boxes(str(onet_file))
        py_onet = debug_info.get('onet', {}).get('bbox_before_calibration', np.empty((0, 4)))

        if len(py_onet) > 0 and py_onet.shape[1] == 4:
            py_onet = np.hstack([py_onet, np.ones((len(py_onet), 1))])

        onet_cmp = compare_box_sets(cpp_onet, py_onet, iou_threshold=0.7)
        onet_cmp['stage'] = 'onet'
        result['stages']['onet'] = onet_cmp

        if onet_cmp['max_error_px'] > 2.0:
            if result['summary']['first_divergence_stage'] is None:
                result['summary']['first_divergence_stage'] = 'onet'
            result['summary']['matched'] = False

    # --- Final Comparison ---
    final_file = cpp_dir / "cpp_mtcnn_final_bbox.txt"
    landmarks_file = cpp_dir / "cpp_mtcnn_5pt_landmarks.txt"

    if final_file.exists():
        cpp_final = parse_cpp_final_bbox(str(final_file))
        py_final = bboxes

        if len(py_final) > 0 and py_final.shape[1] == 4:
            py_final = np.hstack([py_final, np.ones((len(py_final), 1))])

        final_cmp = compare_box_sets(cpp_final, py_final, iou_threshold=0.7)
        final_cmp['stage'] = 'final'

        # Add landmark comparison
        if landmarks_file.exists():
            cpp_lms = parse_cpp_landmarks(str(landmarks_file))
            py_lms_raw = debug_info.get('onet', {}).get('landmarks_raw_normalized', np.empty((0, 5, 2)))

            if len(cpp_lms) > 0 and len(py_lms_raw) > 0:
                # Compare raw normalized landmarks for best matching faces
                lm_errors = []
                for pair in final_cmp['matched_pairs']:
                    cpp_idx = pair['cpp_idx']
                    py_idx = pair['py_idx']

                    if cpp_idx < len(cpp_lms) and py_idx < len(py_lms_raw):
                        cpp_lm = cpp_lms[cpp_idx]
                        py_lm = py_lms_raw[py_idx]

                        diff = np.sqrt(np.sum((cpp_lm - py_lm)**2, axis=1))
                        pair['landmark_errors'] = diff.tolist()
                        pair['mean_lm_error'] = float(np.mean(diff))
                        lm_errors.append(np.mean(diff))

                if lm_errors:
                    final_cmp['mean_landmark_error_norm'] = float(np.mean(lm_errors))
                    final_cmp['max_landmark_error_norm'] = float(max(lm_errors))

        result['stages']['final'] = final_cmp

        if final_cmp['max_error_px'] > 2.0:
            if result['summary']['first_divergence_stage'] is None:
                result['summary']['first_divergence_stage'] = 'final'
            result['summary']['matched'] = False

    # Update summary
    max_errors = [s.get('max_error_px', 0) for s in result['stages'].values()]
    result['summary']['max_divergence_px'] = max(max_errors) if max_errors else 0.0

    return result


def print_divergence_report(result: Dict):
    """Print a human-readable divergence report."""
    print("=" * 70)
    print("MTCNN DIVERGENCE REPORT")
    print("=" * 70)

    summary = result['summary']
    print(f"\nFirst divergence at: {summary['first_divergence_stage'] or 'NONE (matched!)'}")
    print(f"Maximum divergence: {summary['max_divergence_px']:.2f} px")
    print(f"Overall match: {'YES' if summary['matched'] else 'NO'}")

    for stage_name in ['pnet', 'rnet', 'onet', 'final']:
        stage = result['stages'].get(stage_name, {})
        if not stage:
            continue

        print(f"\n--- {stage_name.upper()} ---")
        print(f"  C++ boxes: {stage.get('cpp_count', 'N/A')}")
        print(f"  Py boxes:  {stage.get('py_count', 'N/A')}")
        print(f"  Matched:   {len(stage.get('matched_pairs', []))}")
        print(f"  Max error: {stage.get('max_error_px', 0):.2f} px")
        print(f"  Mean error: {stage.get('mean_error_px', 0):.2f} px")

        if stage.get('max_error_px', 0) > 2.0:
            print(f"  ** DIVERGENCE DETECTED **")

        # Show worst matches
        pairs = stage.get('matched_pairs', [])
        if pairs:
            sorted_pairs = sorted(pairs, key=lambda p: p.get('center_dist', 0), reverse=True)
            print(f"  Top 3 worst matches:")
            for pair in sorted_pairs[:3]:
                cpp_box = pair['cpp_box']
                py_box = pair['py_box']
                print(f"    C++ box {pair['cpp_idx']}: [{cpp_box[0]:.1f}, {cpp_box[1]:.1f}, {cpp_box[2]:.1f}, {cpp_box[3]:.1f}]")
                print(f"    Py  box {pair['py_idx']}: [{py_box[0]:.1f}, {py_box[1]:.1f}, {py_box[2]:.1f}, {py_box[3]:.1f}]")
                print(f"    Error: dx={pair['dx']:.2f}, dy={pair['dy']:.2f}, center={pair['center_dist']:.2f}px")

        # Show landmark comparison for final stage
        if stage_name == 'final' and 'mean_landmark_error_norm' in stage:
            print(f"\n  Landmarks (normalized 0-1):")
            print(f"    Mean error: {stage['mean_landmark_error_norm']:.4f}")
            print(f"    Max error:  {stage['max_landmark_error_norm']:.4f}")
