#!/usr/bin/env python3
"""
Test both CoreML and ONNX backends with real image detection
"""
import sys
import cv2
import numpy as np
sys.path.insert(0, '.')

from pymtcnn import MTCNN

# Find a test image
import glob
test_images = glob.glob("/Users/johnwilsoniv/Documents/SplitFace Open3/test_frames/*.jpg")
if not test_images:
    test_images = glob.glob("/Users/johnwilsoniv/Documents/SplitFace Open3/calibration_frames/*.jpg")

if not test_images:
    print("No test images found!")
    sys.exit(1)

test_image = test_images[0]
print(f"Testing with: {test_image}")

# Load test image
img = cv2.imread(test_image)
if img is None:
    print(f"Failed to load image: {test_image}")
    sys.exit(1)

print(f"Image shape: {img.shape}")

# Test 1: Auto-backend (should select CoreML on macOS)
print("\n" + "="*60)
print("Test 1: Auto-backend selection")
print("="*60)
detector_auto = MTCNN(verbose=True)
print(f"Selected backend: {detector_auto.get_backend_info()}")

bboxes, landmarks = detector_auto.detect(img)
print(f"✓ Auto-backend detected {len(bboxes)} faces")
if len(bboxes) > 0:
    print(f"  First face bbox: {bboxes[0]}")
    print(f"  First face landmarks shape: {landmarks[0].shape}")

# Test 2: Force ONNX backend
print("\n" + "="*60)
print("Test 2: Force ONNX backend")
print("="*60)
try:
    detector_onnx = MTCNN(backend='onnx', verbose=True)
    print(f"ONNX backend: {detector_onnx.get_backend_info()}")

    bboxes_onnx, landmarks_onnx = detector_onnx.detect(img)
    print(f"✓ ONNX backend detected {len(bboxes_onnx)} faces")
    if len(bboxes_onnx) > 0:
        print(f"  First face bbox: {bboxes_onnx[0]}")
        print(f"  First face landmarks shape: {landmarks_onnx[0].shape}")

    # Compare results
    if len(bboxes) == len(bboxes_onnx):
        print(f"✓ Both backends detected same number of faces")
    else:
        print(f"⚠ Different face counts: Auto={len(bboxes)}, ONNX={len(bboxes_onnx)}")

except Exception as e:
    print(f"✗ ONNX backend failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Force CoreML backend
print("\n" + "="*60)
print("Test 3: Force CoreML backend")
print("="*60)
try:
    detector_coreml = MTCNN(backend='coreml', verbose=True)
    print(f"CoreML backend: {detector_coreml.get_backend_info()}")

    bboxes_coreml, landmarks_coreml = detector_coreml.detect(img)
    print(f"✓ CoreML backend detected {len(bboxes_coreml)} faces")
    if len(bboxes_coreml) > 0:
        print(f"  First face bbox: {bboxes_coreml[0]}")
        print(f"  First face landmarks shape: {landmarks_coreml[0].shape}")

except Exception as e:
    print(f"✗ CoreML backend failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All backend tests completed!")
print("="*60)
