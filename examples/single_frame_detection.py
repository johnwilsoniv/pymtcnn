#!/usr/bin/env python3
"""
Example: Single-frame face detection with PyMTCNN

This example demonstrates how to use PyMTCNN for real-time, per-frame face detection.
Best for: webcam feeds, real-time processing, lowest latency scenarios.

Performance: 31.88 FPS (31.4 ms/frame)
"""

import cv2
import time
from pathlib import Path
from pymtcnn import CoreMLMTCNN


def main():
    print("=" * 80)
    print("PyMTCNN - Single Frame Detection Example")
    print("=" * 80)

    # Initialize detector
    print("\n1. Initializing PyMTCNN...")
    detector = CoreMLMTCNN(
        min_face_size=60,          # Minimum face size in pixels
        thresholds=[0.6, 0.7, 0.7], # Detection thresholds
        factor=0.709,               # Image pyramid scale
        verbose=True                # Show loading messages
    )
    print("   Detector initialized successfully!")

    # Load test image
    print("\n2. Loading test image...")
    test_image = Path("test_image.jpg")

    if not test_image.exists():
        print(f"   ERROR: {test_image} not found!")
        print("   Please provide a test image named 'test_image.jpg'")
        return

    img = cv2.imread(str(test_image))
    if img is None:
        print(f"   ERROR: Failed to load {test_image}")
        return

    print(f"   Image loaded: {img.shape[1]}×{img.shape[0]} pixels")

    # Run face detection
    print("\n3. Running face detection...")
    start_time = time.time()
    bboxes, landmarks = detector.detect(img)
    elapsed = time.time() - start_time

    # Display results
    print("\n4. Results:")
    print(f"   Detection time: {elapsed * 1000:.1f} ms")
    print(f"   FPS: {1.0 / elapsed:.2f}")
    print(f"   Faces detected: {len(bboxes)}")

    if len(bboxes) > 0:
        print("\n   Detected faces:")
        for i, bbox in enumerate(bboxes):
            x, y, w, h, conf = bbox
            print(f"   Face {i+1}: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f} pixels, confidence: {conf:.3f}")

        print("\n   Facial landmarks (5 points per face):")
        for i, lm in enumerate(landmarks):
            print(f"   Face {i+1}:")
            print(f"     Left eye:     ({lm[0][0]:.1f}, {lm[0][1]:.1f})")
            print(f"     Right eye:    ({lm[1][0]:.1f}, {lm[1][1]:.1f})")
            print(f"     Nose:         ({lm[2][0]:.1f}, {lm[2][1]:.1f})")
            print(f"     Left mouth:   ({lm[3][0]:.1f}, {lm[3][1]:.1f})")
            print(f"     Right mouth:  ({lm[4][0]:.1f}, {lm[4][1]:.1f})")

    # Visualize results
    print("\n5. Saving visualization...")
    output_img = img.copy()

    # Draw bounding boxes
    for bbox in bboxes:
        x, y, w, h, conf = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence
        label = f"{conf:.3f}"
        cv2.putText(output_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw landmarks
    for lm in landmarks:
        for point in lm:
            cv2.circle(output_img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    output_path = Path("output_single_frame.jpg")
    cv2.imwrite(str(output_path), output_img)
    print(f"   Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
