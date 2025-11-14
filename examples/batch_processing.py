#!/usr/bin/env python3
"""
Example: Batch video processing with PyMTCNN

This example demonstrates how to use PyMTCNN for batch video processing with
cross-frame batching for maximum throughput.

Best for: offline video analysis, maximum throughput scenarios.

Cross-Platform Performance (batch_size=4):
- Apple Silicon (M1/M2/M3): 34.26 FPS with CoreML backend
- NVIDIA GPU: 50+ FPS with ONNX + CUDA backend
- CPU: 5-10 FPS with ONNX backend
"""

import cv2
import time
from pathlib import Path
from pymtcnn import MTCNN


def main():
    print("=" * 80)
    print("PyMTCNN - Batch Video Processing Example")
    print("=" * 80)

    # Initialize detector with automatic backend selection
    print("\n1. Initializing PyMTCNN...")
    detector = MTCNN(
        backend='auto',             # Auto-select best backend (or 'cuda', 'coreml', 'cpu')
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        verbose=True
    )
    print("   Detector initialized successfully!")
    print(f"   Active backend: {detector.get_backend_info()}")

    # Load video
    print("\n2. Loading video...")
    video_path = Path("test_video.mp4")

    if not video_path.exists():
        print(f"   ERROR: {video_path} not found!")
        print("   Please provide a test video named 'test_video.mp4'")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"   ERROR: Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   Video loaded:")
    print(f"     Resolution: {width}×{height}")
    print(f"     FPS: {fps:.2f}")
    print(f"     Total frames: {total_frames}")

    # Process video with batch processing
    print("\n3. Processing video with cross-frame batching...")
    print(f"   Using batch size: 4 (optimal for maximum throughput)")

    batch_size = 4
    all_results = []
    frames_processed = 0
    total_time = 0

    # Limit to first 30 frames for demo
    max_frames = min(30, total_frames)

    while frames_processed < max_frames:
        # Read batch of frames
        batch_frames = []
        for _ in range(batch_size):
            if frames_processed + len(batch_frames) >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            batch_frames.append(frame)

        if len(batch_frames) == 0:
            break

        # Process batch
        start_time = time.time()
        batch_results = detector.detect_batch(batch_frames)
        elapsed = time.time() - start_time

        total_time += elapsed
        frames_processed += len(batch_frames)
        all_results.extend(batch_results)

        # Display progress
        fps_current = len(batch_frames) / elapsed if elapsed > 0 else 0
        print(f"   Processed frames {frames_processed - len(batch_frames) + 1}-{frames_processed}: "
              f"{elapsed * 1000:.1f} ms ({fps_current:.2f} FPS)")

    cap.release()

    # Display results
    print("\n4. Processing Results:")
    print(f"   Total frames processed: {frames_processed}")
    print(f"   Total processing time: {total_time:.2f} s")
    print(f"   Average time per frame: {total_time * 1000 / frames_processed:.1f} ms")
    print(f"   Average FPS: {frames_processed / total_time:.2f}")

    # Face detection statistics
    total_faces = sum(len(bboxes) for bboxes, _ in all_results)
    frames_with_faces = sum(1 for bboxes, _ in all_results if len(bboxes) > 0)

    print("\n5. Face Detection Statistics:")
    print(f"   Total faces detected: {total_faces}")
    print(f"   Frames with faces: {frames_with_faces}/{frames_processed} ({frames_with_faces / frames_processed * 100:.1f}%)")
    print(f"   Average faces per frame: {total_faces / frames_processed:.2f}")

    # Per-frame details
    print("\n6. Per-Frame Details:")
    for i, (bboxes, landmarks) in enumerate(all_results[:10]):  # Show first 10
        print(f"   Frame {i+1}: {len(bboxes)} face(s) detected")
        if len(bboxes) > 0:
            for j, bbox in enumerate(bboxes):
                x, y, w, h, conf = bbox
                print(f"     Face {j+1}: ({x:.0f}, {y:.0f}) {w:.0f}×{h:.0f}, conf={conf:.3f}")

    if len(all_results) > 10:
        print(f"   ... (showing first 10 of {len(all_results)} frames)")

    print("\n" + "=" * 80)
    print("Example complete!")
    print(f"Batch processing achieved {frames_processed / total_time:.2f} FPS")
    print("=" * 80)


if __name__ == "__main__":
    main()
