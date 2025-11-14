#!/usr/bin/env python3
"""
Example: S1 Integration Pattern for PyMTCNN

This example demonstrates the recommended pattern for integrating PyMTCNN
into the S1 video processing pipeline.

Architecture:
    S1 → PyMTCNN (face detection) → PyFaceAU (AU extraction) → S1 (analysis)

Cross-Platform Support:
- Automatically uses best available backend (CUDA > CoreML > CPU)
- Apple Silicon: CoreML backend (34.26 FPS)
- NVIDIA GPU: ONNX + CUDA backend (50+ FPS)
- CPU fallback: ONNX backend (5-10 FPS)

Best practices:
- Use batch processing with batch_size=4 for maximum throughput
- Process videos in chunks to manage memory
- Cache face detections if processing multiple times
"""

import cv2
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from pymtcnn import MTCNN


class S1VideoProcessor:
    """
    Example S1 video processor using PyMTCNN for face detection.

    This class demonstrates the recommended integration pattern for S1.
    Works on any platform with automatic backend selection.
    """

    def __init__(self, batch_size: int = 4, backend: str = 'auto'):
        """
        Initialize S1 video processor.

        Args:
            batch_size: Number of frames to process in each batch (default: 4)
            backend: Backend to use ('auto', 'cuda', 'coreml', 'cpu')
        """
        self.batch_size = batch_size
        self.mtcnn = MTCNN(
            backend=backend,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            verbose=False  # Silent for production use
        )

        backend_info = self.mtcnn.get_backend_info()
        print(f"S1VideoProcessor initialized with batch_size={batch_size}, backend={backend_info}")

    def process_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """
        Process entire video and extract face detections.

        Args:
            video_path: Path to input video file

        Returns:
            List of detection results per frame
        """
        print(f"\nProcessing video: {video_path.name}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video info: {width}×{height}, {fps:.2f} FPS, {total_frames} frames")

        # Process video in batches
        all_results = []
        frames_processed = 0
        total_time = 0

        while frames_processed < total_frames:
            # Read batch of frames
            batch_frames = []
            frame_indices = []

            for _ in range(self.batch_size):
                if frames_processed + len(batch_frames) >= total_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                batch_frames.append(frame)
                frame_indices.append(frames_processed + len(batch_frames) - 1)

            if len(batch_frames) == 0:
                break

            # Detect faces in batch
            start_time = time.time()
            batch_results = self.mtcnn.detect_batch(batch_frames)
            elapsed = time.time() - start_time

            total_time += elapsed
            frames_processed += len(batch_frames)

            # Format results
            for frame_idx, (bboxes, landmarks) in zip(frame_indices, batch_results):
                result = {
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / fps,
                    'num_faces': len(bboxes),
                    'faces': []
                }

                for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
                    x, y, w, h, conf = bbox
                    result['faces'].append({
                        'face_id': i,
                        'bbox': {
                            'x': float(x),
                            'y': float(y),
                            'width': float(w),
                            'height': float(h),
                            'confidence': float(conf)
                        },
                        'landmarks': {
                            'left_eye': [float(lm[0][0]), float(lm[0][1])],
                            'right_eye': [float(lm[1][0]), float(lm[1][1])],
                            'nose': [float(lm[2][0]), float(lm[2][1])],
                            'left_mouth': [float(lm[3][0]), float(lm[3][1])],
                            'right_mouth': [float(lm[4][0]), float(lm[4][1])]
                        }
                    })

                all_results.append(result)

            # Progress update
            if frames_processed % 100 == 0 or frames_processed == total_frames:
                fps_current = frames_processed / total_time if total_time > 0 else 0
                print(f"  Progress: {frames_processed}/{total_frames} frames ({fps_current:.2f} FPS)")

        cap.release()

        # Summary
        total_faces = sum(r['num_faces'] for r in all_results)
        print(f"\n  Processing complete!")
        print(f"    Total frames: {frames_processed}")
        print(f"    Total time: {total_time:.2f} s")
        print(f"    Average FPS: {frames_processed / total_time:.2f}")
        print(f"    Total faces detected: {total_faces}")

        return all_results

    def process_video_with_cache(self, video_path: Path, cache_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Process video with caching for repeated analysis.

        Args:
            video_path: Path to input video file
            cache_dir: Directory to store cached detections (default: video directory)

        Returns:
            List of detection results per frame
        """
        if cache_dir is None:
            cache_dir = video_path.parent

        cache_file = cache_dir / f"{video_path.stem}_detections.json"

        # Check for cached results
        if cache_file.exists():
            print(f"\nLoading cached detections from: {cache_file.name}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Process video
        results = self.process_video(video_path)

        # Save to cache
        print(f"\nSaving detections to cache: {cache_file.name}")
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results


def main():
    print("=" * 80)
    print("S1 Integration Example - PyMTCNN Face Detection")
    print("=" * 80)

    # Example 1: Basic video processing
    print("\n" + "=" * 80)
    print("Example 1: Basic Video Processing")
    print("=" * 80)

    video_path = Path("test_video.mp4")

    if not video_path.exists():
        print(f"\nERROR: {video_path} not found!")
        print("Please provide a test video named 'test_video.mp4'")
        return

    # Initialize S1 processor
    processor = S1VideoProcessor(batch_size=4)

    # Process video
    results = processor.process_video(video_path)

    # Display sample results
    print("\nSample detection results (first 5 frames):")
    for result in results[:5]:
        print(f"\n  Frame {result['frame_index']} (t={result['timestamp']:.2f}s):")
        print(f"    Faces detected: {result['num_faces']}")
        for face in result['faces']:
            bbox = face['bbox']
            print(f"      Face {face['face_id']}: "
                  f"({bbox['x']:.0f}, {bbox['y']:.0f}) "
                  f"{bbox['width']:.0f}×{bbox['height']:.0f}, "
                  f"conf={bbox['confidence']:.3f}")

    # Example 2: Processing with cache
    print("\n" + "=" * 80)
    print("Example 2: Processing with Cache")
    print("=" * 80)

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    # First run: process and cache
    print("\nFirst run (will cache results):")
    results1 = processor.process_video_with_cache(video_path, cache_dir)

    # Second run: load from cache
    print("\nSecond run (should load from cache):")
    results2 = processor.process_video_with_cache(video_path, cache_dir)

    print(f"\nResults match: {results1 == results2}")

    # Example 3: Export for PyFaceAU integration
    print("\n" + "=" * 80)
    print("Example 3: Export for PyFaceAU Integration")
    print("=" * 80)

    output_file = Path("detections_for_pyfaceau.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetections exported to: {output_file}")
    print("This file can be used as input for PyFaceAU AU extraction")

    print("\n" + "=" * 80)
    print("S1 Integration Example Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use these detections as input to PyFaceAU for AU extraction")
    print("2. Feed PyFaceAU results to S1 for final analysis")
    print("3. Consider adding face tracking for temporal consistency")


if __name__ == "__main__":
    main()
