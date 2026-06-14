# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
AutoSolve Video Feature Extractor.
Extracts motion, zoom, distortion, and noise metrics directly from raw video files
using OpenCV (cv2) and NumPy to generate metadata for training the Neural Engine.
"""

import os
import json
import argparse
from typing import Dict, List, Optional
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def extract_features_from_video(video_path: str, sample_rate: int = 5) -> Optional[Dict]:
    """
    Open video and extract average motion, zoom, distortion, and noise metrics.
    """
    if not OPENCV_AVAILABLE:
        print("Error: opencv-python is not installed. Please run: pip install opencv-python")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0 or width <= 0 or height <= 0:
        print(f"Error: Invalid metadata for {video_path}")
        cap.release()
        return None

    # We will sample frames for speed
    step = max(1, frame_count // 30)  # Analyze ~30 frames across the clip

    prev_gray = None
    motions = []
    divergences = []
    noises = []
    curvatures = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # Downsample frame for fast processing
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            # 1. Noise/Grain estimation (variance of high-frequency pixels)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            high_freq = cv2.absdiff(gray, blurred)
            noise_val = float(np.mean(high_freq))
            noises.append(noise_val)

            # 2. Curvature (distortion proxy - using edge counts and deviation)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            if lines is not None:
                # Estimate curvature based on variance of line orientations
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    angles.append(angle)
                curvatures.append(float(np.var(angles)) if angles else 0.0)
            else:
                curvatures.append(0.0)

            # 3. Dense Optical Flow for motion & divergence (zoom)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]

                # Motion magnitude
                mag = np.sqrt(flow_x**2 + flow_y**2)
                motions.append(float(np.mean(mag)))

                # Divergence (zoom proxy)
                grad_x = np.gradient(flow_x, axis=1)
                grad_y = np.gradient(flow_y, axis=0)
                div = grad_x + grad_y
                divergences.append(float(np.mean(div)))

            prev_gray = gray

        frame_idx += 1

    cap.release()

    # Calculate average stats
    mean_motion = float(np.mean(motions)) if motions else 0.0
    max_motion = float(np.max(motions)) if motions else 0.0
    mean_zoom = float(np.mean(np.abs(divergences))) if divergences else 0.0
    mean_noise = float(np.mean(noises)) if noises else 0.0
    mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0

    # Classify motion class
    if mean_motion > 3.0:
        motion_class = 'HIGH'
    elif mean_motion > 0.8:
        motion_class = 'MEDIUM'
    else:
        motion_class = 'LOW'

    return {
        "clip_name": os.path.splitext(os.path.basename(video_path))[0],
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 24.0,
        "frame_count": frame_count,
        "mean_motion": mean_motion,
        "max_motion": max_motion,
        "motion_class": motion_class,
        "zoom_divergence": mean_zoom,
        "distortion_factor": mean_curvature,
        "grain_noise": mean_noise,
        "noise_ratio": mean_noise
    }


def main():
    parser = argparse.ArgumentParser(description="Extract video features directly from raw clips.")
    parser.add_argument("--clips-dir", default="ml/clips", help="Directory containing raw video clips")
    parser.add_argument("--out-dir", default="ml/data/raw", help="Directory to save extracted JSON features")
    args = parser.parse_args()

    if not os.path.exists(args.clips_dir):
        print(f"Clips directory does not exist: {args.clips_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    supported = {'.mp4', '.mov', '.avi', '.mkv', '.ogg', '.webm'}
    files = [f for f in os.listdir(args.clips_dir) if os.path.splitext(f)[1].lower() in supported]

    print(f"Found {len(files)} clips to process in {args.clips_dir}")

    for f in files:
        out_name = f"{os.path.splitext(f)[0]}_video_meta.json"
        out_path = os.path.join(args.out_dir, out_name)
        
        # Skip if already exists (resume support)
        if os.path.exists(out_path):
            print(f"  -> Skipping existing features: {out_name}")
            continue
            
        path = os.path.join(args.clips_dir, f)
        print(f"Extracting features from {f}...")
        feats = extract_features_from_video(path)
        if feats:
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(feats, fh, indent=4)
            print(f"  Saved features to {out_name}")


if __name__ == "__main__":
    main()
