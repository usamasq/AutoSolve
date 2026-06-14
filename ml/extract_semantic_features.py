# SPDX-FileCopyrightText: 2026 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ml/extract_semantic_features.py — Runs YOLOv8-seg or SAM 2 semantic masking
over video frames, extracts normalized dynamic object boundaries, and writes
lightweight JSON features metadata for model training.
"""

import os
import sys
import json
import argparse

# Try imports for YOLOv8
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO moving classes (considered dynamic for camera tracking)
DYNAMIC_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
}


def process_video_semantic_features(video_path: str, out_path: str, model_path: str = "autosolve/models/yolov8n-seg.pt"):
    """Scan video frames, run YOLO segmenter, and output normalized bounding boxes."""
    if not OPENCV_AVAILABLE:
        print("Error: OpenCV (opencv-python) is required to parse video frames.")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Processing '{clip_name}' ({width}x{height}, {frame_count} frames)...")

    # Load YOLO model
    if not YOLO_AVAILABLE:
        raise ImportError(
            "The 'ultralytics' library is required for YOLOv8 semantic segmentation but is not installed. "
            "Please install it using 'pip install ultralytics'."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_to_load = model_path
    
    # Resolve weights path if running from subfolder (like ml/) or root
    if not os.path.exists(model_to_load):
        alt_path = os.path.join("..", model_path)
        if os.path.exists(alt_path):
            model_to_load = alt_path
        elif os.path.exists("yolov8n-seg.pt"):
            model_to_load = "yolov8n-seg.pt"
        else:
            print(f"Warning: YOLO weights at '{model_path}' not found locally. Attempting to download/load 'yolov8n-seg.pt'...")
            model_to_load = "yolov8n-seg.pt"

    # Global cache for the loaded YOLO segmenter model
    global _yolo_model_cache
    if '_yolo_model_cache' not in globals():
        _yolo_model_cache = {}

    if model_to_load in _yolo_model_cache:
        model = _yolo_model_cache[model_to_load]
    else:
        try:
            model = YOLO(model_to_load).to(device)
            _yolo_model_cache[model_to_load] = model
            print(f"Successfully loaded YOLO segmenter ({model_to_load}) on device: {device}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize YOLO model: {e}.\n"
                "Real semantic detections are required for serious training. Please download 'yolov8n-seg.pt' "
                "and place it at 'autosolve/models/yolov8n-seg.pt' manually."
            ) from e


    frames_metadata = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        
        # Run YOLOv8 segmenter on frame
        results = model(frame, verbose=False)[0]
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = results.names[cls_id]
                conf = float(box.conf[0].item())
                
                # Get normalized bounding box coordinates
                # xyxyn: normalized box coordinates [x1, y1, x2, y2]
                xyxyn = box.xyxyn[0].cpu().numpy().tolist()
                
                is_dynamic = cls_name in DYNAMIC_CLASSES
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "box": [round(coord, 4) for coord in xyxyn],
                    "is_dynamic": is_dynamic
                })

        frames_metadata.append({
            "frame_idx": frame_idx,
            "detections": detections
        })
        frame_idx += 1

    cap.release()

    # Save to json file
    output_data = {
        "clip_name": clip_name,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frames": frames_metadata
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Semantic metadata successfully saved to '{out_path}'")
    return True


def run_collection(args):
    """Scan clips directory and run feature extraction on all clips."""
    if not os.path.exists(args.clips_dir):
        print(f"Clips directory '{args.clips_dir}' does not exist.")
        return

    clips = [
        f for f in os.listdir(args.clips_dir)
        if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
    ]

    if not clips:
        print(f"No video files found in '{args.clips_dir}'")
        return

    print(f"Found {len(clips)} clips for semantic feature extraction.")

    for clip in clips:
        video_path = os.path.join(args.clips_dir, clip)
        out_name = f"{os.path.splitext(clip)[0]}_semantic_meta.json"
        out_path = os.path.join(args.out_dir, out_name)
        process_video_semantic_features(video_path, out_path, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoSolve YOLOv8 Semantic Feature Extractor")
    parser.add_argument("--clips-dir", default="ml/clips", help="Directory containing raw video clips")
    parser.add_argument("--out-dir", default="ml/data/raw", help="Directory to output semantic meta JSON logs")
    parser.add_argument("--model-path", default="autosolve/models/yolov8n-seg.pt", help="Path to YOLOv8 segmenter model weights")
    
    parsed_args = parser.parse_args()
    run_collection(parsed_args)
