import os
import sys

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def segment_video_objects(video_path: str, confidence: float = 0.25, progress_callback=None):
    """
    Run semantic segmentation/detection over a video to locate dynamic objects (e.g. people, cars).
    Returns:
        masks_by_frame: Dict mapping frame index (str) to list of normalized bounding boxes [x1, y1, x2, y2]
    """
    if progress_callback:
        progress_callback(0.0, "Loading YOLOv8 segmentation model...")

    try:
        import torch
        import cv2
        # We use ultralytics as it supports both YOLOv8-seg (7MB) and SAM 2 Tiny (80MB) via a unified interface
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(f"Missing required ML dependencies for segmentation in external Python: {str(e)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load the smallest, most efficient segmentation model (YOLOv8 Nano Segment - ~7MB)
    # Loaded locally from the bundled models directory.
    try:
        models_dir = os.environ.get("AUTOSOLVE_MODELS_DIR")
        if not models_dir or not os.path.exists(models_dir):
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        yolo_model_path = os.path.join(models_dir, "yolov8n-seg.pt")
        model = YOLO(yolo_model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO segmentation model from {yolo_model_path}: {str(e)}")

    # Move model to optimal device
    if torch.cuda.is_available():
        model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model.to("mps")

    # Dynamic classes to mask out: person (0), bicycle (1), car (2), motorcycle (3), airplane (4),
    # bus (5), train (6), truck (7), boat (8), bird (14), cat (15), dog (16), horse (17), sheep (18), cow (19)
    dynamic_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19}

    masks_by_frame = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames, f"YOLO masking frame {frame_idx}/{total_frames}...")

        # Run inference on single frame
        results = model(frame, verbose=False, conf=confidence)
        if not results:
            masks_by_frame[str(frame_idx)] = []
            frame_idx += 1
            continue

        boxes = []
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in dynamic_classes:
                    # Get coordinates xyxy
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    # Normalize
                    nx1 = float(x1 / width)
                    ny1 = float(y1 / height)
                    nx2 = float(x2 / width)
                    ny2 = float(y2 / height)
                    boxes.append([nx1, ny1, nx2, ny2])

        masks_by_frame[str(frame_idx)] = boxes
        frame_idx += 1

    cap.release()
    if progress_callback:
        progress_callback(1.0, "YOLO masking complete.")
    return masks_by_frame
