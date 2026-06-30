import os
import sys

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def extract_trajectories(video_path: str, grid_size: int = 8, target_w: int = 384, target_h: int = 288, progress_callback=None):
    """
    Extract dense point trajectories from a video file using CoTracker.
    Returns:
        trajectories: list of list of [x, y] coordinates
        meta: dictionary of video metadata
    """
    try:
        import torch
        import numpy as np
        import cv2
    except ImportError as e:
        raise ImportError(f"Missing required ML dependencies in external Python: {str(e)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 24.0

    # Calculate aspect-ratio preserving scaled dimensions and padding offsets
    scale = min(target_w / width, target_h / height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and resize preserving aspect ratio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Pad with black pixels to fit target_w x target_h canvas
        padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded_frame[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = frame_resized
        frames.append(padded_frame)
        frame_idx += 1
        if progress_callback and frame_count > 0:
            progress_callback(0.15 * (frame_idx / frame_count), f"Reading video frame {frame_idx}/{frame_count}...")

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    # Build video tensor [1, T, 3, H, W]
    video_tensor = torch.from_numpy(np.stack(frames, axis=0)).float()
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, 3, H, W]
    video_tensor = video_tensor.unsqueeze(0)  # [1, T, 3, H, W]

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    video_tensor = video_tensor.to(device)

    # Load CoTracker locally offline from the bundled cotracker_src and models directory
    try:
        worker_dir = os.path.dirname(os.path.abspath(__file__))
        cotracker_src_dir = os.path.join(worker_dir, "cotracker_src")
        
        models_dir = os.environ.get("AUTOSOLVE_MODELS_DIR")
        if not models_dir or not os.path.exists(models_dir):
            models_dir = os.path.join(os.path.dirname(worker_dir), "models")
        
        checkpoint_path_v3 = os.path.join(models_dir, "cotracker3_offline.pth")
        checkpoint_path_scaled = os.path.join(models_dir, "scaled_offline.pth")
        checkpoint_path_v2 = os.path.join(models_dir, "cotracker2.pth")
        
        if os.path.exists(checkpoint_path_v3):
            checkpoint_path = checkpoint_path_v3
            model_entrypoint = "cotracker3_offline"
            print("AutoSolve: Loading CoTracker3 offline model...")
        elif os.path.exists(checkpoint_path_scaled):
            checkpoint_path = checkpoint_path_scaled
            model_entrypoint = "cotracker3_offline"
            print("AutoSolve: Loading CoTracker3 offline model (scaled_offline)...")
        else:
            checkpoint_path = checkpoint_path_v2
            model_entrypoint = "cotracker2"
            print("AutoSolve: CoTracker3 weights (cotracker3_offline.pth) not found. Falling back to CoTracker2...")
            
        # Load model structure locally and run inference with CPU fallback
        try:
            sys.path.insert(0, cotracker_src_dir)
            try:
                from cotracker.predictor import CoTrackerPredictor
                if model_entrypoint == "cotracker3_offline":
                    model = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
                else:
                    model = CoTrackerPredictor(checkpoint=None, window_len=8, v2=True)
            finally:
                if cotracker_src_dir in sys.path:
                    sys.path.remove(cotracker_src_dir)
            
            # Load state dict
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
                
            # Add model. prefix if missing
            sample_key = next(iter(state_dict.keys()))
            if not sample_key.startswith("model."):
                state_dict = {"model." + k: v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            # Run CoTracker inference
            print(f"AutoSolve: Running CoTracker inference on device: {device}...")
            if progress_callback:
                progress_callback(0.15, "Initializing CoTracker model...")
            with torch.no_grad():
                pred_tracks, pred_visibility = model(video_tensor, grid_size=grid_size, progress_callback=progress_callback)
        except Exception as e:
            if device != "cpu":
                print(f"Warning: GPU/device execution failed ({str(e)}). Falling back to CPU...")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                
                try:
                    device = "cpu"
                    video_tensor = video_tensor.to(device)
                    
                    # Reload model on CPU
                    sys.path.insert(0, cotracker_src_dir)
                    try:
                        from cotracker.predictor import CoTrackerPredictor
                        if model_entrypoint == "cotracker3_offline":
                            model = CoTrackerPredictor(checkpoint=None, window_len=60, v2=False)
                        else:
                            model = CoTrackerPredictor(checkpoint=None, window_len=8, v2=True)
                    finally:
                        if cotracker_src_dir in sys.path:
                            sys.path.remove(cotracker_src_dir)
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                    sample_key = next(iter(state_dict.keys()))
                    if not sample_key.startswith("model."):
                        state_dict = {"model." + k: v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()
                    
                    print("AutoSolve: Running CoTracker inference on CPU fallback...")
                    if progress_callback:
                        progress_callback(0.15, "Initializing CoTracker model on CPU...")
                    with torch.no_grad():
                        pred_tracks, pred_visibility = model(video_tensor, grid_size=grid_size, progress_callback=progress_callback)
                except Exception as cpu_e:
                    raise RuntimeError(f"Failed to run CoTracker on CPU fallback: {str(cpu_e)}") from cpu_e
            else:
                raise RuntimeError(f"Failed to run CoTracker model: {str(e)}") from e
    except Exception as outer_e:
        raise RuntimeError(f"Failed to initialize or run CoTracker: {str(outer_e)}") from outer_e

    # Convert to CPU/numpy
    pred_tracks = pred_tracks.cpu().numpy()[0]  # [T, N, 2]
    pred_visibility = pred_visibility.cpu().numpy()[0]  # [T, N]

    num_tracks = pred_tracks.shape[1]
    trajectories = [[] for _ in range(num_tracks)]

    for t_idx in range(num_tracks):
        for f_idx in range(len(frames)):
            # In Blender, trajectories are typically continuous or end if they are lost.
            # We follow the same pattern: keep coordinates if visible.
            if pred_visibility[f_idx, t_idx]:
                px, py = pred_tracks[f_idx, t_idx]
                # Normalize relative to target width and height
                nx = float((px - pad_w) / new_w)
                ny = float((py - pad_h) / new_h)
                # Clamp between [0, 1]
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                trajectories[t_idx].append((nx, ny))
            else:
                # Truncate trajectory when feature becomes invisible
                break

    meta = {
        "clip_name": os.path.splitext(os.path.basename(video_path))[0],
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count
    }

    if progress_callback:
        progress_callback(1.0, "CoTracker tracking complete.")
    return trajectories, meta
