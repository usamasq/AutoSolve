import sys
import os
import json
import socket
import traceback

# Add worker directory and project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check availability of PyTorch and OpenCV
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def handle_ping(args):
    return {
        "ok": True,
        "torch": TORCH_AVAILABLE,
        "cuda": CUDA_AVAILABLE,
        "cv2": CV2_AVAILABLE,
        "device": "cuda" if CUDA_AVAILABLE else "cpu"
    }


def handle_cotrack(args, progress_callback=None):
    if not TORCH_AVAILABLE:
        return {"ok": False, "error": "PyTorch is not installed in the external Python environment."}
    
    video_path = args.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return {"ok": False, "error": f"Invalid video path: {video_path}"}
        
    grid_size = args.get("grid_size", 8)
    
    # Lazy import to avoid loading models on server startup
    from cotracker_runner import extract_trajectories
    
    try:
        trajectories, meta = extract_trajectories(video_path, grid_size=grid_size, progress_callback=progress_callback)
        return {
            "ok": True,
            "trajectories": trajectories,
            "meta": meta
        }
    except Exception as e:
        return {"ok": False, "error": f"CoTracker tracking error: {str(e)}", "trace": traceback.format_exc()}


def handle_sam2_mask(args, progress_callback=None):
    video_path = args.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return {"ok": False, "error": f"Invalid video path: {video_path}"}
        
    confidence = args.get("confidence", 0.25)
    
    from sam2_runner import segment_video_objects
    
    try:
        masks = segment_video_objects(video_path, confidence=confidence, progress_callback=progress_callback)
        return {
            "ok": True,
            "masks": masks
        }
    except Exception as e:
        return {"ok": False, "error": f"Masking error: {str(e)}", "trace": traceback.format_exc()}


def handle_precision_solve(args):
    obs_data = args.get("obs_data")
    init_cameras = args.get("init_cameras")
    init_points = args.get("init_points")
    init_f = args.get("init_f", 1.0)
    init_k1 = args.get("init_k1", 0.0)
    init_k2 = args.get("init_k2", 0.0)
    aspect_ratio = args.get("aspect_ratio", 1.0)
    
    if not obs_data or init_cameras is None or init_points is None:
        return {"ok": False, "error": "Missing initial solve data arrays."}
        
    import numpy as np
    from ba_solver import solve_precision_bundle
    
    try:
        obs_data_list = obs_data
        init_cameras_arr = np.array(init_cameras, dtype=np.float32)
        init_points_arr = np.array(init_points, dtype=np.float32)
        
        opt_cameras, opt_points, f, k1, k2 = solve_precision_bundle(
            obs_data_list, init_cameras_arr, init_points_arr, init_f, init_k1, init_k2, aspect_ratio
        )
        
        return {
            "ok": True,
            "cameras": opt_cameras,
            "points": opt_points.tolist(),
            "f": float(f),
            "k1": float(k1),
            "k2": float(k2)
        }
    except Exception as e:
        return {"ok": False, "error": f"Solver error: {str(e)}", "trace": traceback.format_exc()}


def make_progress_callback(conn):
    def progress_callback(progress, message):
        try:
            payload = json.dumps({
                "ok": True,
                "type": "progress",
                "progress": float(progress),
                "message": str(message)
            }) + "\n"
            conn.sendall(payload.encode('utf-8'))
        except Exception as e:
            print(f"Error sending progress: {e}")
    return progress_callback


def main():
    port = 47832
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
            
    # Bind socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind(("localhost", port))
    except Exception as e:
        print(f"Error binding to port {port}: {e}", file=sys.stderr)
        sys.exit(1)
        
    server_socket.listen(1)
    print(f"AutoSolve Background Worker Server listening on port {port}...")
    
    try:
        while True:
            conn, addr = server_socket.accept()
            print(f"Connection from: {addr}")
            
            buffer = ""
            try:
                while True:
                    data = conn.recv(4096).decode('utf-8')
                    if not data:
                        break
                        
                    buffer += data
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            req = json.loads(line)
                        except json.JSONDecodeError:
                            conn.sendall(json.dumps({"ok": False, "error": "Invalid JSON payload"}).encode('utf-8') + b"\n")
                            continue
                            
                        cmd = req.get("cmd")
                        print(f"Received command: {cmd}")
                        
                        if cmd == "ping":
                            resp = handle_ping(req)
                        elif cmd == "cotrack":
                            resp = handle_cotrack(req, progress_callback=make_progress_callback(conn))
                        elif cmd == "sam2_mask":
                            resp = handle_sam2_mask(req, progress_callback=make_progress_callback(conn))
                        elif cmd == "precision_solve":
                            resp = handle_precision_solve(req)
                        elif cmd == "shutdown":
                            conn.sendall(json.dumps({"ok": True, "message": "Shutting down"}).encode('utf-8') + b"\n")
                            conn.close()
                            print("Received shutdown command. Exiting.")
                            return
                        else:
                            resp = {"ok": False, "error": f"Unknown command: {cmd}"}
                            
                        # Send reply
                        conn.sendall(json.dumps(resp).encode('utf-8') + b"\n")
            except Exception as e:
                print(f"Connection handler error: {e}")
                traceback.print_exc()
            finally:
                conn.close()
    finally:
        server_socket.close()


if __name__ == '__main__':
    main()
