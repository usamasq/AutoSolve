import os
import sys
import json
import socket
import subprocess
import threading
import time

# Subprocess reference
_worker_process = None
# Active background request thread state
_active_request_thread = None
_request_result = None
_request_error = None
_request_completed = False

def get_server_path():
    """Get absolute path to server.py script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")

def is_port_in_use(port=47832):
    """Check if the worker port is already listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("localhost", port))
            return True
        except socket.error:
            return False

def start_worker(python_path, port=47832):
    """Spawn server.py as an external subprocess."""
    global _worker_process
    
    if is_port_in_use(port):
        print(f"AutoSolve worker already running on port {port}.")
        return True, "Already running"

    if not python_path or not os.path.exists(python_path):
        return False, f"Python path does not exist: {python_path}"

    server_script = get_server_path()
    if not os.path.exists(server_script):
        return False, f"Worker script not found at: {server_script}"

    try:
        # Spawn the worker process in the background. 
        # On Windows, we use creationflags to hide the command prompt window if wanted,
        # but displaying it in standard subprocess is fine.
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0 # SW_HIDE

        _worker_process = subprocess.Popen(
            [python_path, server_script, str(port)],
            startupinfo=startupinfo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True
        )
        
        # Give it a second to bind and start listening
        time.sleep(1.5)
        
        if is_port_in_use(port):
            return True, "Started successfully"
        else:
            return False, "Failed to bind to port after spawning"
    except Exception as e:
        return False, f"Exception while spawning worker: {str(e)}"

def stop_worker(port=47832):
    """Send shutdown command to terminate the server cleanly."""
    global _worker_process
    if not is_port_in_use(port):
        if _worker_process:
            try:
                _worker_process.kill()
            except Exception:
                pass
            _worker_process = None
        return True
        
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("localhost", port))
            payload = json.dumps({"cmd": "shutdown"}) + "\n"
            s.sendall(payload.encode('utf-8'))
            s.recv(1024) # Wait for confirmation
    except Exception as e:
        print(f"Warning: error sending shutdown command to worker: {e}")
        
    if _worker_process:
        try:
            _worker_process.terminate()
            try:
                _worker_process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                print("Worker did not terminate cleanly. Force killing...")
                _worker_process.kill()
                _worker_process.wait(timeout=1.0)
        except Exception as e:
            print(f"Warning: exception while stopping worker process: {e}")
        _worker_process = None
        
    return True

def kill_worker():
    """Force terminate the worker process immediately."""
    global _worker_process
    if _worker_process:
        try:
            _worker_process.kill()
            _worker_process.wait(timeout=1.0)
        except Exception:
            pass
        _worker_process = None

def ping_worker(port=47832):
    """Ping the worker to check connection status and get model metadata."""
    if not is_port_in_use(port):
        return {"ok": False, "error": "Worker port is not active."}
        
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(("localhost", port))
            s.sendall(json.dumps({"cmd": "ping"}) + "\n")
            resp = s.recv(4096).decode('utf-8')
            if "\n" in resp:
                resp = resp.split("\n", 1)[0]
            return json.loads(resp)
    except Exception as e:
        return {"ok": False, "error": f"Failed to connect: {str(e)}"}

def _run_request_thread(cmd, args, port):
    """Target function running inside the background thread."""
    global _request_result, _request_error, _request_completed
    
    _request_result = None
    _request_error = None
    _request_completed = False
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(None) # Set no timeout since tracking/solving takes time
            s.connect(("localhost", port))
            
            payload = json.dumps({"cmd": cmd, **args}) + "\n"
            s.sendall(payload.encode('utf-8'))
            
            # Read until newline (delimited protocol)
            buffer = ""
            while True:
                chunk = s.recv(65536).decode('utf-8')
                if not chunk:
                    break
                buffer += chunk
                if "\n" in buffer:
                    line, _ = buffer.split("\n", 1)
                    resp = json.loads(line)
                    if resp.get("ok"):
                        _request_result = resp
                    else:
                        _request_error = resp.get("error", "Unknown server error")
                    break
            if not _request_result and not _request_error:
                _request_error = "Connection closed without response"
    except Exception as e:
        _request_error = f"Network IPC communication error: {str(e)}"
    finally:
        _request_completed = True

def send_worker_command_async(cmd, args, port=47832):
    """
    Launch socket request inside a background thread.
    Returns:
        True if thread started, False otherwise
    """
    global _active_request_thread, _request_completed
    
    if _active_request_thread and _active_request_thread.is_alive():
        print("Warning: Another background request is already active.")
        return False
        
    _request_completed = False
    _active_request_thread = threading.Thread(
        target=_run_request_thread,
        args=(cmd, args, port),
        daemon=True
    )
    _active_request_thread.start()
    return True

def poll_request_status():
    """
    Check if the background thread has finished.
    Returns:
        completed: bool
        result: dict (if successful)
        error: str (if failed)
    """
    global _request_completed, _request_result, _request_error
    if not _request_completed:
        return False, None, None
        
    res = _request_result
    err = _request_error
    
    # Reset state
    _request_completed = False
    
    return True, res, err

# ═══════════════════════════════════════════════════════════
# AUTO-DETECTION & DEPENDENCY INSTALLER UTILITIES
# ═══════════════════════════════════════════════════════════

_install_thread = None
_install_completed = False
_install_success = False
_install_message = ""

def detect_system_python():
    """Look for standard Python installations on the user's system."""
    import shutil
    
    # 1. First, check if there's python3 or python on the system PATH
    for cmd in ["python3", "python"]:
        path = shutil.which(cmd)
        if path:
            # Verify it's not Blender's internal python
            if os.path.realpath(path) != os.path.realpath(sys.executable):
                return os.path.realpath(path)
                
    # 2. Search common directory patterns
    home = os.path.expanduser("~")
    paths = []
    
    if os.name == 'nt': # Windows
        localappdata = os.environ.get("LOCALAPPDATA", "")
        programfiles = os.environ.get("ProgramFiles", "")
        
        # Check standard python installations
        if localappdata:
            py_prog = os.path.join(localappdata, "Programs", "Python")
            if os.path.exists(py_prog):
                for d in os.listdir(py_prog):
                    p = os.path.join(py_prog, d, "python.exe")
                    if os.path.exists(p):
                        paths.append(p)
                        
        # Check Program Files
        if programfiles:
            py_prog = os.path.join(programfiles, "Python")
            if os.path.exists(py_prog):
                for d in os.listdir(py_prog):
                    p = os.path.join(py_prog, d, "python.exe")
                    if os.path.exists(p):
                        paths.append(p)
                        
        # Check Conda locations
        conda_bases = [
            os.path.join(home, "miniconda3"),
            os.path.join(home, "anaconda3"),
            "C:\\miniconda3",
            "C:\\anaconda3"
        ]
        for base in conda_bases:
            p = os.path.join(base, "python.exe")
            if os.path.exists(p):
                paths.append(p)
                
    else: # macOS / Linux
        # Common UNIX paths
        common_paths = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3", # Apple Silicon brew
            # Conda
            os.path.join(home, "miniconda3", "bin", "python3"),
            os.path.join(home, "anaconda3", "bin", "python3"),
            os.path.join(home, "opt", "miniconda3", "bin", "python3"),
            os.path.join(home, "opt", "anaconda3", "bin", "python3"),
        ]
        for p in common_paths:
            if os.path.exists(p):
                paths.append(p)
                
    # Return the first found path
    if paths:
        return paths[0]
        
    return ""

def check_dependencies(python_path):
    """Check if the required AI dependencies are installed in the given Python environment."""
    if not python_path or not os.path.exists(python_path):
        return False
    try:
        # Run a quick check command to import the key libraries
        cmd = [python_path, "-c", "import torch, scipy, cv2, ultralytics; print('OK')"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False

def install_dependencies(python_path, status_callback=None):
    """Install required packages in the specified Python environment."""
    if not python_path or not os.path.exists(python_path):
        return False, "Invalid Python path"
        
    try:
        # Step 1: Upgrade pip
        if status_callback:
            status_callback("Upgrading pip...")
        cmd_pip = [python_path, "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.run(cmd_pip, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        
        # Step 2: Install core deep learning dependencies
        if status_callback:
            status_callback("Installing PyTorch, SciPy, OpenCV, & Ultralytics (~150MB)...")
            
        cmd_install = [python_path, "-m", "pip", "install", "torch", "scipy", "opencv-python", "ultralytics"]
        result = subprocess.run(cmd_install, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        
        if result.returncode == 0:
            return True, "All AI packages installed successfully!"
        else:
            return False, f"Installation failed: {result.stderr.strip()[-300:]}"
    except Exception as e:
        return False, f"Installation error: {str(e)}"

def run_install_async(python_path, status_callback=None):
    """Start the installation thread asynchronously."""
    global _install_thread, _install_completed, _install_success, _install_message
    if _install_thread and _install_thread.is_alive():
        return False
        
    _install_completed = False
    _install_success = False
    _install_message = "Starting package installation..."
    
    def target():
        global _install_completed, _install_success, _install_message
        
        def cb(msg):
            global _install_message
            _install_message = msg
            if status_callback:
                try:
                    status_callback(msg)
                except Exception:
                    pass
                    
        success, msg = install_dependencies(python_path, cb)
        _install_success = success
        _install_message = msg
        _install_completed = True
        
    _install_thread = threading.Thread(target=target, daemon=True)
    _install_thread.start()
    return True

def poll_install_status():
    """Poll the status of the active background install thread."""
    global _install_completed, _install_success, _install_message
    return _install_completed, _install_success, _install_message

