import os
import sys
import json
import socket
import subprocess
import threading
import time

def is_sandboxed_environment():
    """Detect if Blender is running in a sandbox (Store, Snap, Flatpak, App Sandbox)."""
    # 1. Windows Microsoft Store/UWP sandbox
    if os.name == 'nt':
        # Check if the process path or sys.executable is in WindowsApps
        if "windowsapps" in sys.executable.lower() or "pythonsoftwarefoundation" in sys.executable.lower():
            return True
            
    # 2. Linux Snap or Flatpak sandboxes
    if sys.platform.startswith('linux'):
        if os.environ.get("SNAP") or os.environ.get("SNAP_NAME"):
            return True
        if os.environ.get("FLATPAK_ID") or os.path.exists("/.flatpak-info"):
            return True
            
    # 3. macOS App Sandbox
    if sys.platform == 'darwin':
        if os.environ.get("APP_SANDBOX_CONTAINER_ID"):
            return True
            
    return False

def get_sandbox_error_message():
    """Get platform-appropriate error message for sandboxed environments."""
    if sys.platform.startswith('linux'):
        return (
            "Blender is running in a Snap or Flatpak sandbox, which blocks execution "
            "of external Python scripts. Please install the standard version from blender.org."
        )
    elif sys.platform == 'darwin':
        return (
            "The Python executable is sandboxed or lacks permissions to read Blender's addon files. "
            "Please install standard Python from python.org or Homebrew."
        )
    else:
        return (
            "Microsoft Store Python (or a sandboxed copy) is not supported due to UWP file redirection. "
            "Please install standard Python from python.org."
        )

def get_clean_env():
    """Create a copy of os.environ with Blender-specific Python variables removed."""
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env

def is_microsoft_store_python(python_path):
    """
    Check if the Python executable is the Microsoft Store version.
    The Microsoft Store version runs in a UWP sandbox which redirects AppData accesses,
    preventing it from loading server.py from Blender's addon folder in AppData.
    """
    if not python_path:
        return False
    path_lower = python_path.lower()
    
    # Heuristic: check if path contains store-specific folder names
    if "windowsapps" not in path_lower and "pythonsoftwarefoundation" not in path_lower:
        return False
        
    # Standard Python installations on Windows 10/11 put an execution alias
    # under "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe". This is NOT
    # a sandboxed Microsoft Store Python if it can access our addon files.
    # We verify if it is actually sandboxed by performing a functional file check.
    try:
        server_script = get_server_path()
        if os.path.exists(server_script):
            cmd = [python_path, "-c", f"import os; print(os.path.exists({repr(server_script)}))"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_clean_env(), text=True, timeout=3)
            if res.returncode == 0 and "True" in res.stdout:
                return False  # Not sandboxed (e.g. standard python.org alias)
    except Exception:
        pass
        
    return True

def is_functional_python(python_path):
    """
    Verify that the Python executable exists, can run a simple command,
    and is not the Microsoft Store 'not found' placeholder.
    """
    if not python_path or not os.path.exists(python_path):
        return False
    try:
        # Run python with a short print command and a timeout.
        # This will fail on the Microsoft Store execution alias if Python isn't actually installed.
        res = subprocess.run(
            [python_path, "-c", "import sys; print('OK')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=get_clean_env(),
            text=True,
            timeout=3
        )
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False

def verify_python_environment(python_path):
    """
    Verify the Python executable and its deep learning dependencies.
    Returns:
        - "INVALID_PATH" if path does not exist
        - "MS_STORE_PYTHON" if it is the sandboxed Microsoft Store python
        - "NOT_FUNCTIONAL" if it fails to execute a basic command
        - "READY" if all deep learning packages are installed
        - "MISSING" if python is valid but deep learning packages are not installed
    """
    if not python_path or not os.path.exists(python_path):
        return "INVALID_PATH"
        
    if is_microsoft_store_python(python_path):
        return "MS_STORE_PYTHON"
        
    if not is_functional_python(python_path):
        return "NOT_FUNCTIONAL"
        
    # Check if the python executable can actually see/access our server.py file.
    # UWP/Microsoft Store Python is sandboxed and cannot access AppData, so os.path.exists will return False inside it.
    server_script = get_server_path()
    try:
        cmd = [python_path, "-c", f"import os; print(os.path.exists({repr(server_script)}))"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_clean_env(), text=True, timeout=5)
        if res.returncode != 0 or "True" not in res.stdout:
            return "MS_STORE_PYTHON"
    except Exception:
        if is_sandboxed_environment():
            return "MS_STORE_PYTHON"
        return "NOT_FUNCTIONAL"
        
    # Check dependencies
    try:
        cmd = [python_path, "-c", "import torch, scipy, cv2, ultralytics; print('OK')"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=get_clean_env(), text=True, timeout=30)
        if res.returncode == 0 and "OK" in res.stdout:
            return "READY"
    except Exception:
        pass
        
    return "MISSING"

# Subprocess reference
_worker_process = None
# Active background request thread state
_active_request_thread = None
_request_result = None
_request_error = None
_request_completed = False
_request_progress = 0.0
_request_status_msg = ""

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
        ping = ping_worker(port)
        if ping.get("ok"):
            print(f"AutoSolve worker already running on port {port}.")
            return True, "Already running"
        else:
            stop_worker(port)
            if is_port_in_use(port):
                return False, f"Port {port} is in use by another application."

    if not python_path or not os.path.exists(python_path):
        return False, f"Python path does not exist: {python_path}"

    env_status = verify_python_environment(python_path)
    if env_status == "MS_STORE_PYTHON":
        return False, get_sandbox_error_message()

    server_script = get_server_path()
    if not os.path.exists(server_script):
        return False, f"Worker script not found at: {server_script}"

    # Setup log file to capture stderr/stdout and prevent process blocking
    import tempfile
    log_path = os.path.join(tempfile.gettempdir(), "autosolve_worker.log")
    try:
        log_file = open(log_path, "w", encoding="utf-8")
    except Exception:
        log_file = None

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
            stdout=log_file,
            stderr=subprocess.STDOUT if log_file else None,
            startupinfo=startupinfo,
            close_fds=True,
            env=get_clean_env()
        )
        if log_file:
            log_file.close()
        
        # Robustly poll and wait for the server to bind to the port (up to 15 seconds)
        timeout = 15.0
        start_time = time.time()
        started = False
        print(f"AutoSolve: Spawning worker process on port {port}...")
        while time.time() - start_time < timeout:
            if is_port_in_use(port):
                started = True
                break
            
            # Check if process exited early
            exit_code = _worker_process.poll()
            if exit_code is not None:
                err_detail = ""
                try:
                    if os.path.exists(log_path):
                        with open(log_path, "r", encoding="utf-8") as lf:
                            lines = lf.readlines()
                            if lines:
                                err_detail = "\nLast log output:\n" + "".join(lines[-10:])
                except Exception:
                    pass
                print(f"AutoSolve: Worker process exited immediately with code {exit_code}{err_detail}")
                return False, f"Worker process exited immediately with code {exit_code}.{err_detail}"
                
            time.sleep(0.25)
        
        if started:
            print(f"AutoSolve: Worker successfully bound to port {port}.")
            return True, "Started successfully"
        else:
            err_detail = ""
            try:
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()
                        if lines:
                            err_detail = "\nLast log output:\n" + "".join(lines[-10:])
            except Exception:
                pass
            print(f"AutoSolve: Worker failed to bind to port {port} after {timeout} seconds.{err_detail}")
            return False, f"Failed to bind to port {port} after {timeout} seconds.{err_detail}"
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
            payload = json.dumps({"cmd": "ping"}) + "\n"
            s.sendall(payload.encode('utf-8'))
            resp = s.recv(4096).decode('utf-8')
            if "\n" in resp:
                resp = resp.split("\n", 1)[0]
            return json.loads(resp)
    except Exception as e:
        return {"ok": False, "error": f"Failed to connect: {str(e)}"}

def _run_request_thread(cmd, args, port):
    """Target function running inside the background thread."""
    global _request_result, _request_error, _request_completed, _request_progress, _request_status_msg
    
    _request_result = None
    _request_error = None
    _request_completed = False
    
    try:
        print(f"AutoSolve: Sending command '{cmd}' to worker on port {port}...")
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
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    resp = json.loads(line)
                    if resp.get("type") == "progress":
                        _request_progress = resp.get("progress", 0.0)
                        _request_status_msg = resp.get("message", "")
                    else:
                        if resp.get("ok"):
                            _request_result = resp
                            print(f"AutoSolve: Command '{cmd}' completed successfully.")
                        else:
                            _request_error = resp.get("error", "Unknown server error")
                            print(f"AutoSolve: Command '{cmd}' failed: {_request_error}")
                        break
                # If we broke from the inner loop because of a final result/error,
                # we need to break the outer loop too.
                if _request_result is not None or _request_error is not None:
                    break
            if not _request_result and not _request_error:
                _request_error = "Connection closed without response"
                print(f"AutoSolve: Command '{cmd}' connection closed without response.")
    except Exception as e:
        _request_error = f"Network IPC communication error: {str(e)}"
        print(f"AutoSolve: Network IPC error for command '{cmd}': {str(e)}")
    finally:
        _request_completed = True

def send_worker_command_async(cmd, args, port=47832):
    """
    Launch socket request inside a background thread.
    Returns:
        True if thread started, False otherwise
    """
    global _active_request_thread, _request_completed, _request_progress, _request_status_msg
    
    if _active_request_thread and _active_request_thread.is_alive():
        print("Warning: Another background request is already active.")
        return False
        
    _request_completed = False
    _request_progress = 0.0
    _request_status_msg = ""
    _active_request_thread = threading.Thread(
        target=_run_request_thread,
        args=(cmd, args, port),
        daemon=True
    )
    _active_request_thread.start()
    return True

def get_request_progress():
    """
    Get the progress and status message of the active background request.
    Returns:
        progress: float (0.0 to 1.0)
        status_msg: str
    """
    global _request_progress, _request_status_msg
    return _request_progress, _request_status_msg

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

def _get_python_from_launcher():
    if os.name != 'nt':
        return None
    for cmd in ["py", "py.exe"]:
        try:
            res = subprocess.run(
                [cmd, "-3", "-c", "import sys; print(sys.executable)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=get_clean_env(),
                text=True,
                timeout=5
            )
            if res.returncode == 0:
                path = res.stdout.strip()
                if path and os.path.exists(path):
                    return path
        except Exception:
            continue
    return None

def _get_python_from_registry():
    if os.name != 'nt':
        return None
    try:
        import winreg
    except ImportError:
        return None
        
    found_paths = []
    for hkey in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
        for access_mask in [0, winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY]:
            try:
                key_path = r"SOFTWARE\Python\PythonCore"
                sam = winreg.KEY_READ | access_mask if access_mask else winreg.KEY_READ
                with winreg.OpenKey(hkey, key_path, 0, sam) as core_key:
                    num_subkeys = winreg.QueryInfoKey(core_key)[0]
                    for idx in range(num_subkeys):
                        try:
                            ver_name = winreg.EnumKey(core_key, idx)
                            if not ver_name.startswith("3."):
                                continue
                            install_path_key_path = rf"SOFTWARE\Python\PythonCore\{ver_name}\InstallPath"
                            with winreg.OpenKey(hkey, install_path_key_path, 0, sam) as ip_key:
                                try:
                                    path, val_type = winreg.QueryValueEx(ip_key, "ExecutablePath")
                                    if path and os.path.exists(path):
                                        found_paths.append((ver_name, path))
                                        continue
                                except Exception:
                                    pass
                                    
                                try:
                                    dir_path, val_type = winreg.QueryValueEx(ip_key, "")
                                    if dir_path:
                                        path = os.path.join(dir_path, "python.exe")
                                        if os.path.exists(path):
                                            found_paths.append((ver_name, path))
                                except Exception:
                                    pass
                        except Exception:
                            continue
            except Exception:
                continue
                
    if found_paths:
        def version_key(item):
            ver_str = item[0]
            try:
                return [int(x) for x in ver_str.split(".")]
            except Exception:
                return [0]
        found_paths.sort(key=version_key, reverse=True)
        return found_paths[0][1]
        
    return None

def detect_system_python():
    """Look for standard Python installations on the user's system."""
    import shutil
    
    # Helper to check if candidate is a good non-sandboxed Python
    def is_valid_candidate(p):
        return is_functional_python(p) and not is_microsoft_store_python(p)

    # 0. Try Windows launcher & registry first
    if os.name == 'nt':
        launcher_py = _get_python_from_launcher()
        if launcher_py and is_valid_candidate(launcher_py):
            return launcher_py
            
        registry_py = _get_python_from_registry()
        if registry_py and is_valid_candidate(registry_py):
            return registry_py
    
    # 1. First, check if there's python3 or python on the system PATH
    for cmd in ["python3", "python"]:
        path = shutil.which(cmd)
        if path:
            # Verify it's not Blender's internal python
            real_path = os.path.realpath(path)
            if real_path != os.path.realpath(sys.executable):
                if is_valid_candidate(real_path):
                    return real_path
                
    # 2. Search common directory patterns
    home = os.path.expanduser("~")
    paths = []
    
    if os.name == 'nt': # Windows
        localappdata = os.environ.get("LOCALAPPDATA", "")
        # Fallback/override to bypass UWP environment redirection in Microsoft Store Blender
        if not localappdata or "LocalCache" in localappdata:
            localappdata = os.path.join(os.path.expanduser("~"), "AppData", "Local")
        programfiles = os.environ.get("ProgramFiles", "")
        
        # Check standard python installations
        if localappdata:
            py_prog = os.path.join(localappdata, "Programs", "Python")
            if os.path.exists(py_prog):
                for d in os.listdir(py_prog):
                    p = os.path.join(py_prog, d, "python.exe")
                    if os.path.exists(p) and is_valid_candidate(p):
                        paths.append(p)
                        
        # Check Program Files
        if programfiles:
            py_prog = os.path.join(programfiles, "Python")
            if os.path.exists(py_prog):
                for d in os.listdir(py_prog):
                    p = os.path.join(py_prog, d, "python.exe")
                    if os.path.exists(p) and is_valid_candidate(p):
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
            if os.path.exists(p) and is_valid_candidate(p):
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
            if os.path.exists(p) and is_valid_candidate(p):
                paths.append(p)
                
    # Return the first found path
    if paths:
        return paths[0]
        
    return ""

def check_dependencies(python_path):
    """Check if the required AI dependencies are installed in the given Python environment."""
    return verify_python_environment(python_path) == "READY"

def install_dependencies(python_path, status_callback=None):
    """Install required packages in the specified Python environment."""
    if not python_path or not os.path.exists(python_path):
        return False, "Invalid Python path"
        
    env_status = verify_python_environment(python_path)
    if env_status == "MS_STORE_PYTHON":
        return False, get_sandbox_error_message()
        
    try:
        # Step 1: Upgrade pip
        if status_callback:
            status_callback("Upgrading pip...")
        print("AutoSolve: Upgrading pip...")
        cmd_pip = [python_path, "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.run(cmd_pip, env=get_clean_env(), timeout=60)
        
        # Step 2: Install core deep learning dependencies
        if status_callback:
            status_callback("Installing PyTorch, SciPy, OpenCV, & Ultralytics (~150MB)...")
        print("AutoSolve: Installing PyTorch, SciPy, OpenCV, & Ultralytics (~150MB)...")
            
        cmd_install = [python_path, "-m", "pip", "install", "torch", "scipy", "opencv-python", "ultralytics"]
        process = subprocess.Popen(
            cmd_install,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=get_clean_env(),
            text=True
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            output_lines.append(line)
            
        process.stdout.close()
        return_code = process.wait(timeout=600)
        
        if return_code == 0:
            print("AutoSolve: All AI packages installed successfully!")
            return True, "All AI packages installed successfully!"
        else:
            err_msg = "".join(output_lines[-10:])
            print(f"AutoSolve: Installation failed:\n{err_msg}")
            return False, f"Installation failed: {err_msg.strip()}"
    except Exception as e:
        print(f"AutoSolve: Installation error: {str(e)}")
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

