# SPDX-FileCopyrightText: 2024-2025 Your Name
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Background solver manager.

Handles the thread-safe execution of the solve pipeline:
- Main thread: Frame extraction (bpy.data access)
- Background thread: pycolmap solve (no bpy access)
- Modal timer: Polls result queue, updates UI on main thread

⚠️ CRITICAL: Never access bpy.* from the background thread!
"""

import bpy
import threading
import queue
import tempfile
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


class SolveStatus(Enum):
    """Current status of the solve process."""
    IDLE = auto()
    EXTRACTING = auto()  # Main thread - extracting frames
    SOLVING = auto()     # Background thread - running pycolmap
    COMPLETE = auto()    # Done - results available
    FAILED = auto()      # Error occurred
    CANCELLED = auto()   # User cancelled


@dataclass
class SolveProgress:
    """Progress update from solver."""
    status: SolveStatus
    message: str = ""
    progress: float = 0.0  # 0.0 to 1.0


class BackgroundSolver:
    """
    Manages background solve execution with thread safety.
    
    Usage:
        solver = BackgroundSolver()
        solver.start(clip, settings, on_complete_callback)
        # ... poll solver.get_progress() from modal timer ...
        result = solver.get_result()  # When complete
    """
    
    _instance: Optional['BackgroundSolver'] = None
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._result_queue: queue.Queue = queue.Queue()
        self._progress_queue: queue.Queue = queue.Queue()
        self._cancel_flag = threading.Event()
        self._status = SolveStatus.IDLE
        self._result = None
        self._error: Optional[str] = None
        self._temp_dir: Optional[Path] = None
    
    @classmethod
    def get_instance(cls) -> 'BackgroundSolver':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = BackgroundSolver()
        return cls._instance
    
    def is_running(self) -> bool:
        """Check if a solve is currently in progress."""
        return self._status in (SolveStatus.EXTRACTING, SolveStatus.SOLVING)
    
    def start(
        self,
        clip: bpy.types.MovieClip,
        tripod_mode: bool = False,
        use_distortion: bool = True,
        frame_step: int = 1,
        quality_preset: str = 'BALANCED',
    ) -> bool:
        """
        Start the solve process.
        
        Phase 1 (main thread): Extract frames to temp directory
        Phase 2 (background): Run pycolmap solver
        
        Args:
            clip: Movie clip to solve.
            tripod_mode: Use rotation-only model.
            use_distortion: Estimate lens distortion.
            frame_step: Process every Nth frame.
            quality_preset: 'FAST', 'BALANCED', or 'QUALITY'.
        
        Returns:
            True if solve started, False if already running.
        """
        if self.is_running():
            return False
        
        # Reset state
        self._cancel_flag.clear()
        self._result = None
        self._error = None
        self._status = SolveStatus.EXTRACTING
        
        # Create persistent cache directory based on clip name
        # This allows reusing extracted frames between runs
        import re
        safe_name = re.sub(r'[^\w\-_]', '_', clip.name)
        cache_root = Path(tempfile.gettempdir()) / "autosolve_cache" / safe_name
        cache_root.mkdir(parents=True, exist_ok=True)
        
        self._temp_dir = cache_root
        image_dir = self._temp_dir / "frames"
        output_dir = self._temp_dir / "reconstruction"
        
        # Ensure reconstruction dir is clean (don't reuse stale database)
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure frames dir exists
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Store clip path (can't pass clip to thread)
        clip_filepath = bpy.path.abspath(clip.filepath)
        clip_name = clip.name
        clip_frame_start = clip.frame_start
        clip_frame_duration = clip.frame_duration
        clip_size = tuple(clip.size)
        
        # Phase 1: Extract frames (MAIN THREAD)
        self._update_progress(SolveStatus.EXTRACTING, "Checking for existing frames...", 0.0)
        
        try:
            from .frame_extractor import extract_frames
            
            # Check if frames already exist (caching)
            existing_frames = list(image_dir.glob("*.jpg"))
            
            if len(existing_frames) >= 10:
                # Reuse existing frames
                frame_count = len(existing_frames)
                self._update_progress(
                    SolveStatus.EXTRACTING,
                    f"Reusing {frame_count} cached frames",
                    0.3
                )
            else:
                # Extract new frames
                def progress_cb(current, total):
                    pct = current / total if total > 0 else 0
                    percent_int = int(pct * 100)
                    self._update_progress(
                        SolveStatus.EXTRACTING,
                        f"Extracting frame {current} of {total} ({percent_int}%)",
                        pct * 0.3  # Extraction is 30% of total
                    )
                
                frame_count = extract_frames(
                    clip,
                    image_dir,
                    step=frame_step,
                    quality=95,
                    progress_callback=progress_cb,
                )
            
            if frame_count < 10:
                raise ValueError(f"Only {frame_count} frames extracted. Need at least 10.")
                
        except Exception as e:
            self._status = SolveStatus.FAILED
            self._error = str(e)
            self._update_progress(SolveStatus.FAILED, f"Extraction failed: {e}", 0.0)
            return False
        
        # Phase 2: Start background thread for pycolmap
        self._status = SolveStatus.SOLVING
        self._update_progress(SolveStatus.SOLVING, "Starting solver...", 0.3)
        
        self._thread = threading.Thread(
            target=self._run_solver_thread,
            args=(image_dir, output_dir, tripod_mode, use_distortion, quality_preset),
            daemon=True,
        )
        self._thread.start()
        
        return True
    
    def _run_solver_thread(
        self,
        image_dir: Path,
        output_dir: Path,
        tripod_mode: bool,
        use_distortion: bool,
        quality_preset: str,
    ):
        """
        Background thread: Run pycolmap solver.
        
        ⚠️ DO NOT ACCESS bpy.* IN THIS FUNCTION!
        """
        try:
            from .pipeline import run_reconstruction
            
            def progress_cb(message: str, progress: float):
                if self._cancel_flag.is_set():
                    raise InterruptedError("Solve cancelled by user")
                # Map solver progress (0-1) to overall progress (0.3-1.0)
                overall = 0.3 + progress * 0.7
                self._update_progress(SolveStatus.SOLVING, message, overall)
            
            result = run_reconstruction(
                image_dir=image_dir,
                output_dir=output_dir,
                tripod_mode=tripod_mode,
                use_distortion=use_distortion,
                quality_preset=quality_preset,
                progress_callback=progress_cb,
            )
            
            # Success!
            self._result = result
            self._status = SolveStatus.COMPLETE
            self._update_progress(SolveStatus.COMPLETE, "Solve complete!", 1.0)
            self._result_queue.put(('SUCCESS', result))
            
        except InterruptedError:
            self._status = SolveStatus.CANCELLED
            self._update_progress(SolveStatus.CANCELLED, "Cancelled", 0.0)
            self._result_queue.put(('CANCELLED', None))
            
        except Exception as e:
            self._status = SolveStatus.FAILED
            self._error = str(e)
            self._update_progress(SolveStatus.FAILED, f"Solve failed: {e}", 0.0)
            self._result_queue.put(('ERROR', str(e)))
    
    def _update_progress(self, status: SolveStatus, message: str, progress: float):
        """Push progress update to queue (thread-safe)."""
        try:
            # Clear old progress, keep only latest
            while not self._progress_queue.empty():
                self._progress_queue.get_nowait()
        except queue.Empty:
            pass
        
        self._progress_queue.put(SolveProgress(status, message, progress))
    
    def get_progress(self) -> Optional[SolveProgress]:
        """Get latest progress update (call from main thread)."""
        try:
            return self._progress_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_result(self):
        """Get solve result if complete (call from main thread)."""
        if self._status == SolveStatus.COMPLETE:
            return self._result
        return None
    
    def get_error(self) -> Optional[str]:
        """Get error message if failed."""
        return self._error
    
    def cancel(self):
        """Request cancellation of running solve."""
        self._cancel_flag.set()
    
    def cleanup(self):
        """Clean up internal state. Cache files are PRESERVED for reuse."""
        # Note: We do NOT delete self._temp_dir because it is now a persistent cache.
        # It will be cleared by the OS or manually if needed.
        self._temp_dir = None
        
        self._status = SolveStatus.IDLE
        self._result = None
        self._error = None
