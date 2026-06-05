# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ReconstructionReader — Post-solve camera pose analysis.

After every successful camera solve, reads back the per-frame camera
positions and orientations from Blender's reconstruction data and uses
them to:

  1. Detect sudden camera-pose jumps (frames where the reconstruction
     went wrong) and flag those frames for exclusion.
  2. Compute a camera velocity signal to feed back into the settings
     predictor for the next solve attempt.
  3. Export (frame, camera_pos, reprojection_error) tuples for the
     ML training pipeline.

All computation is pure numpy — no external dependencies required.
"""

import bpy
import numpy as np
from mathutils import Vector, Quaternion
from typing import List, Tuple, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════

# A camera pose record: (frame_index, position_3d, rotation_quaternion)
CameraPose = Tuple[int, np.ndarray, np.ndarray]   # (frame, pos[3], quat[4])


# ═══════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION READER
# ═══════════════════════════════════════════════════════════════════════════

class ReconstructionReader:
    """
    Reads and analyses camera reconstruction data from a solved MovieClip.

    Usage:
        reader = ReconstructionReader()
        poses  = reader.read_camera_poses(clip)
        if poses:
            bad_frames = reader.detect_pose_jumps(poses)
            vel        = reader.camera_velocity_signal(poses)
    """

    # Threshold: angular jump (degrees) per frame that signals a bad frame
    DEFAULT_ANGULAR_JUMP_DEG  = 15.0
    # Threshold: positional jump (normalised scene units) per frame
    DEFAULT_POSITION_JUMP     = 0.5

    # ────────────────────────────────────────────────────────────────────────
    # Core Pose Reading
    # ────────────────────────────────────────────────────────────────────────

    def read_camera_poses(
        self,
        clip: bpy.types.MovieClip
    ) -> List[CameraPose]:
        """
        Read all reconstructed camera poses from a clip's tracking data.

        Returns a list of (frame, position, quaternion) sorted by frame,
        or an empty list if the reconstruction is not valid.

        Position is a numpy array [x, y, z].
        Quaternion is a numpy array [w, x, y, z].
        """
        recon = clip.tracking.reconstruction
        if not recon.is_valid:
            return []

        poses: List[CameraPose] = []

        frame_start = clip.frame_start
        frame_end   = clip.frame_start + clip.frame_duration - 1

        for frame in range(frame_start, frame_end + 1):
            try:
                # camera_to_world matrix for this frame
                mat = recon.cameras.matrix_from_frame(frame=frame)
                if mat is None:
                    continue

                # Extract position (last column of 4x4 matrix)
                pos = np.array([mat[0][3], mat[1][3], mat[2][3]], dtype=np.float64)

                # Extract rotation as quaternion from the 3x3 upper-left block
                rot_mat = [
                    [mat[i][j] for j in range(3)]
                    for i in range(3)
                ]
                quat = self._rotation_matrix_to_quaternion(rot_mat)

                poses.append((frame, pos, quat))

            except Exception:
                # Some frames may not have reconstruction data
                continue

        return poses

    def _rotation_matrix_to_quaternion(self, R: list) -> np.ndarray:
        """Convert a 3x3 rotation matrix (list of lists) to a [w,x,y,z] quaternion."""
        r00, r01, r02 = R[0]
        r10, r11, r12 = R[1]
        r20, r21, r22 = R[2]

        trace = r00 + r11 + r22

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (r21 - r12) * s
            y = (r02 - r20) * s
            z = (r10 - r01) * s
        elif r00 > r11 and r00 > r22:
            s = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)
            w = (r21 - r12) / s
            x = 0.25 * s
            y = (r01 + r10) / s
            z = (r02 + r20) / s
        elif r11 > r22:
            s = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)
            w = (r02 - r20) / s
            x = (r01 + r10) / s
            y = 0.25 * s
            z = (r12 + r21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)
            w = (r10 - r01) / s
            x = (r02 + r20) / s
            y = (r12 + r21) / s
            z = 0.25 * s

        return np.array([w, x, y, z], dtype=np.float64)

    # ────────────────────────────────────────────────────────────────────────
    # Jump Detection
    # ────────────────────────────────────────────────────────────────────────

    def detect_pose_jumps(
        self,
        poses: List[CameraPose],
        angular_threshold_deg: float = DEFAULT_ANGULAR_JUMP_DEG,
        position_threshold:    float = DEFAULT_POSITION_JUMP
    ) -> List[int]:
        """
        Detect frames where the camera pose makes an implausibly large jump
        compared to neighboring frames — a sign of a bad reconstruction.

        Returns a list of frame numbers that are flagged as unreliable.
        """
        if len(poses) < 3:
            return []

        bad_frames: List[int] = []

        # Compute per-frame deltas
        pos_deltas  = []
        angle_deltas = []

        for i in range(1, len(poses)):
            f_prev, pos_prev, quat_prev = poses[i - 1]
            f_curr, pos_curr, quat_curr = poses[i]

            frame_gap = max(1, f_curr - f_prev)

            # Positional delta (per frame)
            pos_delta = float(np.linalg.norm(pos_curr - pos_prev)) / frame_gap
            pos_deltas.append(pos_delta)

            # Angular delta — angle between quaternions (degrees per frame)
            dot = float(np.clip(np.dot(quat_prev, quat_curr), -1.0, 1.0))
            # Ensure shortest path
            if dot < 0:
                quat_curr_adj = -quat_curr
                dot = -dot
            else:
                quat_curr_adj = quat_curr
            angle_rad = 2.0 * np.arccos(abs(dot))
            angle_deg = float(np.degrees(angle_rad)) / frame_gap
            angle_deltas.append(angle_deg)

        pos_arr   = np.array(pos_deltas,   dtype=np.float64)
        angle_arr = np.array(angle_deltas, dtype=np.float64)

        # Adaptive thresholds — flag frames that are outliers (> mean + 3*std)
        pos_mean,   pos_std   = float(np.mean(pos_arr)),   float(np.std(pos_arr))
        angle_mean, angle_std = float(np.mean(angle_arr)), float(np.std(angle_arr))

        pos_adaptive   = max(position_threshold,    pos_mean   + 3.0 * pos_std)
        angle_adaptive = max(angular_threshold_deg, angle_mean + 3.0 * angle_std)

        for i in range(len(pos_deltas)):
            if pos_deltas[i] > pos_adaptive or angle_deltas[i] > angle_adaptive:
                # Flag the frame *after* the jump (likely the bad one)
                bad_frames.append(poses[i + 1][0])

        return list(sorted(set(bad_frames)))

    # ────────────────────────────────────────────────────────────────────────
    # Velocity Signal
    # ────────────────────────────────────────────────────────────────────────

    def camera_velocity_signal(self, poses: List[CameraPose]) -> Dict[str, float]:
        """
        Compute average and peak camera velocity from reconstruction poses.

        Returns dict with:
          'mean_linear_vel'   — average per-frame positional displacement
          'peak_linear_vel'   — maximum per-frame positional displacement
          'mean_angular_vel'  — average per-frame angular velocity (degrees)
          'peak_angular_vel'  — maximum per-frame angular velocity (degrees)
          'motion_class'      — 'HIGH', 'MEDIUM', or 'LOW'
        """
        if len(poses) < 2:
            return {
                'mean_linear_vel': 0.0, 'peak_linear_vel': 0.0,
                'mean_angular_vel': 0.0, 'peak_angular_vel': 0.0,
                'motion_class': 'LOW'
            }

        lin_vels   = []
        angle_vels = []

        for i in range(1, len(poses)):
            f_prev, pos_prev, quat_prev = poses[i - 1]
            f_curr, pos_curr, quat_curr = poses[i]
            frame_gap = max(1, f_curr - f_prev)

            lin_vels.append(float(np.linalg.norm(pos_curr - pos_prev)) / frame_gap)

            dot = float(np.clip(np.dot(quat_prev, quat_curr), -1.0, 1.0))
            angle_rad = 2.0 * np.arccos(abs(dot))
            angle_vels.append(float(np.degrees(angle_rad)) / frame_gap)

        mean_lin  = float(np.mean(lin_vels))
        peak_lin  = float(np.max(lin_vels))
        mean_ang  = float(np.mean(angle_vels))
        peak_ang  = float(np.max(angle_vels))

        # Classify based on angular velocity (more reliable than position units)
        if mean_ang > 5.0:
            cls = 'HIGH'
        elif mean_ang > 1.5:
            cls = 'MEDIUM'
        else:
            cls = 'LOW'

        return {
            'mean_linear_vel':  mean_lin,
            'peak_linear_vel':  peak_lin,
            'mean_angular_vel': mean_ang,
            'peak_angular_vel': peak_ang,
            'motion_class':     cls,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Convenience: read reconstruction error from clip
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_reprojection_error(clip: bpy.types.MovieClip) -> float:
        """Return the average reprojection error from the clip's reconstruction."""
        recon = clip.tracking.reconstruction
        if not recon.is_valid:
            return 999.0
        try:
            return float(recon.average_error)
        except Exception:
            return 999.0
