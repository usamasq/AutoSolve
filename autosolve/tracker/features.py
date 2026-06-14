# SPDX-FileCopyrightText: 2025 Usama Bin Shahid
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Trajectory feature extraction for AutoSolve Track Quality Predictor.
Fully packaged inside the addon for runtime access.
"""

import numpy as np
from typing import List, Tuple


def extract_features_from_history(
    coords: List[Tuple[float, float]],
    neighbors_coords: List[List[Tuple[float, float]]],
    region_idx: int,
    footage_idx: int,
    robust_mode: bool,
    history_len: int = 6,
    dist_to_dynamic_mask: float = 1.0,
    on_dynamic_mask: float = 0.0,
    flow_consensus_deviation: float = 0.0
) -> np.ndarray:
    """
    Extract a 20-dimensional feature vector for a track based on its history and semantic boundaries.
    """
    features = np.zeros(20, dtype=np.float32)
    
    track_len = len(coords)
    if track_len == 0:
        return features
        
    # Use at most history_len frames
    inspect_coords = coords[-history_len:]
    if len(inspect_coords) < 2:
        inspect_coords = [inspect_coords[0]] * (2 - len(inspect_coords)) + inspect_coords
        
    # 1. Velocities (deltas)
    vel_x = [inspect_coords[i][0] - inspect_coords[i-1][0] for i in range(1, len(inspect_coords))]
    vel_y = [inspect_coords[i][1] - inspect_coords[i-1][1] for i in range(1, len(inspect_coords))]
    
    features[0] = float(np.mean(vel_x))  # velocity_x_mean
    features[1] = float(np.mean(vel_y))  # velocity_y_mean
    features[2] = float(np.std(vel_x)) if len(vel_x) > 1 else 0.0  # velocity_x_std
    features[3] = float(np.std(vel_y)) if len(vel_y) > 1 else 0.0  # velocity_y_std
    
    # 2. Accelerations (deltas of velocities)
    acc_x = [vel_x[i] - vel_x[i-1] for i in range(1, len(vel_x))] if len(vel_x) > 1 else [0.0]
    acc_y = [vel_y[i] - vel_y[i-1] for i in range(1, len(vel_y))] if len(vel_y) > 1 else [0.0]
    
    features[4] = float(np.mean(acc_x))  # accel_x_mean
    features[5] = float(np.mean(acc_y))  # accel_y_mean
    
    # 3. Direction changes
    dir_changes_x = sum(1 for i in range(1, len(vel_x)) if (vel_x[i] > 0) != (vel_x[i-1] > 0) and abs(vel_x[i]) > 1e-5)
    dir_changes_y = sum(1 for i in range(1, len(vel_y)) if (vel_y[i] > 0) != (vel_y[i-1] > 0) and abs(vel_y[i]) > 1e-5)
    features[6] = float(dir_changes_x)  # dir_change_x
    features[7] = float(dir_changes_y)  # dir_change_y
    
    # 4. Lifespan
    features[8] = float(track_len)  # frames_alive
    
    # 5. Distance to nearest neighbor
    current_pos = coords[-1]
    min_dist = 999.0
    neighbor_vels_x = []
    neighbor_vels_y = []
    
    for n_coords in neighbors_coords:
        if len(n_coords) >= 1:
            n_pos = n_coords[-1]
            dist = ((current_pos[0] - n_pos[0])**2 + (current_pos[1] - n_pos[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                
            # Collect velocity if neighbor has history
            if len(n_coords) >= 2:
                neighbor_vels_x.append(n_coords[-1][0] - n_coords[-2][0])
                neighbor_vels_y.append(n_coords[-1][1] - n_coords[-2][1])
                
    features[9] = float(min_dist) if min_dist < 998.0 else 1.0  # dist_to_nearest
    
    # 6. Neighbor agreement
    my_last_vel_x = vel_x[-1] if vel_x else 0.0
    my_last_vel_y = vel_y[-1] if vel_y else 0.0
    
    if neighbor_vels_x:
        avg_n_vel_x = float(np.mean(neighbor_vels_x))
        avg_n_vel_y = float(np.mean(neighbor_vels_y))
        features[10] = my_last_vel_x - avg_n_vel_x  # neighbor_agreement_x
        features[11] = my_last_vel_y - avg_n_vel_y  # neighbor_agreement_y
    else:
        features[10] = 0.0
        features[11] = 0.0
        
    # 7. Context variables
    features[12] = float(region_idx)  # region_index
    features[13] = float(footage_idx)  # footage_type_index
    features[14] = 1.0 if robust_mode else 0.0  # robust_mode
    
    # 8. Semantic Mask & Consensual Flow features
    features[15] = float(dist_to_dynamic_mask)  # distance to closest dynamic object mask
    features[16] = float(on_dynamic_mask)       # 1.0 if directly inside dynamic object, else 0.0
    features[17] = float(flow_consensus_deviation) # RANSAC flow consensus difference
    features[18] = float(np.std(vel_x + vel_y)) if vel_x else 0.0  # Jitter standard dev
    features[19] = float(len(neighbors_coords))  # Local neighbor track density
    
    return features
