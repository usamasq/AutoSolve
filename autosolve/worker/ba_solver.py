import numpy as np
import scipy.optimize as opt
from scipy.spatial.transform import Rotation as R

def c2w_to_rt(c2w):
    """Convert 4x4 camera-to-world matrix to 6-DOF rotvec/translation (world-to-camera)."""
    if c2w is None:
        return np.zeros(6, dtype=np.float32)
    c2w = np.array(c2w)
    R_c2w = c2w[:3, :3]
    T_c2w = c2w[:3, 3]
    # Invert
    R_w2c = R_c2w.T
    T_w2c = -np.dot(R_w2c, T_c2w)
    rvec = R.from_matrix(R_w2c).as_rotvec()
    return np.hstack((rvec, T_w2c))

def rt_to_c2w(rvec, tvec):
    """Convert 6-DOF rotvec/translation (world-to-camera) to 4x4 camera-to-world matrix."""
    R_w2c = R.from_rotvec(rvec).as_matrix()
    T_w2c = tvec
    # Invert
    R_c2w = R_w2c.T
    T_c2w = -np.dot(R_c2w, T_w2c)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = T_c2w
    return c2w.tolist()

def residuals(params, num_cameras, num_points, camera_indices, point_indices, 
              points_2d, optimize_intrinsics=False):
    """
    Compute residuals (reprojection errors) for least_squares.
    """
    # Extract camera parameters (rvecs, tvecs)
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    rvecs = camera_params[:, :3]
    tvecs = camera_params[:, 3:]
    
    # Extract 3D points
    pts_start = num_cameras * 6
    pts_end = pts_start + num_points * 3
    points_3d = params[pts_start:pts_end].reshape((num_points, 3))
    
    # Extract intrinsics if optimized
    if optimize_intrinsics:
        f = params[pts_end]
        k1 = params[pts_end + 1]
        k2 = params[pts_end + 2]
    else:
        f = 1.0
        k1 = 0.0
        k2 = 0.0
        
    cx, cy = 0.5, 0.5
    
    p_3d = points_3d[point_indices]
    rv = rvecs[camera_indices]
    tv = tvecs[camera_indices]
    
    # Rotate points: Pc = R * Pw + T
    rot_mats = R.from_rotvec(rv).as_matrix()
    p_cam = np.einsum('nij,nj->ni', rot_mats, p_3d) + tv
    
    z = p_cam[:, 2]
    z[np.abs(z) < 1e-5] = 1e-5
    
    xn = p_cam[:, 0] / z
    yn = p_cam[:, 1] / z
    
    r2 = xn**2 + yn**2
    distortion = 1.0 + k1 * r2 + k2 * (r2**2)
    
    xn_dist = xn * distortion
    yn_dist = yn * distortion
    
    # Project to normalized screen coords
    proj_x = f * xn_dist + cx
    proj_y = f * yn_dist + cy
    
    proj_2d = np.column_stack((proj_x, proj_y))
    
    return (proj_2d - points_2d).ravel()

def solve_precision_bundle(obs_data, init_cameras_c2w, init_points, init_f=1.0, init_k1=0.0, init_k2=0.0):
    """
    Run Levenberg-Marquardt bundle adjustment using scipy least_squares.
    """
    num_cameras = len(init_cameras_c2w)
    num_points = len(init_points)
    
    camera_indices = np.array([obs["frame"] for obs in obs_data])
    point_indices = np.array([obs["point_idx"] for obs in obs_data])
    points_2d = np.array([obs["uv"] for obs in obs_data])
    
    # Convert init_cameras to R/T vector
    init_cameras_rt = np.array([c2w_to_rt(c) for c in init_cameras_c2w], dtype=np.float32)
    
    # Flatten initial parameters
    x0 = np.hstack((init_cameras_rt.ravel(), init_points.ravel(), [init_f, init_k1, init_k2]))
    
    # Phase 1: Optimize camera poses and 3D points with fixed intrinsics
    x0_no_intrinsics = x0[:-3]
    
    res = opt.least_squares(
        residuals, x0_no_intrinsics, jac="3-point",
        args=(num_cameras, num_points, camera_indices, point_indices, points_2d, False),
        loss="cauchy", f_scale=0.05, method="trf"
    )
    
    # Update
    x0[:-3] = res.x
    
    # Phase 2: Optimize camera poses, 3D points AND intrinsics
    res_global = opt.least_squares(
        residuals, x0, jac="3-point",
        args=(num_cameras, num_points, camera_indices, point_indices, points_2d, True),
        loss="cauchy", f_scale=0.03, method="trf"
    )
    
    optimized_params = res_global.x
    
    # Reshape optimized cameras R/T
    optimized_cameras_rt = optimized_params[:num_cameras * 6].reshape((num_cameras, 6))
    pts_start = num_cameras * 6
    pts_end = pts_start + num_points * 3
    optimized_points = optimized_params[pts_start:pts_end].reshape((num_points, 3))
    
    f_opt = float(optimized_params[pts_end])
    k1_opt = float(optimized_params[pts_end + 1])
    k2_opt = float(optimized_params[pts_end + 2])
    
    # Convert R/T back to 4x4 camera-to-world matrices
    optimized_cameras_c2w = [rt_to_c2w(cam[:3], cam[3:]) for cam in optimized_cameras_rt]
    
    return optimized_cameras_c2w, optimized_points, f_opt, k1_opt, k2_opt
