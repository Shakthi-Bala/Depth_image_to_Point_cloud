"""
Reprojection check for a cam1→cam2 transformation matrix.

Workflow:
  1. Unproject cam1 depth image → 3D points in cam1 frame
  2. Apply T_cam1_to_cam2 to get points in cam2 frame
  3. Project onto cam2 image plane
  4. Overlay on cam2 RGB image and compute reprojection RMSE
     (RMSE is only meaningful if you also supply 2D correspondences)
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURE THESE
# ---------------------------------------------------------------------------

# Paths
CAM1_DEPTH_PATH = "/path/to/cam1_depth.png"  # uint16 depth, mm
CAM1_RGB_PATH   = "/path/to/cam1_rgb.png"    # for visual reference
CAM2_RGB_PATH   = "/path/to/cam2_rgb.png"    # image to project onto

# Cam1 intrinsics
K1 = np.array([
    [390.61,  0.0,   330.444],
    [  0.0, 390.61,  238.868],
    [  0.0,   0.0,     1.0  ]
], dtype=np.float64)

# Cam2 intrinsics
K2 = np.array([
    [390.61,  0.0,   330.444],
    [  0.0, 390.61,  238.868],
    [  0.0,   0.0,     1.0  ]
], dtype=np.float64)

# 4x4 transformation matrix: points_cam2 = T @ points_cam1
T_cam1_to_cam2 = np.eye(4, dtype=np.float64)  # replace with your matrix

DEPTH_SCALE = 0.001   # mm → meters
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0

# Subsample to keep the overlay readable (1 = use all points)
SUBSAMPLE = 10

# ---------------------------------------------------------------------------
# Optional: known 2D correspondences for quantitative RMSE
#   pts1: (N,2) pixel coords in cam1 image
#   pts3d_cam1: (N,3) corresponding 3D points in cam1 frame
#   pts2_gt: (N,2) ground-truth pixel coords in cam2 image
# Set to None to skip RMSE computation.
# ---------------------------------------------------------------------------
pts1       = None  # e.g. np.array([[u1,v1], [u2,v2], ...])
pts3d_cam1 = None  # e.g. obtained from depth at those pixels
pts2_gt    = None  # e.g. matched with SIFT/ORB in cam2 image

# ---------------------------------------------------------------------------


def unproject(depth_m, K, min_d=MIN_DEPTH_M, max_d=MAX_DEPTH_M):
    """Depth image → (N,3) 3D points in camera frame."""
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_m > min_d) & (depth_m < max_d)
    Z = depth_m[valid]
    X = (u[valid] - K[0, 2]) * Z / K[0, 0]
    Y = (v[valid] - K[1, 2]) * Z / K[1, 1]
    return np.stack([X, Y, Z], axis=-1)   # (N,3)


def transform_points(pts3d, T):
    """Apply 4x4 rigid transform to (N,3) points."""
    ones = np.ones((len(pts3d), 1), dtype=np.float64)
    pts_h = np.hstack([pts3d, ones])          # (N,4)
    return (T @ pts_h.T).T[:, :3]            # (N,3)


def project(pts3d, K):
    """Project (N,3) 3D points → (N,2) pixel coords (float)."""
    pts3d = pts3d[pts3d[:, 2] > 0]           # only points in front
    uv = (K @ pts3d.T).T                      # (N,3)
    uv = uv[:, :2] / uv[:, 2:3]             # (N,2)
    return uv


def overlay_points(img, uv, color=(0, 255, 0), radius=2):
    """Draw projected points on a copy of img."""
    out = img.copy()
    H, W = img.shape[:2]
    for (u, v) in uv.astype(int):
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (u, v), radius, color, -1)
    return out


def reprojection_rmse(pts3d_cam1, T, K2, pts2_gt):
    """Compute reprojection RMSE given 3D points and GT 2D points in cam2."""
    pts3d_cam2 = transform_points(pts3d_cam1, T)
    uv_proj = (K2 @ pts3d_cam2.T).T
    uv_proj = uv_proj[:, :2] / uv_proj[:, 2:3]
    err = np.linalg.norm(uv_proj - pts2_gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

depth_raw = cv2.imread(CAM1_DEPTH_PATH, cv2.IMREAD_UNCHANGED)
if depth_raw is None:
    raise FileNotFoundError(f"Cannot read: {CAM1_DEPTH_PATH}")
depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE

cam2_img = cv2.imread(CAM2_RGB_PATH)
if cam2_img is None:
    raise FileNotFoundError(f"Cannot read: {CAM2_RGB_PATH}")

# 1. Unproject cam1 depth → 3D (cam1 frame)
pts3d_cam1_all = unproject(depth_m, K1)
print(f"Unprojected {len(pts3d_cam1_all)} points from cam1 depth")

# 2. Transform to cam2 frame
pts3d_cam2_all = transform_points(pts3d_cam1_all, T_cam1_to_cam2)

# 3. Project onto cam2 image
uv_all = project(pts3d_cam2_all, K2)
uv_sub = uv_all[::SUBSAMPLE]

# 4. Overlay on cam2 image
overlay = overlay_points(cam2_img, uv_sub, color=(0, 255, 0))
cv2.imshow("Reprojection on cam2 (green = projected from cam1 depth)", overlay)
cv2.imwrite("/tmp/reprojection_overlay.png", overlay)
print("Overlay saved to /tmp/reprojection_overlay.png")

# 5. Quantitative RMSE (if correspondences provided)
if pts3d_cam1 is not None and pts2_gt is not None:
    rmse, per_point = reprojection_rmse(
        np.array(pts3d_cam1, dtype=np.float64),
        T_cam1_to_cam2,
        K2,
        np.array(pts2_gt, dtype=np.float64)
    )
    print(f"\nReprojection RMSE: {rmse:.3f} px")
    print(f"Per-point errors (px): {per_point.round(2)}")
else:
    print("\nNo 2D correspondences provided — skipping RMSE. "
          "Set pts3d_cam1 and pts2_gt to get a numeric error.")

cv2.waitKey(0)
cv2.destroyAllWindows()
