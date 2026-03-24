import cv2
import numpy as np
import open3d as o3d

DEPTH_PNG_PATH = "/path/to/depth.png"
RGB_PATH = "/path/to/rgb.png"
OUTPUT_PLY_PATH = "/path/to/colored_pointcloud.ply"

# Example intrinsics - Intel RealSense D455f
fx = 390.61
fy = 390.61
cx = 330.444
cy = 238.868

DEPTH_SCALE_TO_METERS = 0.001
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0

depth_raw = cv2.imread(DEPTH_PNG_PATH, cv2.IMREAD_UNCHANGED)
rgb = cv2.imread(RGB_PATH, cv2.IMREAD_COLOR)

if depth_raw is None:
    raise FileNotFoundError(f"Could not read depth image: {DEPTH_PNG_PATH}")
if rgb is None:
    raise FileNotFoundError(f"Could not read RGB image: {RGB_PATH}")

rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_TO_METERS

H, W = depth_m.shape
if rgb.shape[:2] != (H, W):
    raise ValueError("RGB and depth image sizes do not match")

u, v = np.meshgrid(np.arange(W), np.arange(H))
valid = (depth_m > MIN_DEPTH_M) & (depth_m < MAX_DEPTH_M)

Z = depth_m[valid]
U = u[valid].astype(np.float32)
V = v[valid].astype(np.float32)

X = (U - cx) * Z / fx
Y = (V - cy) * Z / fy

points = np.stack((X, Y, Z), axis=-1)
colors = rgb[valid].astype(np.float32) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(OUTPUT_PLY_PATH, pcd)
print(f"Saved colored point cloud to: {OUTPUT_PLY_PATH}")

o3d.visualization.draw_geometries([pcd])