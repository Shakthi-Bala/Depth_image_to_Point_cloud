import cv2
import numpy as np
import open3d as o3d


DEPTH_PNG_PATH = "/home/alien/depth_img_to_pc/test_data/depth.png"
OUTPUT_PLY_PATH = "/home/alien/depth_img_to_pc/test_data/pointcloud.ply"

# Example: use these if your depth image is 640x480
fx = 390.61
fy = 390.61
cx = 330.444
cy = 238.868

DEPTH_SCALE_TO_METERS = 0.001

MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0


depth_raw = cv2.imread(DEPTH_PNG_PATH, cv2.IMREAD_UNCHANGED)

if depth_raw is None:
    raise FileNotFoundError(f"Could not read depth image: {DEPTH_PNG_PATH}")

if len(depth_raw.shape) != 2:
    raise ValueError("Depth image must be single-channel")

print("Depth dtype:", depth_raw.dtype)
print("Depth shape:", depth_raw.shape)


depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_TO_METERS

H, W = depth_m.shape


u, v = np.meshgrid(np.arange(W), np.arange(H))


valid = (depth_m > MIN_DEPTH_M) & (depth_m < MAX_DEPTH_M)

Z = depth_m[valid]
U = u[valid].astype(np.float32)
V = v[valid].astype(np.float32)

X = (U - cx) * Z / fx
Y = (V - cy) * Z / fy

points = np.stack((X, Y, Z), axis=-1)

print(f"Generated {points.shape[0]} 3D points")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.io.write_point_cloud(OUTPUT_PLY_PATH, pcd)
print(f"Saved point cloud to: {OUTPUT_PLY_PATH}")

o3d.visualization.draw_geometries([pcd])