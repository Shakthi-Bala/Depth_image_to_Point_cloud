#!/usr/bin/env python3

import os
import sys
import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
PLY_PATH = "/home/alien/depth_img_to_pc/test_data/pointcloud.ply"
OUTPUT_IMG_PATH = "/home/alien/depth_img_to_pc/test_data/pointcloud_view.png"

VOXEL_SIZE = 0.01          # set None to disable downsampling
REMOVE_OUTLIERS = True
NB_NEIGHBORS = 20
STD_RATIO = 2.0

FLIP_Y = False             # set True if orientation looks upside down
FLIP_Z = False             # set True if needed

FIG_W = 10
FIG_H = 8
DPI = 200
POINT_SIZE = 0.3           # matplotlib marker size

# Camera/view for saved image
ELEV = 20                  # elevation angle
AZIM = -60                 # azimuth angle

# Axis display
SHOW_AXES = True
SHOW_TITLE = True
TITLE = "Point Cloud"

# Background
WHITE_BG = True


# =========================================================
# HELPERS
# =========================================================
def fail(msg: str):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def set_axes_equal(ax, points):
    """
    Make 3D plot axes have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_mid = 0.5 * (x.min() + x.max())
    y_mid = 0.5 * (y.min() + y.max())
    z_mid = 0.5 * (z.min() + z.max())

    max_range = 0.5 * max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    )

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


# =========================================================
# LOAD POINT CLOUD
# =========================================================
print(f"[INFO] Loading point cloud from: {PLY_PATH}")

if not os.path.isfile(PLY_PATH):
    fail(f"PLY file not found: {PLY_PATH}")

pcd = o3d.io.read_point_cloud(PLY_PATH)

if pcd is None or len(pcd.points) == 0:
    fail("Point cloud is empty or could not be read")

print(f"[INFO] Loaded {len(pcd.points)} points")

# =========================================================
# OPTIONAL AXIS FLIPS
# =========================================================
if FLIP_Y or FLIP_Z:
    pts = np.asarray(pcd.points).copy()

    if FLIP_Y:
        pts[:, 1] *= -1.0
    if FLIP_Z:
        pts[:, 2] *= -1.0

    pcd.points = o3d.utility.Vector3dVector(pts)
    print("[INFO] Applied axis flip")

# =========================================================
# OPTIONAL DOWNSAMPLE
# =========================================================
if VOXEL_SIZE is not None and VOXEL_SIZE > 0:
    print(f"[INFO] Applying voxel downsampling: {VOXEL_SIZE} m")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"[INFO] After downsampling: {len(pcd.points)} points")

# =========================================================
# OPTIONAL OUTLIER REMOVAL
# =========================================================
if REMOVE_OUTLIERS:
    print("[INFO] Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )
    print(f"[INFO] After filtering: {len(pcd.points)} points")

# =========================================================
# EXTRACT POINTS / COLORS
# =========================================================
points = np.asarray(pcd.points)

if points.shape[0] == 0:
    fail("No points left after filtering")

print("[INFO] Bounding box:")
print("       min:", points.min(axis=0))
print("       max:", points.max(axis=0))

has_colors = pcd.has_colors()
if has_colors:
    colors = np.asarray(pcd.colors)
    if colors.shape[0] != points.shape[0]:
        has_colors = False

# =========================================================
# PLOT AND SAVE
# =========================================================
print("[INFO] Saving point cloud image...")

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = fig.add_subplot(111, projection="3d")

if WHITE_BG:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
else:
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

if has_colors:
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors,
        s=POINT_SIZE,
        depthshade=False
    )
else:
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=POINT_SIZE,
        depthshade=False
    )

ax.view_init(elev=ELEV, azim=AZIM)
set_axes_equal(ax, points)

if SHOW_TITLE:
    ax.set_title(TITLE)

if not SHOW_AXES:
    ax.set_axis_off()
else:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

plt.tight_layout()
plt.savefig(OUTPUT_IMG_PATH, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)

print(f"[INFO] Saved image to: {OUTPUT_IMG_PATH}")
print("[INFO] Done.")