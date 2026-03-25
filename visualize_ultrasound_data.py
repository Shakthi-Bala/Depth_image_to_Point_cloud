#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# =========================================================
# CONFIG
# =========================================================
# Change this if your new simulated file has a different name.
NPZ_PATH_1 = "/home/alien/depth_img_to_pc/test_data/us_on_ply_out/ultrasound_data.npz"
NPZ_PATH_2 = "/home/alien/depth_img_to_pc/test_data/ultrasound.npz"

OUTPUT_VIDEO_PATH = "/home/alien/depth_img_to_pc/test_data/ultrasound_compare.mp4"

VIDEO_FPS = 12
DPI = 140
FIG_W = 16
FIG_H = 9

TITLE_1 = "Simulated Ultrasound"
TITLE_2 = "Recorded Ultrasound"

MAX_FRAMES = None
USE_GLOBAL_YLIM = True
SHOW_RAY_DIST = False

# If enabled, both waveforms are resampled to the same number of bins
# so side-by-side comparison looks cleaner.
RESAMPLE_TO_COMMON_BINS = True

# If you changed the output npz filename in the simulator, update only NPZ_PATH_1.
# This script already supports the newer exact-distance keys:
#   exact_first_return_distances
#   exact_first_return_times
#   exact_first_return_sample_indices


# =========================================================
# HELPERS
# =========================================================
def fail(msg: str):
    raise RuntimeError(msg)


def load_npz_dict(path: str):
    if not os.path.isfile(path):
        fail(f"NPZ file not found: {path}")
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def describe_npz(name: str, d: dict):
    print(f"\n[INFO] {name}:")
    for k, v in d.items():
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"  - {k}: shape={shape}, dtype={dtype}")


def normalize_img(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = 255.0 * (arr - mn) / (mx - mn)
    return np.clip(out, 0, 255).astype(np.uint8)


def raydist_to_brightness(ray_dist: np.ndarray):
    ray_dist = np.asarray(ray_dist, dtype=np.float32)
    finite_vals = ray_dist[np.isfinite(ray_dist)]
    if finite_vals.size == 0:
        return np.zeros(ray_dist.shape, dtype=np.uint8)

    mn = finite_vals.min()
    mx = finite_vals.max()
    if mx <= mn:
        return np.zeros(ray_dist.shape, dtype=np.uint8)

    norm = 1.0 - (ray_dist - mn) / (mx - mn)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(255.0 * norm, 0, 255).astype(np.uint8)


def repeat_to_frames(vec: np.ndarray, nframes: int):
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    return np.repeat(vec, nframes, axis=0)


def repeat_scalar_to_frames(value, nframes: int):
    return np.repeat(np.array([value], dtype=np.float32), nframes, axis=0)


def get_scalar_if_exists(d: dict, key: str, default=np.nan):
    if key not in d:
        return default
    arr = np.asarray(d[key])
    if arr.shape == ():
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return default


def resample_2d_rows(arr: np.ndarray, target_bins: int):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        fail(f"Expected 2D array for resampling, got shape {arr.shape}")
    src_bins = arr.shape[1]
    if src_bins == target_bins:
        return arr.copy()

    x_src = np.linspace(0.0, 1.0, src_bins, dtype=np.float32)
    x_tgt = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)

    out = np.zeros((arr.shape[0], target_bins), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.interp(x_tgt, x_src, arr[i]).astype(np.float32)
    return out


# =========================================================
# EXTRACTOR
# =========================================================
def extract_ultrasound_payload(d: dict, fallback_frames: int = None, fps_guess: float = 20.0):
    """
    Returns:
        {
            "env": 2D array [frames, bins],
            "ray": 2D array or None,
            "peak": 1D array [frames],                  # envelope-derived or scalar distance
            "exact_dist": 1D array [frames],            # exact first-return distance if available
            "exact_time": 1D array [frames],            # exact first-return time if available
            "exact_sample_idx": 1D array [frames],      # exact first-return sample if available
            "time": 1D array [frames],
            "kind": "sequence" or "single",
            "source_key": str
        }
    """

    # -----------------------------------------------------
    # Case 1: full sequence with envelopes
    # -----------------------------------------------------
    if "envelopes" in d:
        env = np.asarray(d["envelopes"], dtype=np.float32)
        if env.ndim != 2:
            fail(f"'envelopes' exists but is not 2D: shape={env.shape}")

        nframes = env.shape[0]

        ray = None
        if "ray_distances" in d:
            ray = np.asarray(d["ray_distances"], dtype=np.float32)
            if ray.ndim != 2:
                ray = None
            elif ray.shape[0] != nframes:
                ray = ray[:nframes]

        if "peak_distances" in d:
            peak = np.asarray(d["peak_distances"], dtype=np.float32).reshape(-1)
            if len(peak) < nframes:
                peak = np.pad(peak, (0, nframes - len(peak)), constant_values=np.nan)
            else:
                peak = peak[:nframes]
        else:
            peak = np.full((nframes,), np.nan, dtype=np.float32)

        if "exact_first_return_distances" in d:
            exact_dist = np.asarray(d["exact_first_return_distances"], dtype=np.float32).reshape(-1)
            if len(exact_dist) < nframes:
                exact_dist = np.pad(exact_dist, (0, nframes - len(exact_dist)), constant_values=np.nan)
            else:
                exact_dist = exact_dist[:nframes]
        else:
            exact_dist = peak.copy()

        if "exact_first_return_times" in d:
            exact_time = np.asarray(d["exact_first_return_times"], dtype=np.float32).reshape(-1)
            if len(exact_time) < nframes:
                exact_time = np.pad(exact_time, (0, nframes - len(exact_time)), constant_values=np.nan)
            else:
                exact_time = exact_time[:nframes]
        else:
            exact_time = np.full((nframes,), np.nan, dtype=np.float32)

        if "exact_first_return_sample_indices" in d:
            exact_sample_idx = np.asarray(d["exact_first_return_sample_indices"], dtype=np.int32).reshape(-1)
            if len(exact_sample_idx) < nframes:
                exact_sample_idx = np.pad(exact_sample_idx, (0, nframes - len(exact_sample_idx)), constant_values=-1)
            else:
                exact_sample_idx = exact_sample_idx[:nframes]
        else:
            exact_sample_idx = np.full((nframes,), -1, dtype=np.int32)

        if "sim_times" in d:
            time_arr = np.asarray(d["sim_times"], dtype=np.float32).reshape(-1)
            if len(time_arr) < nframes:
                extra = np.arange(len(time_arr), nframes, dtype=np.float32) / float(fps_guess)
                time_arr = np.concatenate([time_arr, extra], axis=0)
            else:
                time_arr = time_arr[:nframes]
        else:
            time_arr = np.arange(nframes, dtype=np.float32) / float(fps_guess)

        return {
            "env": env,
            "ray": ray,
            "peak": peak,
            "exact_dist": exact_dist,
            "exact_time": exact_time,
            "exact_sample_idx": exact_sample_idx,
            "time": time_arr,
            "kind": "sequence",
            "source_key": "envelopes",
        }

    # -----------------------------------------------------
    # Case 2: single recorded capture with magnitude
    # -----------------------------------------------------
    if "magnitude" in d:
        mag = np.asarray(d["magnitude"], dtype=np.float32).reshape(-1)
        nframes = fallback_frames if fallback_frames is not None else 1
        env = repeat_to_frames(mag, nframes)

        dist = get_scalar_if_exists(d, "distance", np.nan)
        peak = repeat_scalar_to_frames(dist, nframes)
        exact_dist = repeat_scalar_to_frames(dist, nframes)

        tsec = get_scalar_if_exists(d, "timestamp_sec", np.nan)
        if np.isnan(tsec):
            time_arr = np.arange(nframes, dtype=np.float32) / float(fps_guess)
        else:
            time_arr = repeat_scalar_to_frames(tsec, nframes)

        return {
            "env": env,
            "ray": None,
            "peak": peak,
            "exact_dist": exact_dist,
            "exact_time": np.full((nframes,), np.nan, dtype=np.float32),
            "exact_sample_idx": np.full((nframes,), -1, dtype=np.int32),
            "time": time_arr,
            "kind": "single",
            "source_key": "magnitude",
        }

    # -----------------------------------------------------
    # Case 3: single recorded capture with idata/qdata only
    # -----------------------------------------------------
    if "idata" in d and "qdata" in d:
        i = np.asarray(d["idata"], dtype=np.float32).reshape(-1)
        q = np.asarray(d["qdata"], dtype=np.float32).reshape(-1)
        mag = np.sqrt(i * i + q * q)

        nframes = fallback_frames if fallback_frames is not None else 1
        env = repeat_to_frames(mag, nframes)

        dist = get_scalar_if_exists(d, "distance", np.nan)
        peak = repeat_scalar_to_frames(dist, nframes)
        exact_dist = repeat_scalar_to_frames(dist, nframes)

        tsec = get_scalar_if_exists(d, "timestamp_sec", np.nan)
        if np.isnan(tsec):
            time_arr = np.arange(nframes, dtype=np.float32) / float(fps_guess)
        else:
            time_arr = repeat_scalar_to_frames(tsec, nframes)

        return {
            "env": env,
            "ray": None,
            "peak": peak,
            "exact_dist": exact_dist,
            "exact_time": np.full((nframes,), np.nan, dtype=np.float32),
            "exact_sample_idx": np.full((nframes,), -1, dtype=np.int32),
            "time": time_arr,
            "kind": "single",
            "source_key": "idata_qdata",
        }

    # -----------------------------------------------------
    # Case 4: fallback 2D array heuristic
    # -----------------------------------------------------
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] >= 32:
            nframes = arr.shape[0]
            return {
                "env": arr.astype(np.float32),
                "ray": None,
                "peak": np.full((nframes,), np.nan, dtype=np.float32),
                "exact_dist": np.full((nframes,), np.nan, dtype=np.float32),
                "exact_time": np.full((nframes,), np.nan, dtype=np.float32),
                "exact_sample_idx": np.full((nframes,), -1, dtype=np.int32),
                "time": np.arange(nframes, dtype=np.float32) / float(fps_guess),
                "kind": "sequence",
                "source_key": k,
            }

    fail("Could not find usable ultrasound waveform data in npz")


# =========================================================
# LOAD
# =========================================================
npz1 = load_npz_dict(NPZ_PATH_1)
npz2 = load_npz_dict(NPZ_PATH_2)

describe_npz("NPZ 1", npz1)
describe_npz("NPZ 2", npz2)

payload1 = extract_ultrasound_payload(npz1, fallback_frames=None, fps_guess=20.0)
env1 = payload1["env"]
nframes1 = env1.shape[0]

payload2 = extract_ultrasound_payload(npz2, fallback_frames=nframes1, fps_guess=20.0)
env2 = payload2["env"]

nframes = min(env1.shape[0], env2.shape[0])
if MAX_FRAMES is not None:
    nframes = min(nframes, int(MAX_FRAMES))

env1 = env1[:nframes]
env2 = env2[:nframes]

peak1 = payload1["peak"][:nframes]
peak2 = payload2["peak"][:nframes]

exact_dist1 = payload1["exact_dist"][:nframes]
exact_dist2 = payload2["exact_dist"][:nframes]

exact_time1 = payload1["exact_time"][:nframes]
exact_time2 = payload2["exact_time"][:nframes]

exact_sample_idx1 = payload1["exact_sample_idx"][:nframes]
exact_sample_idx2 = payload2["exact_sample_idx"][:nframes]

time1 = payload1["time"][:nframes]
time2 = payload2["time"][:nframes]

print("\n[INFO] Using extracted payloads:")
print(f"  NPZ1 kind       : {payload1['kind']}")
print(f"  NPZ1 source     : {payload1['source_key']}")
print(f"  NPZ1 env shape  : {env1.shape}")
print(f"  NPZ2 kind       : {payload2['kind']}")
print(f"  NPZ2 source     : {payload2['source_key']}")
print(f"  NPZ2 env shape  : {env2.shape}")
print(f"  Frames used     : {nframes}")

if nframes <= 0:
    fail("No frames available for video")

# Optional resampling for cleaner side-by-side visual comparison
if RESAMPLE_TO_COMMON_BINS:
    common_bins = max(env1.shape[1], env2.shape[1])
    env1 = resample_2d_rows(env1, common_bins)
    env2 = resample_2d_rows(env2, common_bins)
    bins1 = common_bins
    bins2 = common_bins
else:
    bins1 = env1.shape[1]
    bins2 = env2.shape[1]

# =========================================================
# PRECOMPUTE
# =========================================================
if USE_GLOBAL_YLIM:
    y_max = float(max(np.nanmax(env1), np.nanmax(env2)))
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0
else:
    y_max = None

# =========================================================
# FIGURE
# =========================================================
fig, (ax_env1, ax_env2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI)
fig.suptitle("Ultrasound Comparison", fontsize=14)

x1 = np.arange(bins1)
x2 = np.arange(bins2)

line1, = ax_env1.plot(x1, np.zeros_like(x1, dtype=np.float32))
line2, = ax_env2.plot(x2, np.zeros_like(x2, dtype=np.float32))

ax_env1.set_title(f"{TITLE_1} - Current Envelope")
ax_env2.set_title(f"{TITLE_2} - Current Envelope")

ax_env1.set_xlabel("Bin")
ax_env2.set_xlabel("Bin")
ax_env1.set_ylabel("Amplitude")
ax_env2.set_ylabel("Amplitude")

ax_env1.set_xlim(0, bins1 - 1)
ax_env2.set_xlim(0, bins2 - 1)

if y_max is not None:
    ax_env1.set_ylim(0, y_max * 1.05)
    ax_env2.set_ylim(0, y_max * 1.05)

frame_text = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# =========================================================
# SAVE VIDEO
# =========================================================
out_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

writer = FFMpegWriter(
    fps=VIDEO_FPS,
    metadata={"title": "Ultrasound Comparison"},
    bitrate=2500
)

print(f"\n[INFO] Saving video to: {OUTPUT_VIDEO_PATH}")

with writer.saving(fig, OUTPUT_VIDEO_PATH, DPI):
    for i in range(nframes):
        line1.set_ydata(env1[i])
        line2.set_ydata(env2[i])

        if not USE_GLOBAL_YLIM:
            ymax1 = max(1.0, float(np.nanmax(env1[i])) * 1.1)
            ymax2 = max(1.0, float(np.nanmax(env2[i])) * 1.1)
            ax_env1.set_ylim(0, ymax1)
            ax_env2.set_ylim(0, ymax2)

        p1 = peak1[i] if i < len(peak1) else np.nan
        p2 = peak2[i] if i < len(peak2) else np.nan
        ed1 = exact_dist1[i] if i < len(exact_dist1) else np.nan
        ed2 = exact_dist2[i] if i < len(exact_dist2) else np.nan
        t1 = time1[i] if i < len(time1) else np.nan
        t2 = time2[i] if i < len(time2) else np.nan

        frame_text.set_text(
            f"Frame {i+1}/{nframes}   "
            f"{TITLE_1}: t={t1:.3f}s peak={p1:.3f}m exact={ed1:.3f}m   "
            f"{TITLE_2}: t={t2:.3f}s peak={p2:.3f}m exact={ed2:.3f}m"
        )

        writer.grab_frame()

plt.close(fig)
print("[INFO] Done.")