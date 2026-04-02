num_hands = 2
num_cameras = 2
date = '0322'
time = '1939'
system_delay = 17815
show_visualizer = True

from pathlib import Path

import numpy as np

import config
config.configure(num_hands=num_hands)

from acquisition_utils import load_mocap_log, load_realsense_log
from processing_utils import (
    apply_rigid_transforms_per_marker,
    compute_detailed_errors,
    compute_rigid_transforms_per_marker,
    detect_marker_anomalies,
    filter_matching_data,
    find_matching_frames,
)
from visualizer import MarkerVisualizer


MATCH_THRESHOLD_MS = 10
ANOMALY_EPS = 50
ANOMALY_MIN_SAMPLES = 20


def get_mocap_log_path():
    return Path(f'./logs/{date}_{time}_mocap_log.txt')


def get_realsense_log_path(camera_idx):
    if num_cameras == 1:
        return Path(f'./logs/{date}_{time}_realsense_log.txt')
    return Path(f'./logs/{date}_{time}_cam{camera_idx}_realsense_log.txt')


def summarize_overall_errors(errors):
    print("=== Overall Error Summary ===")
    print(f"Mean error: {errors.mean():.2f} mm")
    print(f"Std  error: {errors.std():.2f} mm")
    print(f"Max  error: {errors.max():.2f} mm")


def print_total_error_summary(camera_results):
    print("\n=== Total Error Summary (All Cameras Combined) ===")

    for marker_name in config.NAMES:
        marker_errors = np.concatenate([
            result["error_stats"][marker_name]["all"]
            for result in camera_results
        ])
        print(
            f"{marker_name}: mean={marker_errors.mean():.2f} mm | "
            f"std={marker_errors.std():.2f} mm | "
            f"max={marker_errors.max():.2f} mm"
        )

    combined_errors = np.concatenate([result["errors"] for result in camera_results])
    summarize_overall_errors(combined_errors)


def remove_realsense_anomalies(rs_data, camera_label):
    rs_anomalies, n = detect_marker_anomalies(
        rs_data,
        eps=ANOMALY_EPS,
        min_samples=ANOMALY_MIN_SAMPLES,
    )
    rs_anomalies_times = sorted({
        timestamp
        for anomaly_times in rs_anomalies.values()
        for timestamp in anomaly_times
    })
    print(
        f"Detected {n} anomalous markers and "
        f"{len(rs_anomalies_times)} anomalous timestamps in {camera_label} data."
    )

    for timestamp in rs_anomalies_times:
        rs_data.pop(timestamp, None)

    return rs_data


def analyze_camera(mc_data, camera_idx):
    camera_label = "realsense" if num_cameras == 1 else f"cam{camera_idx}"
    rs_path = get_realsense_log_path(camera_idx)
    if not rs_path.exists():
        raise FileNotFoundError(f"Missing Realsense log: {rs_path}")

    rs_data = load_realsense_log(str(rs_path))

    print(f"\n=== {camera_label} vs mocap ===")
    print(f"Total {camera_label} frames: {len(rs_data)}")

    rs_data = remove_realsense_anomalies(rs_data, camera_label)

    matched_pairs = find_matching_frames(mc_data, rs_data, threshold=MATCH_THRESHOLD_MS)
    print(f"Matched frame count: {len(matched_pairs)}")
    if not matched_pairs:
        raise ValueError(f"No matched frames found for {camera_label}.")

    mocap_matched, rs_matched = filter_matching_data(mc_data, rs_data, matched_pairs)

    transformations = compute_rigid_transforms_per_marker(rs_matched, mocap_matched)
    rs_transformed = apply_rigid_transforms_per_marker(rs_matched, transformations)

    mocap_vec = np.vstack(list(mocap_matched.values()))
    rs_vec = np.vstack(list(rs_transformed.values()))

    error_stats = compute_detailed_errors(mocap_vec, rs_vec)
    errors = np.linalg.norm(rs_vec - mocap_vec, axis=1)

    return {
        "camera_label": camera_label,
        "mocap_matched": mocap_matched,
        "rs_transformed": rs_transformed,
        "error_stats": error_stats,
        "errors": errors,
    }


if num_cameras not in (1, 2):
    raise ValueError(f"Unsupported num_cameras={num_cameras}. Expected 1 or 2.")


mocap_path = get_mocap_log_path()
if not mocap_path.exists():
    raise FileNotFoundError(f"Missing mocap log: {mocap_path}")

mc = load_mocap_log(str(mocap_path), system_delay=system_delay)
print(f"Total mocap frames: {len(mc)}")

camera_indices = [1] if num_cameras == 1 else [1, 2]
camera_results = [analyze_camera(mc, camera_idx) for camera_idx in camera_indices]

if num_cameras == 2:
    print_total_error_summary(camera_results)


if show_visualizer:
    mocap_labels = ["(mc)" + name for name in config.NAMES]
    rs_labels = ["(rs)" + name for name in config.NAMES]
    visualized_result = camera_results[1]

    if num_cameras == 2:
        print("\nVisualizer shows mocap and cam1 after alignment.")

    vis = MarkerVisualizer(
        data_dict1=visualized_result["mocap_matched"],
        data_dict2=visualized_result["rs_transformed"],
        labels1=mocap_labels,
        labels2=rs_labels,
    )
    vis.show()
