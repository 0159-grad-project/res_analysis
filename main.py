date = '0415'
time = '1513'
num_cameras = 2
num_hands = None  # Set to None to infer from the mocap log.
show_visualizer = False
system_delay = None  # Set to None to enable automatic estimation, or specify a fixed delay in ms
ALIGNMENT_MODE = "per_marker"  # per_camera
CALIBRATION_RATIO = 0.2  # None means using all frames for both transform and error

from pathlib import Path

import numpy as np

import config

from acquisition_utils import load_mocap_log, load_realsense_log
from estimate_system_delay import estimate_system_delay, infer_num_hands_from_mocap
from fusion_utils import analyze_weighted_fusion
from processing_utils import (
    apply_rigid_transform,
    apply_rigid_transforms_per_marker,
    build_interpolated_reference,
    compute_detailed_errors,
    compute_rigid_transform,
    compute_rigid_transforms_per_marker,
    detect_marker_anomalies,
    filter_data_by_timestamps,
    split_timestamps_by_ratio,
)
from visualizer import MarkerVisualizer, plot_marker_error_histogram


MOCAP_INTERP_MAX_GAP_MS = 30
CAMERA_PAIR_THRESHOLD_MS = 30
ANOMALY_EPS = 50
ANOMALY_MIN_SAMPLES = 20


def get_mocap_log_path():
    return Path(f'./logs/{date}_{time}_mocap_log.txt')


def get_realsense_log_path(camera_idx):
    if num_cameras == 1:
        return Path(f'./logs/{date}_{time}_realsense_log.txt')
    return Path(f'./logs/{date}_{time}_cam{camera_idx}_realsense_log.txt')


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

    rs_data = load_realsense_log(
        str(rs_path),
        num_hands=num_hands,
    )

    print(f"\n=== {camera_label} vs mocap ===")
    print(f"Total {camera_label} frames: {len(rs_data)}")
    # print(f"Alignment mode: {ALIGNMENT_MODE}")

    rs_data = remove_realsense_anomalies(rs_data, camera_label)

    mocap_reference = build_interpolated_reference(
        mc_data,
        sorted(rs_data.keys()),
        max_gap_ms=MOCAP_INTERP_MAX_GAP_MS,
    )
    print(f"Interpolated mocap frame count: {len(mocap_reference)}")
    if not mocap_reference:
        raise ValueError(f"No interpolated mocap frames found for {camera_label}.")

    rs_reference = {
        timestamp: np.asarray(rs_data[timestamp], dtype=float)
        for timestamp in mocap_reference
    }
    calibration_timestamps, evaluation_timestamps = split_timestamps_by_ratio(
        mocap_reference.keys(),
        calibration_ratio=CALIBRATION_RATIO,
    )

    # if CALIBRATION_RATIO is None:
    #     print(
    #         f"Using all {len(calibration_timestamps)} interpolated frames "
    #         "for both coordinate transform and error computation."
    #     )
    # else:
    #     print(f"Calibration ratio: {CALIBRATION_RATIO:.0%}")
    #     print(f"Calibration frame count: {len(calibration_timestamps)}")
    #     print(f"Evaluation frame count: {len(evaluation_timestamps)}")

    mocap_calibration = filter_data_by_timestamps(mocap_reference, calibration_timestamps)
    rs_calibration = filter_data_by_timestamps(rs_reference, calibration_timestamps)
    mocap_evaluation = filter_data_by_timestamps(mocap_reference, evaluation_timestamps)

    transform = None
    if ALIGNMENT_MODE == "per_marker":
        transform = compute_rigid_transforms_per_marker(
            rs_calibration,
            mocap_calibration,
        )
        rs_transformed_all = apply_rigid_transforms_per_marker(rs_data, transform)
    elif ALIGNMENT_MODE == "per_camera":
        rotation, translation = compute_rigid_transform(rs_calibration, mocap_calibration)
        transform = (rotation, translation)
        rs_transformed_all = apply_rigid_transform(rs_data, rotation, translation)
    else:
        raise ValueError(
            f"Unsupported ALIGNMENT_MODE={ALIGNMENT_MODE}. "
            "Expected 'per_marker' or 'per_camera'."
        )

    rs_transformed_calibration = filter_data_by_timestamps(rs_transformed_all, calibration_timestamps)
    rs_transformed = filter_data_by_timestamps(rs_transformed_all, evaluation_timestamps)
    rs_transformed_for_fusion = rs_transformed

    mocap_vec = np.vstack(list(mocap_evaluation.values()))
    rs_vec = np.vstack(list(rs_transformed.values()))

    error_stats = compute_detailed_errors(mocap_vec, rs_vec, MARKER_NAMES)
    errors = np.linalg.norm(rs_vec - mocap_vec, axis=1)
    weight_error_stats = error_stats
    if CALIBRATION_RATIO is not None:
        calibration_mocap_vec = np.vstack(list(mocap_calibration.values()))
        calibration_rs_vec = np.vstack(list(rs_transformed_calibration.values()))
        weight_error_stats = compute_detailed_errors(
            calibration_mocap_vec,
            calibration_rs_vec,
            MARKER_NAMES,
            print_summary=False,
        )

    return {
        "camera_label": camera_label,
        "mocap_matched": mocap_evaluation,
        "rs_transformed": rs_transformed,
        "rs_transformed_all": rs_transformed_all,
        "rs_transformed_for_fusion": rs_transformed_for_fusion,
        "error_stats": error_stats,
        "weight_error_stats": weight_error_stats,
        "errors": errors,
        "transform": transform,
        "calibration_timestamps": calibration_timestamps,
        "evaluation_timestamps": evaluation_timestamps,
    }


if num_cameras not in (1, 2):
    raise ValueError(f"Unsupported num_cameras={num_cameras}. Expected 1 or 2.")


mocap_path = get_mocap_log_path()
if not mocap_path.exists():
    raise FileNotFoundError(f"Missing mocap log: {mocap_path}")

if num_hands is None:
    num_hands = infer_num_hands_from_mocap(mocap_path)
    print(f"Inferred num_hands: {num_hands}")

MARKER_NAMES = config.get_marker_names(num_hands)

camera_indices = [1] if num_cameras == 1 else [1, 2]
if system_delay is None:
    delay_results = [
        estimate_system_delay(
            mocap_path,
            get_realsense_log_path(camera_idx),
            num_hands=num_hands,
        )
        for camera_idx in camera_indices
    ]

    if num_cameras == 1:
        system_delay = delay_results[0]["delay_ms"]
        print(f"Estimated system_delay: {system_delay} ms")
    else:
        system_delay = int(round(np.mean([result["delay_ms"] for result in delay_results])))
        average_mean_frame_error = float(np.mean([result["mean_frame_error_mm"] for result in delay_results]))
        # for camera_idx, result in zip(camera_indices, delay_results):
        #     print(
        #         f"Camera {camera_idx} estimated system_delay: {result['delay_ms']} ms "
        #         f"(mean frame error: {result['mean_frame_error_mm']:.2f} mm)"
        #     )
        print(f"Averaged system_delay: {system_delay} ms")
        # print(f"Average estimated mean frame error: {average_mean_frame_error:.2f} mm")
else:
    print(f"Using manual system_delay: {system_delay} ms")

mc = load_mocap_log(str(mocap_path), num_hands=num_hands, system_delay=system_delay)
print(f"Total mocap frames: {len(mc)}")

camera_results = [analyze_camera(mc, camera_idx) for camera_idx in camera_indices]
fused_result = None

if num_cameras == 2:
    fused_result = analyze_weighted_fusion(
        camera_results,
        mc,
        MARKER_NAMES,
        pair_threshold_ms=CAMERA_PAIR_THRESHOLD_MS,
        mocap_interp_max_gap_ms=MOCAP_INTERP_MAX_GAP_MS,
    )


if show_visualizer:
    mocap_labels = ["(mc)" + name for name in MARKER_NAMES]
    rs_labels = ["(rs)" + name for name in MARKER_NAMES]
    fused_labels = ["(fused)" + name for name in MARKER_NAMES]
    visualized_result = camera_results[0]
    visualized_labels = rs_labels

    num_cameras = 1

    if num_cameras == 2:
        visualized_result = fused_result
        visualized_labels = fused_labels

    vis = MarkerVisualizer(
        data_dict1=visualized_result["mocap_matched"],
        data_dict2=visualized_result["fused_points"] if num_cameras == 2 else visualized_result["rs_transformed"],
        labels1=mocap_labels,
        labels2=visualized_labels,
        num_hands=num_hands,
    )
    vis.show()

    histogram_result = fused_result if fused_result is not None else camera_results[0]
    plot_marker_error_histogram(histogram_result["error_stats"])
