"""用于双相机时间配对和融合的工具函数。"""

import numpy as np

from processing_utils import evaluate_predictions, interpolate_points_at_timestamp


def pair_timestamps_one_to_one(timestamps_a, timestamps_b, *, threshold_ms=30):
    """
    在时间阈值内，贪心构造双相机的一对一时间配对。

    这里加入了局部向前看一步的策略，避免过早锁定到较差配对，
    尽量让当前帧和下一帧形成更紧的时间匹配。
    """
    pairs = []
    i = 0
    j = 0

    while i < len(timestamps_a) and j < len(timestamps_b):
        ts_a = timestamps_a[i]
        ts_b = timestamps_b[j]
        diff = ts_a - ts_b

        if abs(diff) <= threshold_ms:
            next_a_better = (
                i + 1 < len(timestamps_a)
                and abs(timestamps_a[i + 1] - ts_b) < abs(diff)
            )
            next_b_better = (
                j + 1 < len(timestamps_b)
                and abs(ts_a - timestamps_b[j + 1]) < abs(diff)
            )

            if next_a_better and (not next_b_better or abs(timestamps_a[i + 1] - ts_b) <= abs(ts_a - timestamps_b[j + 1])):
                i += 1
                continue
            if next_b_better:
                j += 1
                continue

            pairs.append((ts_a, ts_b))
            i += 1
            j += 1
            continue

        if diff < 0:
            i += 1
        else:
            j += 1

    return pairs


def _compute_marker_rms_errors(camera_result, marker_names):
    """将单相机下每个 marker 的误差汇总成 RMS，用于后续加权。"""
    rms_errors = []
    error_source = camera_result.get("weight_error_stats", camera_result["error_stats"])

    for marker_name in marker_names:
        marker_errors = np.asarray(error_source[marker_name]["all"], dtype=float)
        rms = np.sqrt(np.mean(marker_errors ** 2))
        rms_errors.append(rms)

    return np.asarray(rms_errors, dtype=float)


def _compute_camera_marker_weights(camera_results, marker_names, *, epsilon=1e-6):
    """将每个 marker 的 RMS 误差转换成类似逆方差形式的融合权重。"""
    weights = []

    for camera_result in camera_results:
        rms_errors = _compute_marker_rms_errors(camera_result, marker_names)
        weights.append(1.0 / np.maximum(rms_errors ** 2, epsilon))

    return weights


def _compute_disagreement_thresholds(
    camera_results,
    marker_names,
    *,
    gate_scale=2.5,
    min_threshold_mm=15.0,
):
    """为每个 marker 设置分歧阈值，用来判断两台相机是否偏差过大。"""
    rms_stack = np.vstack([
        _compute_marker_rms_errors(result, marker_names)
        for result in camera_results
    ])
    return np.maximum(min_threshold_mm, gate_scale * rms_stack.max(axis=0))


def _fuse_weighted_points(camera_points, marker_weights, disagreement_thresholds):
    """
    对单帧多相机点进行 marker 级加权融合。

    如果两台相机在某个 marker 上的分歧远大于正常范围，
    就不再做平均，而是直接退回到历史上更可靠的那台相机。
    """
    stacked_points = np.stack([np.asarray(points, dtype=float) for points in camera_points], axis=0)
    weight_matrix = np.stack(marker_weights, axis=0)

    # 对当前帧的每个 marker 做加权平均。
    weighted_sum = (stacked_points * weight_matrix[:, :, None]).sum(axis=0)
    weight_total = weight_matrix.sum(axis=0)[:, None]
    fused_points = weighted_sum / weight_total

    if stacked_points.shape[0] == 2:
        disagreement = np.linalg.norm(stacked_points[0] - stacked_points[1], axis=1)
        best_camera_idx = np.argmax(weight_matrix, axis=0)

        # 如果两台相机在某个 marker 上分歧过大，则直接信任更可靠的一台。
        for marker_idx, marker_gap in enumerate(disagreement):
            if marker_gap > disagreement_thresholds[marker_idx]:
                fused_points[marker_idx] = stacked_points[best_camera_idx[marker_idx], marker_idx]

    return fused_points


def analyze_weighted_fusion(
    camera_results,
    mocap_data,
    marker_names,
    *,
    pair_threshold_ms=30,
    mocap_interp_max_gap_ms=100,
):
    """
    对双相机流做时间配对，在融合时刻上插值 mocap，并计算融合误差。

    返回融合后的轨迹，以及主程序后续展示所需的误差统计结果。
    """
    if len(camera_results) != 2:
        raise ValueError("Weighted fusion currently expects exactly two camera streams.")

    camera_a = camera_results[0]
    camera_b = camera_results[1]

    paired_timestamps = pair_timestamps_one_to_one(
        sorted(camera_a["rs_transformed_for_fusion"].keys()),
        sorted(camera_b["rs_transformed_for_fusion"].keys()),
        threshold_ms=pair_threshold_ms,
    )
    if not paired_timestamps:
        raise ValueError("No paired camera frames found for fusion.")

    marker_weights = _compute_camera_marker_weights(camera_results, marker_names)
    disagreement_thresholds = _compute_disagreement_thresholds(camera_results, marker_names)

    mocap_reference = {}
    fused_points = {}
    camera_predictions = [{}, {}]
    camera_gaps = []

    mocap_timestamps = sorted(mocap_data.keys())

    for ts_a, ts_b in paired_timestamps:
        # 以双相机时间中点作为融合时刻，并在该时刻插值 mocap。
        fusion_ts = int(round((ts_a + ts_b) / 2.0))
        mocap_points = interpolate_points_at_timestamp(
            mocap_data,
            fusion_ts,
            sorted_timestamps=mocap_timestamps,
            max_gap_ms=mocap_interp_max_gap_ms,
        )
        if mocap_points is None:
            continue

        points_a = np.asarray(camera_a["rs_transformed_for_fusion"][ts_a], dtype=float)
        points_b = np.asarray(camera_b["rs_transformed_for_fusion"][ts_b], dtype=float)
        fused = _fuse_weighted_points(
            [points_a, points_b],
            marker_weights,
            disagreement_thresholds,
        )

        mocap_reference[fusion_ts] = mocap_points
        fused_points[fusion_ts] = fused
        camera_predictions[0][fusion_ts] = points_a
        camera_predictions[1][fusion_ts] = points_b
        camera_gaps.append(abs(ts_a - ts_b))

    if not fused_points:
        raise ValueError("No fused frames remained after mocap interpolation.")

    paired_camera_results = []
    for camera_result, prediction_dict in zip(camera_results, camera_predictions):
        paired_eval = evaluate_predictions(
            mocap_reference,
            prediction_dict,
            marker_names,
            print_summary=False,
        )
        paired_eval["camera_label"] = camera_result["camera_label"]
        paired_camera_results.append(paired_eval)

    print("\n=== fusion vs mocap ===")
    print(f"Fusion frame count: {len(fused_points)}")
    fused_eval = evaluate_predictions(
        mocap_reference,
        fused_points,
        marker_names,
        print_summary=True,
    )

    return {
        "camera_label": "weighted_fusion",
        "mocap_matched": mocap_reference,
        "fused_points": fused_points,
        "error_stats": fused_eval["error_stats"],
        "errors": fused_eval["errors"],
        "paired_camera_results": paired_camera_results,
        "paired_timestamps": paired_timestamps,
        "camera_time_gaps_ms": np.asarray(camera_gaps, dtype=float),
    }
