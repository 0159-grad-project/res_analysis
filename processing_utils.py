from bisect import bisect_left

import numpy as np
from sklearn.cluster import DBSCAN


def find_nearest_timestamp(ts_list, target):
    """
    Find nearest timestamp in sorted ts_list to target.

    Returns:
        t: the closest time in ts_list to target
    """
    pos = bisect_left(ts_list, target)
    if pos == 0:
        return ts_list[0]
    if pos == len(ts_list):
        return ts_list[-1]
    before = ts_list[pos - 1]
    after = ts_list[pos]
    return before if abs(before - target) <= abs(after - target) else after


def find_matching_frames(mocap_data, rs_data, threshold=30):
    """
    Match mocap and realsense timestamps.

    Returns:
        matched_pairs: dict of {mocap_ts: rs_ts}
    """
    mocap_ts = sorted(mocap_data.keys())
    rs_ts = sorted(rs_data.keys())
    matched_pairs = {}

    for rs_t in rs_ts:
        mocap_t = find_nearest_timestamp(mocap_ts, rs_t)
        if abs(mocap_t - rs_t) <= threshold:
            matched_pairs[mocap_t] = rs_t
    return matched_pairs


def filter_matching_data(mocap_data, rs_data, matched_pairs):
    """
    Filter matching data from mocap and rs.

    Returns:
        mocap_matched, rs_matched: matched data (use mocap's timestamp)
    """
    mocap_matched, rs_matched = {}, {}

    for mocap_t, rs_t in matched_pairs.items():
        m_pts = mocap_data[mocap_t]
        r_pts = rs_data[rs_t]
        if m_pts is None or r_pts is None:
            continue
        mocap_matched[mocap_t] = m_pts
        rs_matched[mocap_t] = r_pts  # Note: match to mocap's timestamp

    assert len(mocap_matched) == len(rs_matched)

    return mocap_matched, rs_matched


def get_common_timestamps(*data_dicts):
    """返回所有输入字典共享的时间戳交集，并按时间排序。"""
    if not data_dicts:
        return []

    common_timestamps = set(data_dicts[0].keys())
    for data_dict in data_dicts[1:]:
        common_timestamps &= set(data_dict.keys())

    return sorted(common_timestamps)


def filter_data_by_timestamps(data_dict, timestamps):
    """仅保留指定时间戳的数据，并统一转成 float 数组。"""
    return {
        timestamp: np.asarray(data_dict[timestamp], dtype=float)
        for timestamp in timestamps
        if timestamp in data_dict
    }


def split_timestamps_by_ratio(timestamps, calibration_ratio=None):
    """
    将按时间排序的数据切分为前段标定集和后段评估集。

    当 calibration_ratio 为 None 时，表示沿用旧模式：
    全部数据同时用于坐标变换和误差评估。
    """
    ordered_timestamps = sorted(timestamps)
    if not ordered_timestamps:
        return [], []

    if calibration_ratio is None:
        return ordered_timestamps, ordered_timestamps

    if not 0 < calibration_ratio < 1:
        raise ValueError(
            f"Unsupported calibration_ratio={calibration_ratio}. "
            "Expected None or a float in (0, 1)."
        )
    if len(ordered_timestamps) < 2:
        raise ValueError(
            "At least two timestamps are required when splitting calibration and evaluation data."
        )

    calibration_count = int(len(ordered_timestamps) * calibration_ratio)
    calibration_count = min(max(calibration_count, 1), len(ordered_timestamps) - 1)

    return (
        ordered_timestamps[:calibration_count],
        ordered_timestamps[calibration_count:],
    )


def interpolate_points_at_timestamp(data_dict, target_timestamp, *, sorted_timestamps=None, max_gap_ms=100):
    """
    在任意目标时间戳上，对 3D marker 位置做线性插值。

    只有当目标时间戳两侧的 mocap 帧间隔足够小，才允许插值；
    否则认为该时刻缺少可靠参考值。
    """
    if not data_dict:
        return None

    timestamps = sorted_timestamps if sorted_timestamps is not None else sorted(data_dict.keys())
    pos = bisect_left(timestamps, target_timestamp)

    if pos < len(timestamps) and timestamps[pos] == target_timestamp:
        return np.asarray(data_dict[target_timestamp], dtype=float)
    if pos == 0 or pos == len(timestamps):
        return None

    left_ts = timestamps[pos - 1]
    right_ts = timestamps[pos]
    gap = right_ts - left_ts
    if gap <= 0 or gap > max_gap_ms:
        return None

    left_pts = np.asarray(data_dict[left_ts], dtype=float)
    right_pts = np.asarray(data_dict[right_ts], dtype=float)
    alpha = (target_timestamp - left_ts) / gap
    return (1.0 - alpha) * left_pts + alpha * right_pts


def build_interpolated_reference(data_dict, target_timestamps, *, max_gap_ms=100):
    """对每个目标时间戳做插值，构造时间对齐后的参考数据字典。"""
    sorted_timestamps = sorted(data_dict.keys())
    interpolated = {}

    for timestamp in target_timestamps:
        points = interpolate_points_at_timestamp(
            data_dict,
            timestamp,
            sorted_timestamps=sorted_timestamps,
            max_gap_ms=max_gap_ms,
        )
        if points is not None:
            interpolated[timestamp] = points

    return interpolated


def evaluate_predictions(reference_dict, predicted_dict, marker_names, *, print_summary=True):
    """在共享时间戳上，将预测结果与参考数据进行误差评估。"""
    common_timestamps = get_common_timestamps(reference_dict, predicted_dict)
    if not common_timestamps:
        raise ValueError("No common timestamps available for evaluation.")

    reference = filter_data_by_timestamps(reference_dict, common_timestamps)
    predicted = filter_data_by_timestamps(predicted_dict, common_timestamps)

    reference_vec = np.vstack(list(reference.values()))
    predicted_vec = np.vstack(list(predicted.values()))

    error_stats = compute_detailed_errors(
        reference_vec,
        predicted_vec,
        marker_names,
        print_summary=print_summary,
    )
    errors = np.linalg.norm(predicted_vec - reference_vec, axis=1)

    return {
        "timestamps": common_timestamps,
        "reference": reference,
        "predicted": predicted,
        "error_stats": error_stats,
        "errors": errors,
    }


def compute_rigid_transform(A_dict, B_dict):
    """
    Computer the rigid transformation matrices.

    Returns:
        R, t such that R @ A.T + t[:,None] ~= B.T
    """
    assert len(A_dict) == len(B_dict)
    A = np.vstack(np.array(list(A_dict.values())))
    B = np.vstack(np.array(list(B_dict.values())))
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    t = centroid_B - R_mat @ centroid_A
    return R_mat, t


def apply_rigid_transform(A_dict, R, t):
    """
    Apply rigid-body transform (R, t) to each coordinate array in A_dict.

    Args:
        A_dict: dict[timestamp] -> array-like of shape (n_markers, 3)
        R: (3x3) rotation matrix
        t: (3,) translation vector

    Returns:
        dict[timestamp] -> np.ndarray of shape (n_markers, 3)
    """
    transformed = {}
    for ts, pts in A_dict.items():
        # ensure pts is an (n_markers, 3) array
        arr = np.asarray(pts)                     # shape (n_markers, 3)
        arr_trans = (R @ arr.T).T + t             # apply R then t
        transformed[ts] = arr_trans               # store back
    return transformed


def compute_rigid_transforms_per_marker(A_dict, B_dict):
    """
    Compute one rigid transform per marker.

    Args:
        A_dict: dict[timestamp] -> np.ndarray shape (n_markers, 3)
                e.g. your rs_matched
        B_dict: dict[timestamp] -> np.ndarray shape (n_markers, 3)
                e.g. your mocap_matched

    Returns:
        transforms: dict[marker_idx] -> (R, t)
            where R is (3x3) and t is (3,) mapping A_dict→B_dict
    """
    # ensure same frames
    keys = sorted(A_dict.keys())
    assert set(keys) == set(B_dict.keys()), "Mismatch in timestamps"
    if not keys:
        raise ValueError("At least one timestamp is required to compute per-marker transforms.")

    n_markers = np.asarray(A_dict[keys[0]], dtype=float).shape[0]
    transforms = {}
    for i in range(n_markers):
        # stack marker i across time
        A_i = np.vstack([A_dict[t][i] for t in keys])   # shape (N,3)
        B_i = np.vstack([B_dict[t][i] for t in keys])   # shape (N,3)

        # compute Kabsch
        centroid_A = A_i.mean(axis=0)
        centroid_B = B_i.mean(axis=0)
        AA = A_i - centroid_A
        BB = B_i - centroid_B
        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_B - R @ centroid_A

        transforms[i] = (R, t)
    return transforms


def apply_rigid_transforms_per_marker(A_dict, transforms):
    """
    Apply a per-marker rigid transform to each frame in A_dict.

    Returns:
        dict[timestamp] -> np.ndarray shape (n_markers, 3)
    """
    out = {}
    for ts, pts in A_dict.items():
        pts = np.asarray(pts)               # (n_markers,3)
        transformed = np.empty_like(pts)
        for i, (R, t) in transforms.items():
            transformed[i] = (R @ pts[i]) + t
        out[ts] = transformed
    return out


def compute_detailed_errors(mocap_vec, rs_vec, marker_names, print_summary=True):
    """
    Compute rigid alignment and return detailed error breakdown per marker.

    Returns:
        error_summary: dict[label] -> {mean, median, std, max, all_errors}
    """
    n_markers = len(marker_names)
    errors = np.linalg.norm(rs_vec - mocap_vec, axis=1)  # shape (N * n_markers,)
    error_summary = {}

    for i, marker_name in enumerate(marker_names):
        marker_errors = errors[i::n_markers]
        error_summary[marker_name] = {
            "mean": marker_errors.mean(),
            "median": np.median(marker_errors),
            "std": marker_errors.std(),
            "max": marker_errors.max(),
            "all": marker_errors,
        }

    if print_summary:
        print("=== Per-Marker Error Summary ===")
        label_width = max(len(marker_name) for marker_name in error_summary)
        for marker_name, stats in error_summary.items():
            print(f"{marker_name:<{label_width}}: mean={stats['mean']:.2f} mm | median={stats['median']:.2f} mm")

        print("=== Overall Error Summary ===")
        print(f"Mean error: {errors.mean():.2f} mm")
        print(f"Median error: {np.median(errors):.2f} mm")
        print(f"Std  error: {errors.std():.2f} mm")

    return error_summary


def detect_marker_anomalies(data_dict, *, eps=5, min_samples=5, metric="euclidean"):
    timestamps = sorted(data_dict.keys())
    if not timestamps:
        return {}, 0

    n_markers = np.asarray(data_dict[timestamps[0]], dtype=float).shape[0]
    trajectories = {
        i: np.vstack([data_dict[t][i] for t in timestamps])
        for i in range(n_markers)
    }

    anomalies = {}
    num = 0

    for i, traj in trajectories.items():
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clustering.fit_predict(traj)
        # label == -1 → noise → anomaly
        anomalies[i] = [timestamps[j] for j, label in enumerate(labels) if label == -1]
        num += len(anomalies[i])

    return anomalies, num
