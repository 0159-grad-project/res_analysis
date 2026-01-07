import numpy as np
from bisect import bisect_left
from sklearn.cluster import DBSCAN

from config import *

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
    Filter matching data from mocap and rs

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
        rs_matched[mocap_t] = r_pts # Note: match to mocap's timestamp

    assert len(mocap_matched) == len(rs_matched)

    return mocap_matched, rs_matched


def compute_rigid_transform(A_dict, B_dict):
    """
    Computer the rigid transformation matrices
    
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
    Apply rigid-body transform (R, t) to each 6x3 coordinate array in A_dict.

    Args:
        A_dict: dict[timestamp] -> array-like of shape (n_markers, 3)
        R:   (3x3) rotation matrix
        t:   (3,) translation vector

    Returns:
        dict[timestamp] -> np.ndarray of shape (n_markers, 3)
    """
    transformed = {}
    for ts, pts in A_dict.items():
        # ensure pts is an (n_markers, 3) array
        arr = np.asarray(pts)                     # shape (6,3)
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
    
    transforms = {}
    for i in range(N_MARKERS):
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
            Vt[-1,:] *= -1
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
    
def compute_detailed_errors(mocap_vec, rs_vec, print_summary=True):
    """
    Compute rigid alignment and return detailed error breakdown per marker.

    Returns:
        error_summary: dict[label] -> {mean, std, max, all_errors}
    """
    errors = np.linalg.norm(rs_vec - mocap_vec, axis=1)  # shape (N * n_markers,)
    error_summary = {}

    for i in range(N_MARKERS):  # marker index 0 to n_markers
        marker_errors = errors[i::N_MARKERS]
        error_summary[f"{NAMES[i]}"] = {
            "mean": marker_errors.mean(),
            "std": marker_errors.std(),
            "max": marker_errors.max(),
            "all": marker_errors
        }

    if print_summary:
        #print("=== Per-Marker Error Summary ===")
        for k, v in error_summary.items():
            print(f"{k}: mean={v['mean']:.2f} mm | std={v['std']:.2f} mm | max={v['max']:.2f} mm")

        overall = errors
        print("=== Overall Error Summary ===")
        print(f"Mean error: {overall.mean():.2f} mm")
        print(f"Std  error: {overall.std():.2f} mm")
        print(f"Max  error: {overall.max():.2f} mm")

    return error_summary

def detect_marker_anomalies(data_dict, *,
                            eps=5,
                            min_samples=5,
                            metric='euclidean'):
    timestamps = sorted(data_dict.keys())
    trajectories = {i: np.vstack([data_dict[t][i] for t in timestamps])
                    for i in range(N_MARKERS)}

    anomalies = {}
    num = 0

    for i, traj in trajectories.items():
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clustering.fit_predict(traj)
        # label == -1 → noise → anomaly
        anomalies[i] = [timestamps[j] for j, label in enumerate(labels) if label == -1]
        num += len(anomalies[i])

    return anomalies, num
