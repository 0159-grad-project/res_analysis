import ast

import numpy as np

RS_MARKER_INDICES = [0, 4, 8, 12, 16, 20]
RS_HAND_OFFSET = 21


def load_mocap_log(path, num_hands, system_delay):
    """
    Load mocap log data as {timestamp_ms: np.ndarray[n_markers, 3]}.

    Points are reordered once based on the first valid frame so they match
    the marker layout configured in config.py.
    """
    mocap_data = {}
    marker_order = None
    expected_markers = 6 * num_hands

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = ast.literal_eval(line)
            for timestamp, coords in entry.items():
                if timestamp is None or coords is None:
                    continue
                if any(point is None or None in point for point in coords):
                    continue

                points = np.asarray(coords, dtype=float)
                if points.shape != (expected_markers, 3):
                    continue

                if marker_order is None:
                    marker_order = get_mocap_marker_order(points, num_hands)
                    # print(f"Inferred mocap marker order: {marker_order}")

                mocap_data[int(timestamp) + system_delay] = points[marker_order]

    if not mocap_data:
        raise ValueError(f"No valid mocap frames loaded from {path}.")

    return mocap_data

def load_realsense_log(path, num_hands):
    """
    Load realsense_log.txt, returns dict: timestamp_ms -> np.array shape (n_markers, 3).
    Converts meters -> millimeters and keeps the original selected-landmark logic.
    """
    rs_data = {}
    expected_markers = 6 * num_hands
    rs_ordered_indices = get_rs_ordered_indices(num_hands)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            ts = int(parts[0])
            coords = []
            invalid = False
            for j in rs_ordered_indices:
                base = 1 + j * 6
                try:
                    X = float(parts[base + 3])
                    Y = float(parts[base + 4])
                    Z = float(parts[base + 5])
                except (IndexError, ValueError):
                    invalid = True
                    break
                if np.isnan(X) or np.isnan(Y) or np.isnan(Z) or Z == 0.0:
                    invalid = True
                    break
                coords.append([X * 1000, Y * 1000, Z * 1000])
            if not invalid and len(coords) == expected_markers:
                rs_data[ts] = np.array(coords)
    return rs_data


def get_mocap_marker_order(points, num_hands):
    """
    Infer the mocap marker order from one valid frame.

    Returns a list of indices that reorders the raw mocap points into the
    marker order configured in config.py.
    """
    if num_hands == 1:
        wrist_idx = int(np.argmin(points[:, 0]))
        remaining = [i for i in range(points.shape[0]) if i != wrist_idx]
        remaining_sorted = sorted(remaining, key=lambda i: points[i, 1])
        return [wrist_idx] + [int(i) for i in remaining_sorted]

    if num_hands == 2:
        x_sorted = np.argsort(points[:, 0])
        wrist_candidates = x_sorted[:2]
        left_wrist_idx, right_wrist_idx = sorted(
            wrist_candidates,
            key=lambda i: points[i, 1],
            reverse=True,
        )

        remaining = [
            i
            for i in range(points.shape[0])
            if i not in (left_wrist_idx, right_wrist_idx)
        ]
        remaining_sorted = sorted(remaining, key=lambda i: points[i, 1], reverse=True)
        left_fingers = remaining_sorted[:5]
        right_fingers = remaining_sorted[5:]

        return (
            [int(left_wrist_idx)]
            + [int(i) for i in reversed(left_fingers)]
            + [int(right_wrist_idx)]
            + [int(i) for i in right_fingers]
        )

    raise ValueError(f"Unsupported num_hands={num_hands}. Expected 1 or 2.")


def get_rs_ordered_indices(num_hands):
    ordered_indices = []
    for hand_idx in range(num_hands):
        hand_offset = hand_idx * RS_HAND_OFFSET
        ordered_indices.extend(hand_offset + idx for idx in RS_MARKER_INDICES)
    return ordered_indices
