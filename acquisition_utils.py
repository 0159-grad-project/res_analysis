import ast
import numpy as np

import config

RS_MARKER_INDICES = [0, 4, 8, 12, 16, 20]
RS_HAND_OFFSET = 21


def load_mocap_log(path, system_delay=-16500):
    """
    Load mocap_log.txt, returns dict: timestamp_ms -> np.array shape (config.N_MARKERS, 3)
    Reorders points to match NAMES in config.py.
    """
    mocap_data = {}
    order = None
    defined_order = False
    with open(path, 'r') as f:
        for line in f:
            entry = ast.literal_eval(line)
            for ts_str, coords in entry.items():
                if ts_str is None or None in coords:
                    continue
                ts = int(ts_str)
                arr = np.array(coords, dtype=float)
                if arr.shape[0] != config.N_MARKERS:
                    continue

                # Reorder to the marker order configured in config.py.
                if not defined_order:
                    order = _get_marker_order_auto(arr)
                    defined_order = True
                arr = arr[order]
                mocap_data[ts + system_delay] = arr

    return mocap_data

def load_realsense_log(path=f'./logs/20250610_realsense_log.txt'):
    """
    Load realsense_log.txt, returns dict: timestamp_ms -> np.array shape (config.N_MARKERS, 3)
    Converts meters -> millimeters and reorders to match NAMES in config.py.
    """
    rs_data = {}
    rs_ordered_indices = _get_rs_ordered_indices()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
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
            if not invalid and len(coords) == config.N_MARKERS:
                rs_data[ts] = np.array(coords)
    return rs_data


def _get_marker_order_auto(arr):
    """
    arr: np.ndarray, shape (config.N_MARKERS, 3), columns = [x, y, z]

    Returns:
        order: list[int]
        such that arr[order] matches NAMES from config.py.
    """
    assert arr.shape[0] == config.N_MARKERS, (
        f"Expected exactly {config.N_MARKERS} markers"
    )

    if config.NUM_HANDS == 1:
        return _get_single_hand_marker_order(arr)
    if config.NUM_HANDS == 2:
        return _get_two_hand_marker_order(arr)

    raise ValueError(f"Unsupported num_hands={config.NUM_HANDS}. Expected 1 or 2.")


def _get_rs_ordered_indices():
    ordered_indices = []

    for hand_idx in range(config.NUM_HANDS):
        hand_offset = hand_idx * RS_HAND_OFFSET
        ordered_indices.extend(hand_offset + idx for idx in RS_MARKER_INDICES)

    return ordered_indices


def _get_single_hand_marker_order(arr):
    wrist_idx = int(np.argmin(arr[:, 0]))

    remaining = [i for i in range(config.N_MARKERS) if i != wrist_idx]
    remaining_sorted = sorted(
        remaining,
        key=lambda i: arr[i, 1],
        reverse=False
    )

    order = [wrist_idx] + remaining_sorted
    order = [int(i) for i in order]
    print(order)

    return order


def _get_two_hand_marker_order(arr):
    x_sorted = np.argsort(arr[:, 0])
    wrist_candidates = x_sorted[:2]
    left_wrist_idx, right_wrist_idx = sorted(
        wrist_candidates, key=lambda i: arr[i, 1], reverse=True
    )

    remaining = [
        i for i in range(config.N_MARKERS)
        if i not in (left_wrist_idx, right_wrist_idx)
    ]
    remaining_sorted = sorted(remaining, key=lambda i: arr[i, 1], reverse=True)

    markers_per_hand = config.N_MARKERS // config.NUM_HANDS
    finger_markers_per_hand = markers_per_hand - 1
    expected_finger_markers = finger_markers_per_hand * config.NUM_HANDS
    if len(remaining_sorted) != expected_finger_markers:
        raise ValueError(
            f"Expected {expected_finger_markers} finger markers after removing wrists."
        )

    left_fingers = remaining_sorted[:finger_markers_per_hand]
    right_fingers = remaining_sorted[finger_markers_per_hand:]

    left_fingers_thumb_to_pinky = list(reversed(left_fingers))
    order = (
        [left_wrist_idx]
        + left_fingers_thumb_to_pinky
        + [right_wrist_idx]
        + right_fingers
    )
    order = [int(i) for i in order]
    print(order)

    return order
