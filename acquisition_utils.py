import numpy as np
import ast

from config import *

def load_mocap_log(path, system_delay=-16500):
    """
    Load mocap_log.txt, returns dict: timestamp_ms -> np.array shape (6,3)
    Reorders points to [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
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
                if arr.shape[0] != N_MARKERS:
                    continue

                # Reorder to [wrist, thumb, index, middle, ring, pinky]
                if not defined_order:
                    order = _get_marker_order_auto(arr)
                    defined_order = True
                arr = arr[order]
                mocap_data[ts + system_delay] = arr

    return mocap_data

def load_realsense_log(path = f'./logs/20250610_realsense_log.txt'):
    """
    Load realsense_log.txt, returns dict: timestamp_ms -> np.array shape (6,3)
    Converts meters -> millimeters
    """
    rs_data = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            ts = int(parts[0])
            coords = []
            for j in [0, 4, 8, 12, 16, 20]:
                base = 1 + j * 6
                try:
                    X = float(parts[base + 3])
                    Y = float(parts[base + 4])
                    Z = float(parts[base + 5])
                except (IndexError, ValueError):
                    break
                # skip invalid zeros
                if Z == 0.0:
                    continue
                coords.append([X * 1000, Y * 1000, Z * 1000]) # m to mm
            if len(coords) == N_MARKERS:
                rs_data[ts] = np.array(coords)
    return rs_data


def _get_marker_order_auto(arr):
    """
    arr: np.ndarray, shape (6, 3), columns = [x, y, z]

    Returns:
        order: list[int]
        such that arr[order] =
        [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    """
    assert arr.shape[0] == N_MARKERS, f"Expected exactly {N_MARKERS} markers"

    # wrist = point with minimum x
    wrist_idx = int(np.argmin(arr[:, 0]))

    # sort remaining by y descending
    remaining = [i for i in range(6) if i != wrist_idx]
    remaining_sorted = sorted(
        remaining,
        key=lambda i: arr[i, 1],
        reverse=False
    )

    order = [wrist_idx] + remaining_sorted
    print(order)

    return order
