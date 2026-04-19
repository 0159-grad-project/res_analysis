import ast
from pathlib import Path

import numpy as np

from acquisition_utils import load_mocap_log, load_realsense_log
from processing_utils import compute_rigid_transform, interpolate_points_at_timestamp

# ====== Configure here ======
MOCAP_LOG_PATH = Path("./logs/0409_1253_mocap_log.txt")
CAMERA_LOG_PATH = Path("./logs/0409_1253_cam1_realsense_log.txt")

# Set to None to infer automatically from the mocap log.
NUM_HANDS = None

MIN_DELAY_MS = -40000
MAX_DELAY_MS = 40000
COARSE_STEP_MS = 200
INTERP_GAP_MS = 30
CALIBRATION_RATIO = 0.2
MIN_MATCHED_FRAMES = 30


def infer_num_hands_from_mocap(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = ast.literal_eval(line)
            for timestamp, coords in entry.items():
                if timestamp is None or coords is None:
                    continue

                marker_count = len(coords)
                if marker_count == 6:
                    return 1
                if marker_count == 12:
                    return 2
                raise ValueError(
                    f"Unsupported mocap marker count: {marker_count}. "
                    "Expected 6 markers for one hand or 12 markers for two hands."
                )

    raise ValueError("No valid mocap frame found.")


# 从 rs_data 里均匀抽取最多 max_frames 帧
def sample_rs_frames(rs_data, max_frames):
    timestamps = sorted(rs_data.keys())
    if len(timestamps) <= max_frames:
        return {timestamp: rs_data[timestamp] for timestamp in timestamps}

    sampled_indices = np.linspace(0, len(timestamps) - 1, num=max_frames, dtype=int)
    return {
        timestamps[idx]: rs_data[timestamps[idx]]
        for idx in np.unique(sampled_indices)
    }


def evaluate_delay(
    delay_ms,
    mocap_data,
    rs_data,
    *,
    max_gap_ms,
    calibration_ratio,
    min_frames,
):
    mocap_timestamps = sorted(mocap_data.keys())
    matched_frames = []

    for rs_timestamp in sorted(rs_data.keys()):
        mocap_points = interpolate_points_at_timestamp(
            mocap_data,
            rs_timestamp - delay_ms,
            sorted_timestamps=mocap_timestamps,
            max_gap_ms=max_gap_ms,
        )
        if mocap_points is None:
            continue

        matched_frames.append(
            (
                rs_timestamp,
                np.asarray(rs_data[rs_timestamp], dtype=float),
                mocap_points,
            )
        )

    if len(matched_frames) < min_frames:
        return None

    calibration_count = int(len(matched_frames) * calibration_ratio)
    calibration_count = min(max(calibration_count, 1), len(matched_frames) - 1)

    rs_calibration = {
        idx: frame[1]
        for idx, frame in enumerate(matched_frames[:calibration_count])
    }
    mocap_calibration = {
        idx: frame[2]
        for idx, frame in enumerate(matched_frames[:calibration_count])
    }
    rotation, translation = compute_rigid_transform(rs_calibration, mocap_calibration)

    rs_evaluation = np.vstack([frame[1] for frame in matched_frames[calibration_count:]])
    mocap_evaluation = np.vstack([frame[2] for frame in matched_frames[calibration_count:]])
    rs_transformed = (rotation @ rs_evaluation.T).T + translation

    point_errors = np.linalg.norm(rs_transformed - mocap_evaluation, axis=1)
    markers_per_frame = matched_frames[0][1].shape[0]
    frame_errors = point_errors.reshape(-1, markers_per_frame).mean(axis=1)

    return {
        "delay_ms": int(delay_ms),
        "matched_frames": len(matched_frames),
        "evaluation_frames": len(frame_errors),
        "median_frame_error_mm": float(np.median(frame_errors)),
        "mean_frame_error_mm": float(np.mean(frame_errors)),
        "p90_frame_error_mm": float(np.percentile(frame_errors, 90)),
    }


def is_better_candidate(candidate, incumbent):
    if incumbent is None:
        return True

    return (
        candidate["mean_frame_error_mm"],
        candidate["median_frame_error_mm"],
        -candidate["matched_frames"],
        abs(candidate["delay_ms"]),
    ) < (
        incumbent["mean_frame_error_mm"],
        incumbent["median_frame_error_mm"],
        -incumbent["matched_frames"],
        abs(incumbent["delay_ms"]),
    )


def search_best_delay(
    mocap_data,
    rs_data,
    *,
    min_delay,
    max_delay,
    coarse_step,
    max_gap_ms,
    calibration_ratio,
    min_frames,
):
    if coarse_step <= 0:
        raise ValueError("coarse_step must be positive.")

    stage_summaries = []
    rs_sample = sample_rs_frames(rs_data, max_frames=300)

    stages = [
        {
            "name": "coarse",
            "delays": range(min_delay, max_delay + 1, coarse_step),
        }
    ]

    best = None
    for stage in stages:
        for delay_ms in stage["delays"]:
            candidate = evaluate_delay(
                delay_ms,
                mocap_data,
                rs_sample,
                max_gap_ms=max_gap_ms,
                calibration_ratio=calibration_ratio,
                min_frames=min_frames,
            )
            if candidate is None:
                continue
            if is_better_candidate(candidate, best):
                best = candidate

        if best is None:
            raise ValueError("No valid delay candidate found in coarse search range.")

        stage_summaries.append(
            {
                "name": stage["name"],
                "best_delay_ms": best["delay_ms"],
                "median_frame_error_mm": best["median_frame_error_mm"],
                "matched_frames": best["matched_frames"],
            }
        )

    refine_step = max(10, coarse_step // 10)
    refine_window = coarse_step * 2
    refine_best = None
    for delay_ms in range(
        best["delay_ms"] - refine_window,
        best["delay_ms"] + refine_window + 1,
        refine_step,
    ):
        candidate = evaluate_delay(
            delay_ms,
            mocap_data,
            rs_sample,
            max_gap_ms=max_gap_ms,
            calibration_ratio=calibration_ratio,
            min_frames=min_frames,
        )
        if candidate is None:
            continue
        if is_better_candidate(candidate, refine_best):
            refine_best = candidate

    if refine_best is not None:
        best = refine_best
        stage_summaries.append(
            {
                "name": "refine",
                "best_delay_ms": best["delay_ms"],
                "median_frame_error_mm": best["median_frame_error_mm"],
                "matched_frames": best["matched_frames"],
            }
        )

    fine_window = max(20, refine_step * 2)
    fine_best = None
    for delay_ms in range(best["delay_ms"] - fine_window, best["delay_ms"] + fine_window + 1):
        candidate = evaluate_delay(
            delay_ms,
            mocap_data,
            rs_sample,
            max_gap_ms=max_gap_ms,
            calibration_ratio=calibration_ratio,
            min_frames=min_frames,
        )
        if candidate is None:
            continue
        if is_better_candidate(candidate, fine_best):
            fine_best = candidate

    if fine_best is not None:
        best = fine_best
        stage_summaries.append(
            {
                "name": "fine",
                "best_delay_ms": best["delay_ms"],
                "median_frame_error_mm": best["median_frame_error_mm"],
                "matched_frames": best["matched_frames"],
            }
        )

    full_result = evaluate_delay(
        best["delay_ms"],
        mocap_data,
        rs_data,
        max_gap_ms=max_gap_ms,
        calibration_ratio=calibration_ratio,
        min_frames=min_frames,
    )
    if full_result is None:
        raise ValueError("Best delay from sampled search failed during full evaluation.")

    return full_result, stage_summaries


def estimate_system_delay(
    mocap_log_path,
    camera_log_path,
    *,
    num_hands=None,
    min_delay_ms=MIN_DELAY_MS,
    max_delay_ms=MAX_DELAY_MS,
    coarse_step_ms=COARSE_STEP_MS,
    interp_gap_ms=INTERP_GAP_MS,
    calibration_ratio=CALIBRATION_RATIO,
    min_matched_frames=MIN_MATCHED_FRAMES,
):
    if min_delay_ms > max_delay_ms:
        raise ValueError("min_delay_ms must be <= max_delay_ms.")
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be in (0, 1).")

    num_hands = num_hands or infer_num_hands_from_mocap(mocap_log_path)
    mocap_data = load_mocap_log(mocap_log_path, num_hands, system_delay=0)
    camera_data = load_realsense_log(camera_log_path, num_hands)

    best_result, stage_summaries = search_best_delay(
        mocap_data,
        camera_data,
        min_delay=min_delay_ms,
        max_delay=max_delay_ms,
        coarse_step=coarse_step_ms,
        max_gap_ms=interp_gap_ms,
        calibration_ratio=calibration_ratio,
        min_frames=min_matched_frames,
    )
    best_result["num_hands"] = num_hands
    best_result["stage_summaries"] = stage_summaries
    return best_result


def main():
    best_result = estimate_system_delay(
        MOCAP_LOG_PATH,
        CAMERA_LOG_PATH,
        num_hands=NUM_HANDS,
        min_delay_ms=MIN_DELAY_MS,
        max_delay_ms=MAX_DELAY_MS,
        coarse_step_ms=COARSE_STEP_MS,
        interp_gap_ms=INTERP_GAP_MS,
        calibration_ratio=CALIBRATION_RATIO,
        min_matched_frames=MIN_MATCHED_FRAMES,
    )

    print(f"Num hands: {best_result['num_hands']}")
    print(f"Estimated system_delay: {best_result['delay_ms']} ms")
    print(f"Matched frames: {best_result['matched_frames']}")
    print(f"Evaluation frames: {best_result['evaluation_frames']}")
    print(f"Median frame error: {best_result['median_frame_error_mm']:.2f} mm")
    print(f"Mean frame error: {best_result['mean_frame_error_mm']:.2f} mm")
    print(f"P90 frame error: {best_result['p90_frame_error_mm']:.2f} mm")
    # print("Search stages:")
    # for summary in best_result["stage_summaries"]:
    #     print(
    #         f"  {summary['name']}: "
    #         f"delay={summary['best_delay_ms']} ms, "
    #         f"median_error={summary['median_frame_error_mm']:.2f} mm, "
    #         f"matched_frames={summary['matched_frames']}"
    #     )


if __name__ == "__main__":
    main()
