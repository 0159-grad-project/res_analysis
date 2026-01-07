import numpy as np

from config import *
from visualizer import *
from acquisition_utils import *
from processing_utils import *


# Load data
date = '0106'
time = '1513'
mocap = load_mocap_log(f'./logs/{date}_{time}_mocap_log.txt', system_delay=-16330)
rs = load_realsense_log(f'./logs/{date}_{time}_realsense_log.txt')
print(f"Total mocap frames: {len(mocap)}")
print(f"Total rs frames: {len(rs)}")

# Remove anomalies in rs
# 对每一个关键点分别聚类后，删除在eps半径内，邻居数小于min_samples的点
rs_anomalies, n = detect_marker_anomalies(rs, eps=25, min_samples=20)
rs_anomalies_times = set(time for anomaly in rs_anomalies.values() for time in anomaly)
print(f"Detected {n} anomalous markers and {len(rs_anomalies_times)} anomalous time in Realsense data.")
if len(rs_anomalies_times) > 0:
    for t in rs_anomalies_times:
            del rs[t]

# Filter matched frames
matched_pairs = find_matching_frames(mocap, rs, threshold=8)
print(f"Matched frame count: {len(matched_pairs)}")
mocap_matched, rs_matched = filter_matching_data(mocap, rs, matched_pairs)


# -------- analyze together --------
# print("\nAnalyzing combined data...")
# Compute & apply rigid tranform rs -> mocap
# One transformation matrix for all coordiantes:
R, t = compute_rigid_transform(rs_matched, mocap_matched)
rs_transformed = apply_rigid_transform(rs_matched, R, t)

# Compute error-stats:
mocap_vec = np.vstack(list(mocap_matched.values()))
rs_vec    = np.vstack(list(rs_transformed.values()))
# error_stats = compute_detailed_errors(mocap_vec, rs_vec)
# plot_marker_error_histogram(error_stats)


# -------- analyze per marker --------
print("\nAnalyzing per-marker data...")
transformations = compute_rigid_transforms_per_marker(rs_matched, mocap_matched)
rs_transformed_per_marker = apply_rigid_transforms_per_marker(rs_matched, transformations)

# Compute error-stats:
rs_vec = np.vstack(list(rs_transformed_per_marker.values()))
error_stats = compute_detailed_errors(mocap_vec, rs_vec)
# plot_marker_error_histogram(error_stats)


rs_labels = ["(rs)" + n for n in NAMES]
mocap_labels = ["(mc)" + n for n in NAMES]

vis = MarkerVisualizer(
    data_dict1=mocap_matched,
    data_dict2=rs_transformed_per_marker, # rs_transformed_per_marker,
    labels1=mocap_labels,
    labels2=rs_labels)
vis.show()