import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

PRIMARY_HAND_COLORS = ["red", "green", "blue", "orange", "purple", "cyan"]
SECONDARY_HAND_COLORS = [
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:gray",
    "tab:blue",
    "tab:orange",
]


def plot_marker_error_histogram(error_summary):
    for marker, stats in error_summary.items():
        plt.hist(stats["all"], bins=30, alpha=0.6, label=marker)
    plt.xlabel("Euclidean error (mm)")
    plt.ylabel("Frequency")
    plt.title("Per-marker error distribution")
    plt.legend()
    plt.show()


class MarkerVisualizer:
    def __init__(
        self,
        data_dict1,
        data_dict2=None,
        labels1=None,
        labels2=None,
        *,
        num_hands,
    ):
        self.data1 = data_dict1
        self.data2 = data_dict2
        self.num_hands = num_hands
        self.n_markers = _infer_marker_count(data_dict1)
        self.labels1 = labels1 if labels1 else [f"Marker {i}" for i in range(self.n_markers)]
        self.labels2 = labels2 if labels2 else [f"Marker {i}" for i in range(self.n_markers)]

        self.colors = _build_colors(num_hands)
        self.marker_style1 = "o"
        self.marker_style2 = "*" if data_dict2 else None
        self.link_pairs = _build_link_pairs(num_hands, self.n_markers)

        self.timestamps = sorted(data_dict1.keys())
        self.timestamp_to_index = {t: i for i, t in enumerate(self.timestamps)}

        # Visibility flags
        self.show_mocap = True
        self.show_rs = True if self.data2 else False

        self._setup_plot()

    def _setup_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.25 if self.data2 else 0.2)

        ax_jump = plt.axes([0.45, 0.11, 0.2, 0.04])
        self.jump_box = TextBox(ax_jump, "Go to Timestamp", initial=str(self.timestamps[0]))
        self.jump_box.on_submit(self._jump_to_timestamp)

        # Initialize markers and lines
        self.markers1, self.lines1 = self._init_markers(self.data1, self.labels1, self.marker_style1)
        if self.data2:
            self.markers2, self.lines2 = self._init_markers(self.data2, self.labels2, self.marker_style2)

        # Compute axis limits
        all_points = []
        for data in (self.data1, self.data2) if self.data2 else (self.data1,):
            for pts in data.values():
                all_points.append(pts)
        all_points = np.vstack(all_points)
        self.ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
        self.ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
        self.ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.set_title(f"Timestamp: {self.timestamps[0]} ms")

        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.2, 1))

        # Slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.04])
        self.slider = Slider(
            ax_slider,
            "t (ms)",
            valmin=self.timestamps[0],
            valmax=self.timestamps[-1],
            valinit=self.timestamps[0],
            valstep=self.timestamps,
        )
        self.slider.on_changed(self._update)

        # Prev/Next buttons
        ax_prev = plt.axes([0.25, 0.1, 0.05, 0.03])
        ax_next = plt.axes([0.7, 0.1, 0.05, 0.03])
        self.btn_prev = Button(ax_prev, "<")
        self.btn_next = Button(ax_next, ">")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)

        # Check buttons only if both datasets exist
        if self.data2:
            ax_check = plt.axes([0.7, 0.15, 0.15, 0.10])
            self.check = CheckButtons(ax_check, ["Show Mocap", "Show Realsense"], [True, True])
            self.check.on_clicked(self._toggle_visibility)

    def _init_markers(self, data, labels, marker_style):
        markers = []
        lines = []
        ts0 = self.timestamps[0]
        points = data.get(ts0, np.full((self.n_markers, 3), np.nan))
        for i in range(self.n_markers):
            pt = points[i]
            scatter = self.ax.scatter(
                pt[0],
                pt[1],
                pt[2],
                color=self.colors[i],
                marker=marker_style,
                label=f"{labels[i]} ({marker_style})",
            )
            markers.append(scatter)
        for wrist_idx, finger_idx in self.link_pairs:
            line, = self.ax.plot(
                [points[wrist_idx, 0], points[finger_idx, 0]],
                [points[wrist_idx, 1], points[finger_idx, 1]],
                [points[wrist_idx, 2], points[finger_idx, 2]],
                color=self.colors[finger_idx],
                linestyle="--",
                linewidth=1.2,
            )
            lines.append(line)
        return markers, lines

    def _update(self, val):
        timestamp = int(val)
        pts1 = self.data1.get(timestamp, np.full((self.n_markers, 3), np.nan))

        for i in range(self.n_markers):
            self.markers1[i]._offsets3d = ([pts1[i, 0]], [pts1[i, 1]], [pts1[i, 2]])
            self.markers1[i].set_visible(self.show_mocap)
        for line, (wrist_idx, finger_idx) in zip(self.lines1, self.link_pairs):
            line.set_data([pts1[wrist_idx, 0], pts1[finger_idx, 0]], [pts1[wrist_idx, 1], pts1[finger_idx, 1]])
            line.set_3d_properties([pts1[wrist_idx, 2], pts1[finger_idx, 2]])
            line.set_visible(self.show_mocap)

        if self.data2:
            pts2 = self.data2.get(timestamp, np.full((self.n_markers, 3), np.nan))
            for i in range(self.n_markers):
                self.markers2[i]._offsets3d = ([pts2[i, 0]], [pts2[i, 1]], [pts2[i, 2]])
                self.markers2[i].set_visible(self.show_rs)
            for line, (wrist_idx, finger_idx) in zip(self.lines2, self.link_pairs):
                line.set_data([pts2[wrist_idx, 0], pts2[finger_idx, 0]], [pts2[wrist_idx, 1], pts2[finger_idx, 1]])
                line.set_3d_properties([pts2[wrist_idx, 2], pts2[finger_idx, 2]])
                line.set_visible(self.show_rs)

        self.ax.set_title(f"Timestamp: {timestamp} ms")
        self.fig.canvas.draw_idle()

    def _toggle_visibility(self, label):
        if label == "Show Mocap":
            self.show_mocap = not self.show_mocap
            for marker in self.markers1:
                marker.set_visible(self.show_mocap)
            for line in self.lines1:
                line.set_visible(self.show_mocap)
        elif label == "Show Realsense" and self.data2:
            self.show_rs = not self.show_rs
            for marker in self.markers2:
                marker.set_visible(self.show_rs)
            for line in self.lines2:
                line.set_visible(self.show_rs)
        self.fig.canvas.draw_idle()

    def _jump_to_timestamp(self, text):
        try:
            timestamp = int(text)
            if timestamp in self.timestamps:
                self.slider.set_val(timestamp)
            else:
                closest = min(self.timestamps, key=lambda t: abs(t - timestamp))
                print(f"Timestamp not found. Jumping to closest: {closest}")
                self.slider.set_val(closest)
        except ValueError:
            print("Invalid timestamp input.")

    def _get_current_index(self):
        current = int(round(self.slider.val))
        idx = self.timestamp_to_index.get(current)
        if idx is None:
            idx = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - current))
        return idx

    def _step_frame(self, direction):
        if not self.timestamps:
            return
        idx = self._get_current_index()
        next_idx = min(max(idx + direction, 0), len(self.timestamps) - 1)
        if next_idx != idx:
            self.slider.set_val(self.timestamps[next_idx])

    def _on_prev(self, _event):
        self._step_frame(-1)

    def _on_next(self, _event):
        self._step_frame(1)

    def show(self):
        plt.show()


def _build_colors(num_hands):
    if num_hands == 1:
        return PRIMARY_HAND_COLORS.copy()
    if num_hands == 2:
        return PRIMARY_HAND_COLORS + SECONDARY_HAND_COLORS

    raise ValueError(f"Unsupported num_hands={num_hands}. Expected 1 or 2.")


def _build_link_pairs(num_hands, n_markers):
    markers_per_hand = n_markers // num_hands
    return [
        (hand_idx * markers_per_hand, hand_idx * markers_per_hand + finger_idx)
        for hand_idx in range(num_hands)
        for finger_idx in range(1, markers_per_hand)
    ]


def _infer_marker_count(data_dict):
    if not data_dict:
        raise ValueError("Visualizer requires at least one frame to infer marker count.")

    first_timestamp = next(iter(sorted(data_dict.keys())))
    return np.asarray(data_dict[first_timestamp], dtype=float).shape[0]
