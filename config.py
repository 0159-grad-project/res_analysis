SINGLE_HAND_NAMES = [
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]

TWO_HAND_NAMES = [
    "left_wrist",
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
    "right_wrist",
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
]

def get_marker_names(num_hands=1):
    if num_hands == 1:
        return SINGLE_HAND_NAMES.copy()
    if num_hands == 2:
        return TWO_HAND_NAMES.copy()

    raise ValueError(f"Unsupported num_hands={num_hands}. Expected 1 or 2.")


