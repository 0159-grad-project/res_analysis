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

NUM_HANDS = 1
N_MARKERS = len(SINGLE_HAND_NAMES)
NAMES = SINGLE_HAND_NAMES.copy()


def configure(num_hands=1):
    global NUM_HANDS, N_MARKERS, NAMES

    if num_hands == 1:
        names = SINGLE_HAND_NAMES
    elif num_hands == 2:
        names = TWO_HAND_NAMES
    else:
        raise ValueError(f"Unsupported num_hands={num_hands}. Expected 1 or 2.")

    NUM_HANDS = num_hands
    NAMES = names.copy()
    N_MARKERS = len(NAMES)


configure()
