POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
COORD_DIMS = 2

POSE_FEATURES_PER_PERSON = POSE_LANDMARK_COUNT * COORD_DIMS
HAND_FEATURES_PER_HAND = HAND_LANDMARK_COUNT * COORD_DIMS
FEATURES_PER_PERSON = POSE_FEATURES_PER_PERSON + (2 * HAND_FEATURES_PER_HAND)
POSE_VELOCITY_FEATURES = POSE_FEATURES_PER_PERSON
SUMMARY_FEATURES = 2  # trunk angle + hip height
ENHANCED_EXTRA_FEATURES = POSE_VELOCITY_FEATURES + SUMMARY_FEATURES
LEGACY_ENHANCED_EXTRA_FEATURES = POSE_VELOCITY_FEATURES


def compute_num_features(max_people, max_hands=None):
    if max_people < 1:
        raise ValueError("max_people must be >= 1")
    if max_hands is None:
        max_hands = max_people * 2
    if max_hands < 0:
        raise ValueError("max_hands must be >= 0")
    return (max_people * POSE_FEATURES_PER_PERSON) + (max_hands * HAND_FEATURES_PER_HAND)


def infer_people_from_num_features(num_features):
    if num_features <= 0:
        return None
    if num_features % FEATURES_PER_PERSON != 0:
        return None
    return num_features // FEATURES_PER_PERSON


def infer_enhancement_variant(num_features, base_features):
    if num_features == base_features:
        return "base"
    if num_features == base_features + LEGACY_ENHANCED_EXTRA_FEATURES:
        return "velocity"
    if num_features == base_features + ENHANCED_EXTRA_FEATURES:
        return "full"
    return None


def resolve_feature_layout(num_features, max_people_arg=0, max_hands_arg=0):
    """
    Returns (max_people, max_hands) while allowing enhanced feature vectors
    that include either:
    - pose velocity only (legacy Pi layout)
    - pose velocity + trunk angle + hip height
    """
    inferred_people = infer_people_from_num_features(num_features)
    if max_people_arg > 0:
        max_people = max_people_arg
    elif inferred_people:
        max_people = inferred_people
    else:
        max_people = 1

    max_hands = max_hands_arg if max_hands_arg > 0 else max_people * 2
    expected_features = compute_num_features(max_people, max_hands)
    enhancement_variant = infer_enhancement_variant(num_features, expected_features)
    if enhancement_variant is None:
        legacy_expected = expected_features + LEGACY_ENHANCED_EXTRA_FEATURES
        enhanced_expected = expected_features + ENHANCED_EXTRA_FEATURES
        raise ValueError(
            "Model input features do not match requested layout: "
            f"model={num_features}, expected={expected_features}, legacy_enhanced={legacy_expected}, "
            f"enhanced={enhanced_expected}, max_people={max_people}, max_hands={max_hands}"
        )
    return max_people, max_hands


def build_frame_features(pose_landmarks, hand_landmarks, max_people, max_hands):
    return build_frame_features_with_options(
        pose_landmarks=pose_landmarks,
        hand_landmarks=hand_landmarks,
        max_people=max_people,
        max_hands=max_hands,
        normalize_geometry=False,
    )


def _normalize_entity_landmarks(landmarks, center_index=0):
    if not landmarks:
        return []

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    scale = max(max_x - min_x, max_y - min_y, 1e-6)

    center = landmarks[center_index] if 0 <= center_index < len(landmarks) else landmarks[0]
    cx = center.x
    cy = center.y

    normalized = []
    for lm in landmarks:
        normalized.append(((lm.x - cx) / scale, (lm.y - cy) / scale))
    return normalized


def build_frame_features_with_options(
    pose_landmarks,
    hand_landmarks,
    max_people,
    max_hands,
    normalize_geometry=False,
):
    features = []

    poses = pose_landmarks or []
    for i in range(max_people):
        if i < len(poses):
            if normalize_geometry:
                normalized = _normalize_entity_landmarks(poses[i], center_index=0)
                for x, y in normalized:
                    features.extend([x, y])
            else:
                for lm in poses[i]:
                    features.extend([lm.x, lm.y])
        else:
            features.extend([0.0] * POSE_FEATURES_PER_PERSON)

    hands = hand_landmarks or []
    for i in range(max_hands):
        if i < len(hands):
            if normalize_geometry:
                normalized = _normalize_entity_landmarks(hands[i], center_index=0)
                for x, y in normalized:
                    features.extend([x, y])
            else:
                for lm in hands[i]:
                    features.extend([lm.x, lm.y])
        else:
            features.extend([0.0] * HAND_FEATURES_PER_HAND)

    return features


def enhance_sequence_features(sequence, include_summary_features=True):
    """
    Add temporal/dynamic features to a sequence of per-frame features.
    Input: sequence (T, F) where F includes pose+hands 2D coords (default 150).
    Output:
    - include_summary_features=True:  F + pose_velocity(66) + trunk_angle(1) + hip_height(1) => 218 when F=150
    - include_summary_features=False: F + pose_velocity(66) => 216 when F=150
    """
    import numpy as np

    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] < POSE_FEATURES_PER_PERSON:
        raise ValueError("Expected sequence of shape (timesteps, features>=pose features)")

    T, F = seq.shape
    velocity = np.zeros_like(seq)
    if T > 1:
        velocity[1:] = seq[1:] - seq[:-1]

    pose_velocity = velocity[:, :POSE_FEATURES_PER_PERSON]

    def get_point(frame, landmark_idx):
        idx = landmark_idx * 2
        return float(frame[idx]), float(frame[idx + 1])

    angles = np.zeros((T, 1), dtype=np.float32)
    hip_heights = np.zeros((T, 1), dtype=np.float32)
    for i in range(T):
        lsx, lsy = get_point(seq[i], 11)
        rsx, rsy = get_point(seq[i], 12)
        lhx, lhy = get_point(seq[i], 23)
        rhx, rhy = get_point(seq[i], 24)

        mid_sx, mid_sy = (lsx + rsx) / 2.0, (lsy + rsy) / 2.0
        mid_hx, mid_hy = (lhx + rhx) / 2.0, (lhy + rhy) / 2.0

        dx = mid_hx - mid_sx
        dy = mid_hy - mid_sy
        angles[i, 0] = np.degrees(np.arctan2(dx, dy + 1e-6))
        hip_heights[i, 0] = mid_hy

    extras = [seq, pose_velocity]
    if include_summary_features:
        extras.extend([angles, hip_heights])
    enhanced = np.concatenate(extras, axis=1)
    return enhanced
