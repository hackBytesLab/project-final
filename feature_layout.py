POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
COORD_DIMS = 2

POSE_FEATURES_PER_PERSON = POSE_LANDMARK_COUNT * COORD_DIMS
HAND_FEATURES_PER_HAND = HAND_LANDMARK_COUNT * COORD_DIMS
FEATURES_PER_PERSON = POSE_FEATURES_PER_PERSON + (2 * HAND_FEATURES_PER_HAND)


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


def resolve_feature_layout(num_features, max_people_arg=0, max_hands_arg=0):
    inferred_people = infer_people_from_num_features(num_features)
    if max_people_arg > 0:
        max_people = max_people_arg
    elif inferred_people:
        max_people = inferred_people
    else:
        max_people = 1

    max_hands = max_hands_arg if max_hands_arg > 0 else max_people * 2
    expected_features = compute_num_features(max_people, max_hands)
    if expected_features != num_features:
        raise ValueError(
            "Model input features do not match requested layout: "
            f"model={num_features}, expected={expected_features}, "
            f"max_people={max_people}, max_hands={max_hands}"
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
