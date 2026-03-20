POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
COORD_DIMS = 2

POSE_FEATURES_PER_PERSON = POSE_LANDMARK_COUNT * COORD_DIMS
HAND_FEATURES_PER_HAND = HAND_LANDMARK_COUNT * COORD_DIMS
FEATURES_PER_PERSON = POSE_FEATURES_PER_PERSON + (2 * HAND_FEATURES_PER_HAND)

# Enhanced add-ons per frame (for single-person layout this expands 150 -> 216):
# - pose velocity (x,y) for all 33 pose landmarks => 66 values
ENHANCED_EXTRA_FEATURES = POSE_FEATURES_PER_PERSON


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
    """
    Returns (max_people, max_hands) while allowing enhanced feature vectors
    that include geometric extras (trunk angle, aspect ratio, hip height ratio, |Δθ|).
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
    if expected_features != num_features:
        enhanced_expected = expected_features + ENHANCED_EXTRA_FEATURES
        if num_features != enhanced_expected:
            raise ValueError(
                "Model input features do not match requested layout: "
                f"model={num_features}, expected={expected_features} or enhanced={enhanced_expected}, "
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


def _safe_point_from_pose_frame(frame_pose_only, lm_idx):
    """
    Extract (x, y) from a pose-only frame vector (first 66 dims of a full frame).
    """
    idx = lm_idx * 2
    return float(frame_pose_only[idx]), float(frame_pose_only[idx + 1])


def compute_geometric_features(frame):
    """
    Compute geometric descriptors from a single frame.

    frame: 1D array containing at least the first POSE_FEATURES_PER_PERSON values.
           (Extra features, if present, are ignored.)

    Returns ndarray (3,) in the order:
        [trunk_angle_deg, aspect_ratio, hip_height_ratio]
    """
    import numpy as np

    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < POSE_FEATURES_PER_PERSON:
        raise ValueError("Frame must be 1D with at least pose coordinates")

    pose = arr[:POSE_FEATURES_PER_PERSON]

    nose_x, nose_y = _safe_point_from_pose_frame(pose, 0)
    lsh_x, lsh_y = _safe_point_from_pose_frame(pose, 11)
    rsh_x, rsh_y = _safe_point_from_pose_frame(pose, 12)
    lhip_x, lhip_y = _safe_point_from_pose_frame(pose, 23)
    rhip_x, rhip_y = _safe_point_from_pose_frame(pose, 24)
    lank_x, lank_y = _safe_point_from_pose_frame(pose, 27)
    rank_x, rank_y = _safe_point_from_pose_frame(pose, 28)

    mid_sh_x = (lsh_x + rsh_x) * 0.5
    mid_sh_y = (lsh_y + rsh_y) * 0.5
    mid_hip_x = (lhip_x + rhip_x) * 0.5
    mid_hip_y = (lhip_y + rhip_y) * 0.5
    mid_ank_y = (lank_y + rank_y) * 0.5

    dx = mid_hip_x - mid_sh_x
    dy = mid_hip_y - mid_sh_y
    trunk_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))

    all_y = [nose_y, mid_sh_y, mid_hip_y, mid_ank_y]
    all_x = [lsh_x, rsh_x, lhip_x, rhip_x]
    body_H = max(all_y) - min(all_y) + 1e-6
    body_W = max(all_x) - min(all_x) + 1e-6
    aspect_ratio = body_H / body_W

    hip_height_ratio = mid_hip_y / (mid_ank_y + 1e-6)
    return np.array([trunk_angle, aspect_ratio, hip_height_ratio], dtype=np.float32)


def enhance_sequence_features(sequence):
    """
    Add pose velocity to a sequence of per-frame features.
    Input: sequence (T, F) where F includes pose+hands 2D coords (default 150).
    Output dims (single-person): 150 + 66 = 216.
    """
    import numpy as np

    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] < POSE_FEATURES_PER_PERSON:
        raise ValueError("Expected sequence of shape (timesteps, features>=pose features)")

    pose = seq[:, :POSE_FEATURES_PER_PERSON]  # (T, 66) pose only
    vel = np.zeros_like(pose, dtype=np.float32)
    if len(pose) > 1:
        vel[1:] = pose[1:] - pose[:-1]

    enhanced = np.concatenate(
        [
            seq,
            vel,
        ],
        axis=1,
    )
    return enhanced
