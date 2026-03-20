import numpy as np
from feature_layout import compute_geometric_features


DEFAULT_THRESHOLDS = {
    "trunk_angle_fall": 60.0,
    "trunk_angle_prefall": 30.0,
    "aspect_ratio_fall": 0.8,
    "hip_height_ratio_fall": 0.85,
    "hip_height_ratio_prefall": 0.70,
    "angular_velocity_fall": 5.0,
}


def _find_label_id(label_names, target):
    """หา index ของ label (case-insensitive, strip whitespace)"""
    target = target.strip().lower()
    for i, name in enumerate(label_names):
        if name.strip().lower() == target:
            return i
    return None


def _find_label_id_multi(label_names, *targets):
    """
    หา label id จากหลาย target — return id แรกที่เจอ
    ใช้ is not None เพื่อรองรับกรณี index = 0 (falsy แต่ valid)
    """
    for target in targets:
        result = _find_label_id(label_names, target)
        if result is not None:
            return result
    return None


def rule_based_check(frame, sequence=None, prev_frame=None, thresholds=DEFAULT_THRESHOLDS):
    """
    Rule-based state classifier จาก geometric features

    ลำดับ priority:
      FALL → PRE_FALL → NO_FALL

    Parameters
    ----------
    frame       : array (F,) — frame ล่าสุด
    sequence    : array (T, F) optional — ถ้าส่งมาจะใช้ max ang_vel ทั้ง window
                  ดีกว่าดูแค่ last frame เพราะ Falling spike อาจอยู่กลาง sequence
    prev_frame  : array (F,) optional — fallback ถ้าไม่มี sequence
    thresholds  : dict
    """
    geo = compute_geometric_features(frame)
    trunk_angle, aspect_ratio, hip_height_ratio = geo

    # Angular velocity — ใช้ max ทั้ง sequence ถ้าทำได้
    if sequence is not None and len(sequence) > 1:
        angles = np.array([compute_geometric_features(f)[0] for f in sequence], dtype=np.float32)
        ang_vel = float(np.max(np.abs(np.diff(angles))))
    elif prev_frame is not None:
        prev_geo = compute_geometric_features(prev_frame)
        ang_vel = abs(float(trunk_angle) - float(prev_geo[0]))
    else:
        ang_vel = 0.0

    # FALL: นอนแล้ว (AR ต่ำ + สะโพกสูง + ลำตัวเอียงมาก หรือ velocity spike)
    if (aspect_ratio < thresholds["aspect_ratio_fall"]
            and hip_height_ratio > thresholds["hip_height_ratio_fall"]):
        if (trunk_angle > thresholds["trunk_angle_fall"]
                or ang_vel > thresholds["angular_velocity_fall"]):
            return "FALL"

    # PRE_FALL: ท่าเสี่ยง (เอียง + สะโพกเริ่มลง)
    if (trunk_angle > thresholds["trunk_angle_prefall"]
            and hip_height_ratio > thresholds["hip_height_ratio_prefall"]):
        return "PRE_FALL"

    return "NO_FALL"


def fuse_rule_with_lstm(probs, label_names, sequence, thresholds=DEFAULT_THRESHOLDS, prob_floor=0.35):
    """
    Fuse LSTM softmax output กับ rule-based classifier

    Priority (safety-first):
      1. Rule says FALL           → FALL
      2. LSTM says Fall ≥ prob_floor → FALL
      3. Rule says PRE_FALL       → PRE_FALL
      4. LSTM argmax (default)

    Returns
    -------
    (final_label_id, final_label_name, used_rule: bool)
    """
    last_frame = sequence[-1]

    lstm_idx   = int(np.argmax(probs))
    lstm_score = float(probs[lstm_idx])

    # ใช้ sequence ทั้งหมดเพื่อคำนวณ ang_vel max
    rule_pred = rule_based_check(last_frame, sequence=sequence, thresholds=thresholds)

    # ✅ FIX: ใช้ _find_label_id_multi แทน 'or' เพื่อรองรับ index = 0
    fall_id = _find_label_id_multi(label_names, "fall")
    pref_id = _find_label_id_multi(label_names, "pre-fall", "pre_fall")

    # 1. Rule บอก FALL → เชื่อ rule (safety-first)
    if rule_pred == "FALL" and fall_id is not None:
        return fall_id, label_names[fall_id], True

    # 2. LSTM บอก Fall ด้วย confidence พอ → เชื่อ LSTM
    if fall_id is not None and lstm_idx == fall_id and lstm_score >= prob_floor:
        return fall_id, label_names[fall_id], False

    # 3. Rule บอก PRE_FALL → escalate
    if rule_pred == "PRE_FALL" and pref_id is not None:
        return pref_id, label_names[pref_id], True

    # 4. Default: ใช้ LSTM argmax
    return lstm_idx, label_names[lstm_idx], False


def separate_prefall_falling(enhanced_sequence, label_names):
    """
    Override ระหว่าง PRE_FALL กับ FALLING โดยใช้ joint velocity spike

    ใช้เฉพาะเมื่อ --enhance-features เปิด (sequence มี 216 features)
    velocity อยู่ที่ cols 150..216 → reshape เป็น (T, 33, 2)

    Heuristic:
      Falling  = max joint speed > 0.05  หรือ  mean top-5 speed > 0.03
      Pre-Fall = ไม่มี spike (เปลี่ยนช้า)

    Returns
    -------
    label_id (int) หรือ None ถ้าหาไม่เจอ / feature ไม่พอ
    """
    if enhanced_sequence.shape[1] < 216:
        return None

    vel = enhanced_sequence[:, 150:216].reshape(enhanced_sequence.shape[0], 33, 2)
    joint_speed = np.linalg.norm(vel, axis=2)          # (T, 33)
    max_speed   = float(np.max(joint_speed))
    # mean ของ top-5 joint ที่เร็วสุดต่อ frame แล้ว average ทุก frame
    mean_top_speed = float(np.mean(np.sort(joint_speed, axis=1)[:, -5:]))

    # ✅ FIX: ใช้ _find_label_id_multi + รองรับ "Falling" / "falling" / "FALLING"
    fall_idx = _find_label_id_multi(label_names, "falling")
    pref_idx = _find_label_id_multi(label_names, "pre-fall", "pre_fall")

    if fall_idx is None or pref_idx is None:
        return None

    if max_speed > 0.05 or mean_top_speed > 0.03:
        return fall_idx  # มี spike → Falling
    return pref_idx      # ช้า → Pre-Fall
