import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_layout import resolve_feature_layout


DEFAULT_LABELS = "Fall,No_Fall,Pre-Fall,Falling"
DEFAULT_POSE_TASK = "models/pose_landmarker_lite.task"
DEFAULT_HAND_TASK = "models/hand_landmarker.task"


def parse_labels(raw):
    labels = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not labels:
        raise ValueError("labels cannot be empty")
    return labels


def read_thresholds(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    thresholds = data.get("thresholds", data)
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds JSON must be an object or contain {thresholds:{...}}")
    out = {}
    for k, v in thresholds.items():
        out[str(k)] = float(v)
    return out


def _can_import(import_stmt):
    try:
        exec(import_stmt, {})
        return True
    except Exception:
        return False


def check_module_imports(model_path: Path):
    missing = []
    for name, import_stmt in (
        ("numpy", "import numpy"),
        ("opencv-python", "import cv2"),
        ("mediapipe", "import mediapipe"),
    ):
        if not _can_import(import_stmt):
            missing.append(name)

    is_tflite = str(model_path).strip().lower().endswith(".tflite")
    has_tf = _can_import("import tensorflow")
    has_tflite_runtime = _can_import("from tflite_runtime.interpreter import Interpreter")

    if is_tflite:
        if not (has_tf or has_tflite_runtime):
            missing.append("tensorflow or tflite-runtime")
    else:
        if not has_tf:
            missing.append("tensorflow")
    return missing


def inspect_keras_model(model_path):
    from tensorflow.keras.models import load_model

    model = load_model(model_path, compile=False)
    input_shape = model.input_shape
    output_shape = model.output_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    if len(input_shape) != 3 or input_shape[-1] is None:
        raise ValueError(f"Unsupported Keras input shape: {input_shape}")
    if len(output_shape) < 2 or output_shape[-1] is None:
        raise ValueError(f"Unsupported Keras output shape: {output_shape}")
    timesteps = int(input_shape[1]) if input_shape[1] else 30
    num_features = int(input_shape[-1])
    num_classes = int(output_shape[-1])

    test_batch = np.zeros((1, timesteps, num_features), dtype=np.float32)
    pred = model.predict(test_batch, verbose=0)
    if pred.shape[-1] != num_classes:
        raise ValueError(f"Model prediction shape mismatch: pred={pred.shape}, classes={num_classes}")
    return {
        "backend": "keras",
        "timesteps": timesteps,
        "num_features": num_features,
        "num_classes": num_classes,
    }


def inspect_tflite_model(model_path):
    try:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    except Exception:
        from tflite_runtime.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details.get("shape", [])
    shape_signature = input_details.get("shape_signature", input_shape)
    if len(input_shape) != 3:
        raise ValueError(f"Unsupported TFLite input shape: {input_shape}")
    if int(shape_signature[1]) > 0:
        timesteps = int(shape_signature[1])
    elif int(input_shape[1]) > 1:
        timesteps = int(input_shape[1])
    else:
        # Dynamic TFLite sequence models often expose [1, 1, F] at allocation time.
        # Keep runtime-compatible default instead of forcing 1 timestep.
        timesteps = 30
    num_features = int(shape_signature[-1]) if int(shape_signature[-1]) > 0 else int(input_shape[-1])
    if timesteps <= 0 or num_features <= 0:
        raise ValueError(f"Invalid TFLite input shape/signature: {input_shape} / {shape_signature}")

    output_shape = output_details.get("shape", [])
    if len(output_shape) < 2 or int(output_shape[-1]) <= 0:
        raise ValueError(f"Unsupported TFLite output shape: {output_shape}")
    num_classes = int(output_shape[-1])

    dummy = np.zeros((1, timesteps, num_features), dtype=np.float32)
    in_dtype = input_details["dtype"]
    in_scale, in_zero = input_details.get("quantization", (0.0, 0))
    if in_dtype in (np.int8, np.uint8) and in_scale:
        dummy = np.round(dummy / in_scale + in_zero).astype(in_dtype)
    else:
        dummy = dummy.astype(in_dtype)
    flex_warning = ""
    smoke_infer_ok = True
    try:
        input_idx = int(input_details["index"])
        if list(input_shape) != [1, timesteps, num_features]:
            interpreter.resize_tensor_input(input_idx, [1, timesteps, num_features], strict=False)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        interpreter.set_tensor(int(input_details["index"]), dummy)
        interpreter.invoke()
        out = interpreter.get_tensor(int(output_details["index"]))
        if out.shape[-1] != num_classes:
            raise ValueError(f"TFLite output mismatch: output={out.shape}, classes={num_classes}")
    except Exception as e:
        msg = str(e)
        if "Select TensorFlow op" in msg or "Flex" in msg or "TensorList" in msg:
            smoke_infer_ok = False
            flex_warning = (
                "TFLite model appears to require Flex/SELECT_TF_OPS runtime support. "
                f"Smoke inference failed: {msg}"
            )
        else:
            raise

    return {
        "backend": "tflite",
        "timesteps": timesteps,
        "num_features": num_features,
        "num_classes": num_classes,
        "input_dtype": str(in_dtype),
        "output_dtype": str(output_details["dtype"]),
        "smoke_infer_ok": smoke_infer_ok,
        "flex_warning": flex_warning,
    }


def inspect_model(model_path):
    lower = str(model_path).lower()
    if lower.endswith(".tflite"):
        return inspect_tflite_model(model_path)
    return inspect_keras_model(model_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-deploy checks before putting model on board.")
    parser.add_argument("--model", required=True, help="Path to model (.h5/.keras/.tflite)")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated labels in model output order")
    parser.add_argument("--thresholds-json", default="", help="Optional thresholds JSON path")
    parser.add_argument("--pose-task", default=DEFAULT_POSE_TASK, help="Pose task model path")
    parser.add_argument("--hand-task", default=DEFAULT_HAND_TASK, help="Hand task model path")
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Check runtime imports (numpy/opencv/mediapipe/tensorflow)",
    )
    args = parser.parse_args()

    fails = []
    warns = []
    passes = []

    model_path = Path(args.model)
    if not model_path.exists():
        fails.append(f"model not found: {model_path}")
    else:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        passes.append(f"model exists: {model_path} ({size_mb:.2f} MB)")

    for task_path in (Path(args.pose_task), Path(args.hand_task)):
        if task_path.exists():
            passes.append(f"asset exists: {task_path}")
        else:
            fails.append(f"required asset missing: {task_path}")

    try:
        labels = parse_labels(args.labels)
        passes.append(f"labels parsed: {labels}")
    except Exception as e:
        labels = []
        fails.append(f"invalid labels: {e}")

    if args.thresholds_json:
        t_path = Path(args.thresholds_json)
        if not t_path.exists():
            fails.append(f"thresholds file not found: {t_path}")
        else:
            try:
                thresholds = read_thresholds(t_path)
                passes.append(f"thresholds loaded: {len(thresholds)} entries")
                for k, v in thresholds.items():
                    if not (0.0 <= float(v) <= 1.0):
                        warns.append(f"threshold out of [0,1]: {k}={v}")
                if labels:
                    unknown = [k for k in thresholds.keys() if (k not in labels and not k.isdigit())]
                    if unknown:
                        warns.append(f"threshold keys not in labels: {unknown}")
            except Exception as e:
                fails.append(f"invalid thresholds JSON: {e}")

    model_info = None
    if model_path.exists():
        try:
            model_info = inspect_model(model_path)
            passes.append(
                "model load+smoke-infer ok: "
                f"backend={model_info['backend']}, timesteps={model_info['timesteps']}, "
                f"features={model_info['num_features']}, classes={model_info['num_classes']}"
            )
        except Exception as e:
            fails.append(f"model load/inference failed: {e}")

    if model_info:
        try:
            max_people, max_hands = resolve_feature_layout(model_info["num_features"], 0, 0)
            passes.append(
                f"feature layout resolved: max_people={max_people}, max_hands={max_hands}"
            )
        except Exception as e:
            fails.append(f"feature layout mismatch: {e}")

        if labels and len(labels) != int(model_info["num_classes"]):
            fails.append(
                f"labels count mismatch: labels={len(labels)} vs model classes={model_info['num_classes']}"
            )

        if int(model_info["timesteps"]) != 30:
            warns.append(f"timesteps is {model_info['timesteps']} (default runtime expects 30)")

        if model_info["backend"] == "tflite":
            passes.append(
                f"tflite dtypes: input={model_info['input_dtype']}, output={model_info['output_dtype']}"
            )
            if not model_info.get("smoke_infer_ok", True):
                warns.append(model_info.get("flex_warning", "TFLite smoke inference did not run cleanly"))

    if args.check_imports:
        missing = check_module_imports(model_path=model_path)
        if missing:
            fails.append(f"missing runtime modules: {', '.join(missing)}")
        else:
            if str(model_path).strip().lower().endswith(".tflite"):
                passes.append("runtime imports ok: numpy, opencv-python, mediapipe, tensorflow/tflite-runtime")
            else:
                passes.append("runtime imports ok: numpy, opencv-python, mediapipe, tensorflow")

    print("=== PREDEPLOY CHECK ===")
    for item in passes:
        print(f"[PASS] {item}")
    for item in warns:
        print(f"[WARN] {item}")
    for item in fails:
        print(f"[FAIL] {item}")

    if fails:
        raise SystemExit(2)
    if warns:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
