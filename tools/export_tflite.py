import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def representative_dataset_generator(x_path, max_samples=256):
    x = np.load(x_path, mmap_mode="r")
    if x.ndim != 3:
        raise ValueError(f"Expected 3D X.npy, got shape={x.shape}")
    n = x.shape[0]
    if n <= 0:
        raise ValueError("Representative dataset is empty")
    use_n = min(int(max_samples), n)
    # Evenly spaced indices keep coverage deterministic.
    indices = np.linspace(0, n - 1, num=use_n, dtype=np.int64)
    for idx in indices:
        sample = x[int(idx)].astype(np.float32, copy=False)
        yield [np.expand_dims(sample, axis=0)]


def parse_tf_dtype(name):
    name = str(name).strip().lower()
    if name == "float32":
        return tf.float32
    if name == "int8":
        return tf.int8
    if name == "uint8":
        return tf.uint8
    raise ValueError(f"Unsupported dtype: {name}")


def clone_as_lite_friendly_rnn(model, fixed_timesteps):
    if not isinstance(model, tf.keras.Sequential):
        raise ValueError("Lite-friendly RNN export currently supports Sequential models only")
    if fixed_timesteps is None or int(fixed_timesteps) <= 0:
        raise ValueError("--fixed-timesteps must be a positive integer")

    input_shape = model.input_shape
    if len(input_shape) != 3 or input_shape[-1] is None:
        raise ValueError(f"Expected 3D model input with known feature size, got {input_shape}")

    fixed_timesteps = int(fixed_timesteps)
    num_features = int(input_shape[-1])
    cloned_layers = [tf.keras.Input(shape=(fixed_timesteps, num_features))]

    for layer in model.layers:
        config = layer.get_config()
        config.pop("batch_input_shape", None)
        config.pop("batch_shape", None)
        if isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU, tf.keras.layers.SimpleRNN)):
            config["unroll"] = True
        cloned_layers.append(layer.__class__.from_config(config))

    cloned_model = tf.keras.Sequential(cloned_layers, name=f"{model.name}_lite")
    cloned_model.build((None, fixed_timesteps, num_features))

    for src_layer, dst_layer in zip(model.layers, cloned_model.layers):
        if src_layer.weights:
            dst_layer.set_weights(src_layer.get_weights())

    return cloned_model


def load_export_model(args):
    model = tf.keras.models.load_model(args.keras_model, compile=False)
    if args.lite_friendly_rnn:
        model = clone_as_lite_friendly_rnn(model, fixed_timesteps=args.fixed_timesteps)
    return model


def build_converter(args):
    model = load_export_model(args)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization == "none":
        return converter

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.quantization == "dynamic":
        return converter

    if args.quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
        return converter

    if args.quantization == "int8":
        if not args.representative_x:
            raise ValueError("--representative-x is required for int8 quantization")
        converter.representative_dataset = lambda: representative_dataset_generator(
            args.representative_x,
            max_samples=args.representative_samples,
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = parse_tf_dtype(args.inference_input_type)
        converter.inference_output_type = parse_tf_dtype(args.inference_output_type)
        return converter

    raise ValueError(f"Unknown quantization mode: {args.quantization}")


def apply_select_tf_ops(converter):
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Needed by many recurrent models (LSTM/GRU) during conversion.
    converter._experimental_lower_tensor_list_ops = False


def has_flex_ops(model_content):
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()
    return any("Flex" in op["op_name"] for op in ops)


def main():
    parser = argparse.ArgumentParser(description="Export Keras model (.h5/.keras) to TFLite")
    parser.add_argument("--keras-model", required=True, help="Input Keras model path")
    parser.add_argument("--output", required=True, help="Output .tflite path")
    parser.add_argument(
        "--fixed-timesteps",
        type=int,
        default=0,
        help="Override time dimension for export (required by --lite-friendly-rnn)",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "dynamic", "float16", "int8"],
        default="float16",
        help="TFLite quantization mode",
    )
    parser.add_argument(
        "--representative-x",
        default="",
        help="Path to X.npy for representative calibration (required for int8)",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=256,
        help="Max representative samples for int8 calibration",
    )
    parser.add_argument(
        "--inference-input-type",
        choices=["float32", "int8", "uint8"],
        default="float32",
        help="TFLite inference input dtype (int8 mode only)",
    )
    parser.add_argument(
        "--inference-output-type",
        choices=["float32", "int8", "uint8"],
        default="float32",
        help="TFLite inference output dtype (int8 mode only)",
    )
    parser.add_argument(
        "--select-tf-ops",
        action="store_true",
        help="Allow SELECT_TF_OPS (often required by LSTM/GRU models).",
    )
    parser.add_argument(
        "--lite-friendly-rnn",
        action="store_true",
        help="Rebuild Sequential RNNs with fixed timesteps and unrolled recurrent layers for built-in TFLite ops.",
    )
    parser.add_argument(
        "--require-builtins-only",
        action="store_true",
        help="Fail if the exported model still contains Flex ops.",
    )
    args = parser.parse_args()

    if args.lite_friendly_rnn and args.fixed_timesteps <= 0:
        raise ValueError("--lite-friendly-rnn requires --fixed-timesteps")

    converter = build_converter(args)
    if args.select_tf_ops:
        apply_select_tf_ops(converter)

    try:
        tflite_model = converter.convert()
    except Exception as e:
        if not args.select_tf_ops:
            raise RuntimeError(
                f"TFLite conversion failed: {e}\n"
                "Hint: retry with --select-tf-ops for recurrent models."
            ) from e
        raise

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved: {out_path}")
    print(f"[INFO] Size: {size_mb:.2f} MB")
    if args.require_builtins_only:
        if has_flex_ops(tflite_model):
            raise RuntimeError("Exported model still contains Flex ops")
        print("[INFO] Verified: built-in TFLite ops only (no Flex ops)")


if __name__ == "__main__":
    main()
