import os
import argparse
import json
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from lstm_model import build_lstm_model


def generate_sample(data_dir, samples=200, timesteps=30, num_features=150, num_classes=4):
    os.makedirs(data_dir, exist_ok=True)
    X = np.random.rand(samples, timesteps, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(samples,))
    np.save(os.path.join(data_dir, 'X.npy'), X)
    np.save(os.path.join(data_dir, 'y.npy'), y)
    print(f"Generated sample data -> {data_dir}/X.npy, {data_dir}/y.npy")


def load_data(data_dir):
    x_path = os.path.join(data_dir, 'X.npy')
    y_path = os.path.join(data_dir, 'y.npy')
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError('Expected X.npy and y.npy in data directory')
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def parse_labels(raw_labels):
    labels = [x.strip() for x in (raw_labels or '').split(',') if x.strip()]
    return labels


def load_labels_from_class_map(data_dir):
    class_map_path = os.path.join(data_dir, 'class_map.json')
    if not os.path.exists(class_map_path):
        return None

    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)

    id_to_name = {}
    for name, idx in class_map.items():
        try:
            id_to_name[int(idx)] = str(name)
        except (TypeError, ValueError):
            continue

    if not id_to_name:
        return None

    max_id = max(id_to_name.keys())
    return [id_to_name.get(i, f'class_{i}') for i in range(max_id + 1)]


def resolve_label_names(data_dir, labels_arg, num_classes):
    labels = parse_labels(labels_arg)
    if labels:
        if len(labels) != num_classes:
            raise ValueError(
                f'--labels count ({len(labels)}) does not match num_classes ({num_classes})'
            )
        return labels

    map_labels = load_labels_from_class_map(data_dir)
    if map_labels and len(map_labels) == num_classes:
        return map_labels

    return [f'class_{i}' for i in range(num_classes)]


def save_eval_reports(eval_dir, label_names, y_true, y_pred):
    os.makedirs(eval_dir, exist_ok=True)
    labels_idx = list(range(len(label_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_csv = os.path.join(eval_dir, 'confusion_matrix.csv')
    with open(cm_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\pred'] + label_names)
        for i, row in enumerate(cm):
            writer.writerow([label_names[i]] + row.tolist())

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_json = os.path.join(eval_dir, 'classification_report.json')
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    report_txt = os.path.join(eval_dir, 'classification_report.txt')
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=labels_idx,
                target_names=label_names,
                zero_division=0,
            )
        )

    summary = {
        'accuracy': report_dict.get('accuracy', 0.0),
        'macro_precision': report_dict.get('macro avg', {}).get('precision', 0.0),
        'macro_recall': report_dict.get('macro avg', {}).get('recall', 0.0),
        'macro_f1': report_dict.get('macro avg', {}).get('f1-score', 0.0),
        'weighted_precision': report_dict.get('weighted avg', {}).get('precision', 0.0),
        'weighted_recall': report_dict.get('weighted avg', {}).get('recall', 0.0),
        'weighted_f1': report_dict.get('weighted avg', {}).get('f1-score', 0.0),
    }
    summary_json = os.path.join(eval_dir, 'metrics_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Saved evaluation:')
    print(' -', cm_csv)
    print(' -', report_json)
    print(' -', report_txt)
    print(' -', summary_json)
    print(
        'Validation summary: '
        f"accuracy={summary['accuracy']:.4f}, "
        f"macro_recall={summary['macro_recall']:.4f}, "
        f"macro_f1={summary['macro_f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description='Train LSTM fall-detection model')
    parser.add_argument('--data-dir', default='data', help='Folder containing X.npy and y.npy')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out', default='models/lstm_fall_model.h5', help='Output model path')
    parser.add_argument('--eval-dir', default='work_csv', help='Folder to save confusion matrix and metric reports')
    parser.add_argument('--labels', default='', help='Optional comma-separated labels in class-id order')
    parser.add_argument('--generate-sample', action='store_true', help='Generate small random dataset and exit')
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample(args.data_dir)
        return

    X, y = load_data(args.data_dir)
    if X.ndim != 3:
        raise ValueError('X must be 3D: (n_samples, timesteps, num_features)')

    _, _, num_features = X.shape

    # Ensure labels are integer class ids 0..C-1
    if y.ndim > 1:
        y = y.flatten()
    if y.size == 0:
        raise ValueError('y is empty')
    if np.min(y) < 0:
        raise ValueError('y must contain non-negative class ids')
    classes = int(np.max(y)) + 1
    label_names = resolve_label_names(args.data_dir, args.labels, classes)

    y_cat = to_categorical(y, num_classes=classes)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    model = build_lstm_model(num_features, classes)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    chk = ModelCheckpoint(args.out, save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=args.epochs,
              batch_size=args.batch_size,
              callbacks=[chk, early])

    print('Training finished. Best model saved to', args.out)
    val_probs = model.predict(X_val, verbose=0)
    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(val_probs, axis=1)
    save_eval_reports(args.eval_dir, label_names, y_val_true, y_val_pred)


if __name__ == '__main__':
    main()
