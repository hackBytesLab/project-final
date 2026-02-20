import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
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


def main():
    parser = argparse.ArgumentParser(description='Train LSTM fall-detection model')
    parser.add_argument('--data-dir', default='data', help='Folder containing X.npy and y.npy')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out', default='models/lstm_fall_model.h5', help='Output model path')
    parser.add_argument('--generate-sample', action='store_true', help='Generate small random dataset and exit')
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample(args.data_dir)
        return

    X, y = load_data(args.data_dir)
    if X.ndim != 3:
        raise ValueError('X must be 3D: (n_samples, timesteps, num_features)')

    n_samples, timesteps, num_features = X.shape
    classes = len(np.unique(y))

    # Ensure labels are integer class ids 0..C-1
    if y.ndim > 1:
        y = y.flatten()

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


if __name__ == '__main__':
    main()
