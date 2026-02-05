import tensorflow as tf
from tensorflow.keras import layers, models

# ===== สร้างโมเดล LSTM =====
def build_lstm_model(num_features, num_classes):
    model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(None, num_features)),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ===== ตัวอย่างการ predict =====
def predict_sequence(model, sequence):
    # sequence shape: (timesteps, num_features)
    sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    prediction = model.predict(sequence)
    return prediction.argmax(axis=1)[0]