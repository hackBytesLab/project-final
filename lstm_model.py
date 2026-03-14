import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization, Dropout


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    gamma = float(gamma)
    alpha_vector = None
    if isinstance(alpha, (list, tuple)):
        alpha_vector = tf.constant([float(x) for x in alpha], dtype=tf.float32)
    else:
        alpha = float(alpha)

    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(
            y_pred,
            tf.keras.backend.epsilon(),
            1.0 - tf.keras.backend.epsilon(),
        )
        cross_entropy = -y_true * tf.math.log(y_pred)
        modulating = tf.pow(1.0 - y_pred, gamma)
        if alpha_vector is not None:
            alpha_weight = y_true * alpha_vector
        else:
            alpha_weight = alpha * y_true
        loss = alpha_weight * modulating * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    focal_loss.__name__ = "categorical_focal_loss"
    return focal_loss


def resolve_loss(loss_name="categorical_crossentropy", focal_gamma=2.0, focal_alpha=0.25):
    normalized = (loss_name or "categorical_crossentropy").strip().lower()
    if normalized in ("ce", "crossentropy", "categorical_crossentropy"):
        return "categorical_crossentropy"
    if normalized in ("focal", "categorical_focal_loss"):
        return categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def build_lstm_model(
    num_features,
    num_classes,
    loss_name="categorical_crossentropy",
    focal_gamma=2.0,
    focal_alpha=0.25,
    dropout_rate=0.3,
    use_batchnorm=True,
):
    layers_list = [
        layers.LSTM(128, return_sequences=True, input_shape=(None, num_features)),
    ]
    if use_batchnorm:
        layers_list.append(BatchNormalization())
    layers_list.append(Dropout(dropout_rate))
    layers_list.append(layers.LSTM(64, return_sequences=False))
    if use_batchnorm:
        layers_list.append(BatchNormalization())
    layers_list.append(Dropout(dropout_rate))
    layers_list.append(layers.Dense(64, activation="relu"))
    layers_list.append(layers.Dense(num_classes, activation="softmax"))

    model = models.Sequential(layers_list)
    model.compile(
        optimizer="adam",
        loss=resolve_loss(
            loss_name=loss_name,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        ),
        metrics=["accuracy"],
    )
    return model


def predict_sequence(model, sequence):
    sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    prediction = model.predict(sequence, verbose=0)
    return prediction.argmax(axis=1)[0]
