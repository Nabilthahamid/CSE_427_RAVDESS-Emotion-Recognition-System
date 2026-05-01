"""Optional CNN + BiLSTM training on log-mel spectrogram sequences."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd


def _require_tf():
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
        return tf, keras
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is not installed. Install optional deep dependencies first."
        ) from exc


def _load_log_mel(file_path: str, sr: int = 22050, duration: float = 3.0, n_mels: int = 64) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=sr, duration=duration, offset=0.5)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # time-major for sequence modeling
    return log_mel.T.astype(np.float32)


def _to_tensor_dataset(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[str]]:
    x_list = [_load_log_mel(path) for path in frame["path"].tolist()]
    max_t = max(arr.shape[0] for arr in x_list)
    n_mels = x_list[0].shape[1]

    x = np.zeros((len(x_list), max_t, n_mels), dtype=np.float32)
    for i, arr in enumerate(x_list):
        x[i, : arr.shape[0], :] = arr

    labels = sorted(frame["emotion"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    y_idx = np.array([label_to_idx[v] for v in frame["emotion"].tolist()], dtype=np.int64)
    return x, y_idx, label_to_idx, labels


def train_cnn_bilstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    epochs: int = 25,
) -> dict[str, float]:
    tf, keras = _require_tf()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_train, y_train, label_to_idx, labels = _to_tensor_dataset(train_df)
    x_val, y_val, _, _ = _to_tensor_dataset(val_df)
    x_test, y_test, _, _ = _to_tensor_dataset(test_df)

    # Ensure shared time dimension.
    max_t = max(x_train.shape[1], x_val.shape[1], x_test.shape[1])

    def pad_time(x_in: np.ndarray) -> np.ndarray:
        if x_in.shape[1] == max_t:
            return x_in
        x_out = np.zeros((x_in.shape[0], max_t, x_in.shape[2]), dtype=np.float32)
        x_out[:, : x_in.shape[1], :] = x_in
        return x_out

    x_train = pad_time(x_train)
    x_val = pad_time(x_val)
    x_test = pad_time(x_test)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_t, x_train.shape[2])),
            keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False)),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(len(labels), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    from sklearn.metrics import f1_score  # local import to keep module lightweight

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    model.save(out / "cnn_bilstm.keras")

    import json

    with (out / "cnn_bilstm_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (out / "cnn_bilstm_labels.json").open("w", encoding="utf-8") as handle:
        json.dump(label_to_idx, handle, indent=2)

    return metrics
