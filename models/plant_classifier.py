"""
plant_classifier.py
CNN-based plant classifier built on TensorFlow/Keras.

The PlantClassifier class wraps all model creation, training, evaluation,
prediction, and persistence logic.  Import it from main.py or processor.py.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scripts import config
from scripts.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class PlantClassifier:
    """
    Lightweight CNN plant identifier.

    Architecture
    ------------
    Three convolutional blocks (Conv → BatchNorm → MaxPool → Dropout)
    followed by a dense head.  Kept deliberately shallow to avoid
    overfitting on limited iNat training data.

    Parameters
    ----------
    model_path:
        Where to save / load the trained .keras model file.
    input_size:
        (height, width) fed into the network.  Must match segmentation output.
    confidence_threshold:
        Predictions below this value are reported as "unknown".
    """

    def __init__(
        self,
        model_path: Path = config.MODEL_FILE,
        input_size: tuple[int, int] = config.MODEL_INPUT_SIZE,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold

        self.model: Optional[keras.Model] = None
        self.class_names: list[str] = []
        self._class_index_path = self.model_path.parent / "class_index.json"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_dir: Path = config.INAT_DIR,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        validation_split: float = config.VALIDATION_SPLIT,
        learning_rate: float = config.LEARNING_RATE,
    ) -> keras.callbacks.History:
        """
        Train (or retrain) the classifier on images organised as:
            data_dir/<species_name>/<image_files>

        Saves the trained model and class index automatically.
        """
        logger.info("Loading training data from %s", data_dir)
        train_ds, val_ds = self._build_datasets(data_dir, batch_size, validation_split)

        self.class_names = train_ds.class_names
        num_classes = len(self.class_names)
        logger.info("Classes found: %d", num_classes)

        self.model = self._build_model(num_classes, learning_rate)

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ]

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        self._save()
        logger.info("Training complete. Model saved to %s", self.model_path)
        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data_dir: Path = config.INAT_DIR, batch_size: int = config.BATCH_SIZE) -> dict[str, float]:
        """
        Evaluate the model on a held-out validation split.
        Returns a dict with 'loss' and 'accuracy'.
        """
        self._require_model()
        _, val_ds = self._build_datasets(data_dir, batch_size, validation_split=config.VALIDATION_SPLIT)
        loss, accuracy = self.model.evaluate(val_ds, verbose=0)  # type: ignore[union-attr]
        metrics = {"loss": float(loss), "accuracy": float(accuracy)}
        logger.info("Evaluation — loss: %.4f  accuracy: %.4f", loss, accuracy)
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_image(self, image: np.ndarray) -> tuple[str, float]:
        """
        Predict the species of a single pre-processed image array.

        Parameters
        ----------
        image:
            NumPy array of shape (H, W, 3), uint8 or float32.

        Returns
        -------
        (species_name, confidence)
            species_name is "unknown" when confidence < threshold.
        """
        self._require_model()
        img = self._preprocess(image)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 3)
        probs = self.model.predict(img, verbose=0)[0]  # type: ignore[union-attr]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        species = self.class_names[idx] if confidence >= self.confidence_threshold else "unknown"
        return species, confidence

    def predict_batch(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Run predict_image over a list of images."""
        return [self.predict_image(img) for img in images]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load a previously trained model and class index from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model at {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        with self._class_index_path.open() as fh:
            self.class_names = json.load(fh)
        logger.info("Model loaded from %s (%d classes)", self.model_path, len(self.class_names))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_datasets(
        self,
        data_dir: Path,
        batch_size: int,
        validation_split: float,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Build train/val tf.data.Dataset objects with augmentation."""
        h, w = self.input_size

        train_ds = keras.utils.image_dataset_from_directory(
            str(data_dir),
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=(h, w),
            batch_size=batch_size,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            str(data_dir),
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=(h, w),
            batch_size=batch_size,
        )

        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomBrightness(0.1),
        ])

        normalise = layers.Rescaling(1.0 / 255)

        def prepare_train(x, y):
            x = augmentation(x, training=True)
            return normalise(x), y

        def prepare_val(x, y):
            return normalise(x), y

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        val_ds = val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        return train_ds, val_ds

    def _build_model(self, num_classes: int, learning_rate: float) -> keras.Model:
        """Construct a lightweight CNN suitable for plant classification."""
        h, w = self.input_size
        inputs = keras.Input(shape=(h, w, 3))

        x = self._conv_block(inputs, filters=32)
        x = self._conv_block(x, filters=64)
        x = self._conv_block(x, filters=128)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(config.DROPOUT_RATE)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name="plant_classifier")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary(print_fn=logger.info)
        return model

    @staticmethod
    def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
        """Conv2D → BatchNorm → MaxPool → Dropout block."""
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
        return x

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalise a single image to model input shape."""
        import cv2
        h, w = self.input_size
        resized = cv2.resize(image, (w, h))
        return resized.astype(np.float32) / 255.0

    def _save(self) -> None:
        ensure_dir(self.model_path.parent)
        self.model.save(str(self.model_path))  # type: ignore[union-attr]
        with self._class_index_path.open("w") as fh:
            json.dump(self.class_names, fh, indent=2)

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
