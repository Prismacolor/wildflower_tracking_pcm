"""
CNN-based plant classifier built on TensorFlow/Keras, using a pretrained
EfficientNetB0 backbone (transfer learning) with a custom classification head.

The PlantClassifier class wraps all model creation, training, evaluation,
prediction, and persistence logic. Import it from main.py or processor.py.

Training data is scoped by iNat place_id (data/iNat_data/<place_id>/), so
switching restoration sites only requires changing config.INAT_PLACE_ID
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from scripts import config
from utils.utils import (
    build_filtered_training_dir,
    collect_images,
    ensure_dir,
    get_logger,
)

logger = get_logger(__name__)


class PlantClassifier:
    """
    Plant identifier built on a pretrained EfficientNetB0 backbone (frozen)
    with a small trainable classification head. Transfer learning is used
    because each prairie species typically only has a few dozen to a few
    hundred training photos — far too few to train a deep CNN from scratch
    without overfitting.

    Parameters
    ----------
    model_path:
        Where to save / load the trained .keras model file.
    input_size:
        (height, width) fed into the network. Must match segmentation output.
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

        self.model: keras.Model | None = None
        self.class_names: list[str] = []
        self._class_index_path = self.model_path.parent / "class_index.json"


    def train(
        self,
        place_ids: list = config.INAT_PLACE_IDS,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        validation_split: float = config.VALIDATION_SPLIT,
        learning_rate: float = config.LEARNING_RATE,
        min_samples_per_species: int = config.INAT_MIN_PHOTOS_PER_SPECIES,
    ) -> keras.callbacks.History:
        """
        Train (or retrain) the classifier on images organised as:
            data/iNat_data/<place_id>/<species_name>/<image_files>

        Species with fewer than min_samples_per_species images are excluded
        from training and logged, but never deleted from disk. Class weights
        are computed so rarer species contribute proportionally to the loss.

        Saves the trained model and class index automatically.
        """
        working_dir = config.MODELS_DIR / "_train_working_dir"
        filtered_dir = build_filtered_training_dir(
            inat_base_dir=config.INAT_DIR,
            place_ids=place_ids,
            working_dir=working_dir,
            min_samples=min_samples_per_species,
        )

        logger.info(f"Loading training data from {filtered_dir}")
        train_ds, val_ds, class_names = self._build_datasets(
            filtered_dir, batch_size, validation_split
        )

        self.class_names = class_names
        num_classes = len(self.class_names)
        logger.info(f"Classes found: {num_classes}")

        class_weights = self._compute_class_weights(filtered_dir, class_names)
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
            class_weight=class_weights,
        )

        self._save()
        logger.info(f"Training complete. Model saved to {self.model_path}")
        return history

    def finetune(
            self,
            place_ids: list[str] = config.INAT_PLACE_IDS,
            epochs: int = 20,
            learning_rate: float = 1e-5,
            unfreeze_layers: int = 20,
            batch_size: int = config.BATCH_SIZE,
            validation_split: float = config.VALIDATION_SPLIT,
    ) -> keras.callbacks.History:
        """
        Fine-tune the top layers of the EfficientNetB0 backbone after initial
        training. Unfreezes the top N layers of the frozen backbone and trains
        at a very low learning rate so pretrained weights are adjusted gently
        rather than overwritten.

        Call this after train() has converged. Model must already be loaded
        or trained before calling this.
        """
        self._require_model()

        base_model = self.model.layers[1]  # EfficientNetB0 is the second layer
        base_model.trainable = True

        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        trainable_count = sum(1 for l in base_model.layers if l.trainable)
        logger.info(
            f"Fine-tuning: unfroze top {unfreeze_layers} of "
            f"{len(base_model.layers)} EfficientNet layers "
            f"({trainable_count} trainable layers total)"
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        working_dir = config.MODELS_DIR / "_train_working_dir"
        filtered_dir = build_filtered_training_dir(
            inat_base_dir=config.INAT_DIR,
            place_ids=place_ids,
            working_dir=working_dir,
            min_samples=config.INAT_MIN_PHOTOS_PER_SPECIES,
        )

        train_ds, val_ds, _ = self._build_datasets(
            filtered_dir,
            batch_size,
            validation_split,
            class_names=self.class_names,
        )

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
        logger.info(f"Fine-tuning complete. Model saved to {self.model_path}")
        return history

    def evaluate(
        self,
        place_ids: list = config.INAT_PLACE_IDS,
        batch_size: int = config.BATCH_SIZE,
    ) -> dict[str, float]:
        """
        Evaluate the model on a held-out validation split, using data from
        data/iNat_data/<place_id>/.

        Uses the exact class list saved at training time (self.class_names)
        so label indices always line up with the loaded model, regardless
        of new or sparse species folders present in data_dir.

        Returns a dict with 'loss' and 'accuracy'.
        """
        working_dir = config.MODELS_DIR / "_eval_working_dir"
        filtered_dir = build_filtered_training_dir(
            inat_base_dir=config.INAT_DIR,
            place_ids=place_ids,
            working_dir=working_dir,
            min_samples=config.INAT_MIN_PHOTOS_PER_SPECIES,
        )

        _, val_ds, _ = self._build_datasets(
            filtered_dir,
            batch_size,
            validation_split=config.VALIDATION_SPLIT,
            class_names=self.class_names,
        )

        loss, accuracy = self.model.evaluate(val_ds, verbose=0)
        metrics = {"loss": float(loss), "accuracy": float(accuracy)}
        logger.info(f"Evaluation — loss: {loss:.4f}  accuracy: {accuracy:.4f}")
        return metrics

    def predict_image(self, image: np.ndarray) -> tuple[str, float]:
        """
        Predict the species of a single pre-processed image array.

        Parameters
        ----------
        image:
            NumPy array of shape (H, W, 3), uint8 or float32, RGB order.

        Returns
        -------
        (species_name, confidence)
            species_name is "unknown" when confidence < threshold.
        """
        self._require_model()
        img = self._preprocess(image)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 3)
        probs = self.model.predict(img, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        species = self.class_names[idx] if confidence >= self.confidence_threshold else "unknown"
        return species, confidence

    def predict_batch(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Run predict_image over a list of images."""
        return [self.predict_image(img) for img in images]

    def load(self) -> None:
        """Load a previously trained model and class index from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"No saved model at {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        with self._class_index_path.open() as fh:
            self.class_names = json.load(fh)
        logger.info(f"Model loaded from {self.model_path} ({len(self.class_names)} classes)")

    def _build_datasets(
        self,
        data_dir: Path,
        batch_size: int,
        validation_split: float,
        class_names: list[str] | None = None,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
        """
        Build train/val tf.data.Dataset objects with augmentation and
        EfficientNet preprocessing.

        If class_names is provided, the dataset is restricted to exactly
        those classes (in that order) — used by evaluate() to guarantee
        label alignment with a previously trained model.
        """
        h, w = self.input_size

        train_ds = keras.utils.image_dataset_from_directory(
            str(data_dir),
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=(h, w),
            batch_size=batch_size,
            class_names=class_names,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            str(data_dir),
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=(h, w),
            batch_size=batch_size,
            class_names=class_names,
        )

        resolved_class_names = train_ds.class_names   # capture BEFORE .map()

        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomBrightness(0.1),
        ])

        def prepare_train(x, y):
            x = augmentation(x, training=True)
            x = preprocess_input(x)
            return x, y

        def prepare_val(x, y):
            x = preprocess_input(x)
            return x, y

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        val_ds = val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        return train_ds, val_ds, resolved_class_names

    def _build_model(self, num_classes: int, learning_rate: float) -> keras.Model:
        """
        Construct a transfer-learning classifier using a pretrained
        EfficientNetB0 backbone (frozen) with a custom classification head.
        Freezing the backbone keeps the trainable parameter count small,
        which helps avoid overfitting on a few dozen-to-hundred photos
        per species.
        """
        h, w = self.input_size
        inputs = keras.Input(shape=(h, w, 3))

        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(h, w, 3),
            pooling="avg",
        )
        base_model.trainable = False   # freeze pretrained weights

        x = base_model(inputs, training=False)
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

    def _compute_class_weights(self, data_dir: Path, class_names: list[str]) -> dict[int, float]:
        """
        Compute class weights inversely proportional to class frequency,
        so rare prairie species contribute as much to the loss as common
        ones rather than being drowned out.
        """
        counts = []
        for name in class_names:
            species_dir = Path(data_dir) / name
            counts.append(len(collect_images(species_dir)))

        total = sum(counts)
        num_classes = len(class_names)

        weights = {
            idx: total / (num_classes * count) if count > 0 else 1.0
            for idx, count in enumerate(counts)
        }
        return weights

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and apply EfficientNet preprocessing to a single image."""
        import cv2
        h, w = self.input_size
        resized = cv2.resize(image, (w, h)).astype(np.float32)
        return preprocess_input(resized)

    def _save(self) -> None:
        ensure_dir(self.model_path.parent)
        self.model.save(str(self.model_path))
        with self._class_index_path.open("w") as fh:
            json.dump(self.class_names, fh, indent=2)

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")