"""
plant_classifier.py
CNN-based plant classifier built on TensorFlow/Keras using a pretrained
EfficientNetB4 backbone with a custom classification head.

Pipeline:
    1. augment_and_group()  — flip + rotate every photo, group by species
                              across all places into data/augmented_data/
    2. build_training_set() — filter species with < 25 photos, copy accepted
                              species into data/training_data/
    3. train()              — load training_data/, split 80/20, train model
    4. save model           — saved to models/plant_classifier.keras
"""

import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

from scripts import config
from utils.utils import collect_images, ensure_dir, get_logger

logger = get_logger(__name__)

# Directories for augmented and filtered training data
AUGMENTED_DIR: Path = config.DATA_DIR / "augmented_data"
TRAINING_DIR: Path = config.DATA_DIR / "training_data"
MIN_PHOTOS: int = 25


class PlantClassifier:
    """
    Plant identifier built on a pretrained EfficientNetB4 backbone with a
    custom classification head.

    Steps:
        clf = PlantClassifier()
        clf.augment_and_group()   # step 1 — augment and group by species
        clf.build_training_set()  # step 2 — filter out sparse species
        clf.train()               # step 3 — train and save model
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

    def augment_and_group(
            self,
            inat_dir: Path = config.INAT_DIR,
            place_ids: list[str] = config.INAT_PLACE_IDS,
            output_dir: Path = AUGMENTED_DIR,
    ) -> Path:
        """
        Group and augment photos by species across all places, using the first
        place in place_ids as the primary (left) — only species present in the
        primary place are included.

        Three versions of each photo are generated:
            1. Original
            2. Horizontal flip
            3. 90° clockwise rotation

        Safe to re-run — skips files that already exist.
        """
        ensure_dir(output_dir)

        primary_id = place_ids[0]
        primary_dir = Path(inat_dir) / primary_id

        if not primary_dir.exists():
            raise FileNotFoundError(
                f"Primary place directory not found: {primary_dir}"
            )

        # Build the species set from the primary place only
        primary_species = {d.name for d in primary_dir.iterdir() if d.is_dir()}
        logger.info(
            f"Primary place {primary_id}: {len(primary_species)} species — "
            f"supplemental places will be filtered to match"
        )

        total_written = 0
        for place_id in place_ids:
            place_dir = Path(inat_dir) / place_id
            if not place_dir.exists():
                logger.warning(f"Place directory not found, skipping: {place_dir}")
                continue

            is_primary = place_id == primary_id
            species_dirs = [d for d in place_dir.iterdir() if d.is_dir()]

            # For supplemental places, only process species in the primary set
            if not is_primary:
                before = len(species_dirs)
                species_dirs = [d for d in species_dirs if d.name in primary_species]
                logger.info(
                    f"  Place {place_id}: {len(species_dirs)} of {before} species "
                    f"match primary — including those only"
                )
            else:
                logger.info(f"  Place {place_id} (primary): {len(species_dirs)} species")

            for species_dir in species_dirs:
                species_output = ensure_dir(output_dir / species_dir.name)
                originals = [
                    p for p in collect_images(species_dir)
                    if not p.name.startswith("aug_")
                ]

                for img_path in originals:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"Could not read image, skipping: {img_path}")
                        continue

                    stem = f"{place_id}_{img_path.stem}"

                    # 1. Original
                    orig_dest = species_output / f"{stem}_orig{img_path.suffix}"
                    if not orig_dest.exists():
                        cv2.imwrite(str(orig_dest), image)
                        total_written += 1

                    # 2. Horizontal flip
                    flip_dest = species_output / f"{stem}_hflip{img_path.suffix}"
                    if not flip_dest.exists():
                        flipped = cv2.flip(image, 1)
                        cv2.imwrite(str(flip_dest), flipped)
                        total_written += 1

                    # 3. 90° clockwise rotation
                    rot_dest = species_output / f"{stem}_rot90{img_path.suffix}"
                    if not rot_dest.exists():
                        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        cv2.imwrite(str(rot_dest), rotated)
                        total_written += 1

        logger.info(
            f"Augmentation complete. {total_written} files written to {output_dir}"
        )
        return output_dir


    def build_training_set(
        self,
        augmented_dir: Path = AUGMENTED_DIR,
        output_dir: Path = TRAINING_DIR,
        min_photos: int = MIN_PHOTOS,
    ) -> Path:
        """
        Count photos in each species folder in augmented_dir. Species with
        fewer than min_photos total are excluded. Accepted species are copied
        into output_dir/<species_name>/ for use as the training/validation set.

        Safe to re-run — rebuilds output_dir from scratch each time to stay
        in sync if augmented_data changes.
        """
        if output_dir.exists():
            shutil.rmtree(output_dir)
        ensure_dir(output_dir)

        species_dirs = [d for d in Path(augmented_dir).iterdir() if d.is_dir()]
        included = []
        excluded = []

        for species_dir in species_dirs:
            count = len(collect_images(species_dir))
            if count < min_photos:
                excluded.append((species_dir.name, count))
            else:
                included.append((species_dir.name, count))
                shutil.copytree(species_dir, output_dir / species_dir.name)

        logger.info(
            f"Training set built: {len(included)} species included, "
            f"{len(excluded)} excluded (< {min_photos} photos)"
        )
        if excluded:
            logger.info(
                f"Excluded species: "
                f"{[f'{n} ({c})' for n, c in sorted(excluded)]}"
            )

        return output_dir


    def train(
        self,
        training_dir: Path = TRAINING_DIR,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        validation_split: float = config.VALIDATION_SPLIT,
        learning_rate: float = config.LEARNING_RATE,
    ) -> keras.callbacks.History:
        """
        Load images from training_dir, split 80/20 into train/val datasets
        ensuring every species is represented in both splits, build and compile
        EfficientNetB4, train, and save the model.
        """
        if not Path(training_dir).exists():
            raise FileNotFoundError(
                f"Training directory not found: {training_dir}. "
                "Run augment_and_group() then build_training_set() first."
            )

        logger.info(f"Loading training data from {training_dir}")
        train_ds, val_ds, class_names = self._build_datasets(
            training_dir, batch_size, validation_split
        )

        self.class_names = class_names
        num_classes = len(self.class_names)
        logger.info(f"Training on {num_classes} species")

        self.model = self._build_model(num_classes, learning_rate)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        self._save()
        logger.info(f"Training complete. Model saved to {self.model_path}")
        return history


    def evaluate(
        self,
        training_dir: Path = TRAINING_DIR,
        batch_size: int = config.BATCH_SIZE,
    ) -> dict[str, float]:
        """
        Evaluate the model on the validation split of training_dir.
        Uses the exact class list saved at training time.
        """
        self._require_model()

        _, val_ds, _ = self._build_datasets(
            training_dir,
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
        Predict the species of a single image array (H, W, 3), RGB order.
        Returns (species_name, confidence).
        species_name is 'unknown' when confidence < threshold.
        """
        self._require_model()
        img = self._preprocess(image)
        img = np.expand_dims(img, axis=0)
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
        Build train/val datasets from data_dir with EfficientNetB4 preprocessing.
        Uses a fixed seed so the 80/20 split is reproducible and every species
        appears in both splits.
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

        def prepare_train(x, y):
            return preprocess_input(x), y

        def prepare_val(x, y):
            return preprocess_input(x), y

        AUTOTUNE = tf.data.AUTOTUNE  # autotune helps optimize resources at runtime
        train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        val_ds = val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

        return train_ds, val_ds, resolved_class_names

    def _build_model(self, num_classes: int, learning_rate: float) -> keras.Model:
        """
        EfficientNetB4 backbone (frozen, pretrained ImageNet weights) with a
        custom classification head for num_classes plant species.
        """
        h, w = self.input_size

        base_model = EfficientNetB4(
            include_top=False,
            weights="imagenet",
            input_shape=(h, w, 3),
        )
        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.DROPOUT_RATE, name="top_dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

        model = keras.Model(
            inputs=base_model.input,
            outputs=outputs,
            name="plant_classifier",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        model.summary(print_fn=logger.info)
        return model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and apply EfficientNetB4 preprocessing to a single image."""
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