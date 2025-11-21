import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


class DataLoader(Sequence):
    def __init__(
        self,
        data_dir,
        csv_path,
        batch_size,
        preprocessor,
        to_tensor,
        file_paths=None,
        labels=None,
        shuffle=True,
        augment=True
    ):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.to_tensor = to_tensor
        self.shuffle = shuffle
        self.augment = augment

        self.augment_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.1),
        ])

        if file_paths is None:
            self.file_paths, self.labels = self._load_labels()
        else:
            self.file_paths = np.array(file_paths)
            self.labels = np.array(labels, dtype=np.float32)

        self.on_epoch_end()

    def _create_base_id(self, ID):
        parts = ID.split("_")
        return parts[0] + "_" + parts[1]

    def _get_type(self, ID):
        parts = ID.split("_")
        return parts[2]

    def _load_labels(self):
        df = pd.read_csv(self.csv_path)

        base_ids = []
        types = []

        for ID in df["ID"]:
            base_ids.append(self._create_base_id(ID))
            types.append(self._get_type(ID))

        df["base_id"] = base_ids
        df["type"] = types

        hemorrhage_types = ["epidural", "intraparenchymal", "intraventricular",
                            "subarachnoid", "subdural", "any"]

        df_wide = df.pivot_table(index="base_id", columns="type", values="Label", fill_value=0).reset_index()

        for col in hemorrhage_types:
            if col not in df_wide.columns:
                df_wide[col] = 0

        file_paths = []
        for bid in df_wide["base_id"]:
            file_paths.append(os.path.join(self.data_dir, bid + ".dcm"))

        labels = df_wide[hemorrhage_types].values

        return np.array(file_paths), np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            idx = np.arange(len(self.file_paths))
            np.random.shuffle(idx)
            self.file_paths = self.file_paths[idx]
            self.labels = self.labels[idx]

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_files = self.file_paths[start:end]
        batch_labels = self.labels[start:end]

        batch_images = []
        valid_labels = []

        for fp, lbl in zip(batch_files, batch_labels):
            try:
                img = self.preprocessor.preprocess(fp)
                batch_images.append(img)
                valid_labels.append(lbl)
            except:
                continue

        if len(batch_images) == 0:
            raise ValueError(f"No valid files in batch {index}")

        batch_images = np.array(batch_images)
        batch_images = self.to_tensor(batch_images)

        if self.augment:
            batch_images = self.augment_layer(batch_images)

        return batch_images, np.array(valid_labels)

    def split(self, val_split=0.1):
        total = len(self.file_paths)
        val_size = int(total * val_split)

        val_files = self.file_paths[:val_size]
        val_labels = self.labels[:val_size]

        train_files = self.file_paths[val_size:]
        train_labels = self.labels[val_size:]

        train_loader = DataLoader(
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            batch_size=self.batch_size,
            preprocessor=self.preprocessor,
            to_tensor=self.to_tensor,
            file_paths=train_files,
            labels=train_labels,
            shuffle=True,
            augment=True
        )

        val_loader = DataLoader(
            data_dir=self.data_dir,
            csv_path=self.csv_path,
            batch_size=self.batch_size,
            preprocessor=self.preprocessor,
            to_tensor=self.to_tensor,
            file_paths=val_files,
            labels=val_labels,
            shuffle=False,
            augment=False
        )

        return train_loader, val_loader
