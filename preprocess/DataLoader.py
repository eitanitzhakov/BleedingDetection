import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

class DataLoader:
    def __init__(self, data_dir, csv_path, batch_size, preprocessor, to_tensor, shuffle=True, augment=True):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.to_tensor = to_tensor
        self.shuffle = shuffle

        self.augment_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.1),
        ])


        self.file_paths, self.labels = self._load_labels()
        self.on_epoch_end()

    def _load_labels(self):
        df = pd.read_csv(self.csv_path)
        df["base_id"] = df["ID"].apply(lambda x: "_".join(x.split("_")[:2]))
        df["type"] = df["ID"].apply(lambda x: x.split("_")[2])

        hemorrhage_types = ["epidural", "intraparenchymal", "intraventricular",
                            "subarachnoid", "subdural", "any"]

        df_wide = df.pivot_table(index="base_id", columns="type", values="Label", fill_value=0).reset_index()

        for col in hemorrhage_types:
            if col not in df_wide.columns:
                df_wide[col] = 0

        file_paths = [os.path.join(self.data_dir, f"{bid}.dcm") for bid in df_wide["base_id"]]
        labels = df_wide[hemorrhage_types].values  # (N, 6)

        return np.array(file_paths), np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.file_paths) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.file_paths))
            np.random.shuffle(indices)
            self.file_paths = self.file_paths[indices]
            self.labels = self.labels[indices]

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_files = self.file_paths[start:end]
        batch_labels = self.labels[start:end]

        batch_images = []
        valid_labels = []

        for file_path, label_vec in zip(batch_files, batch_labels):
            try:
                img = self.preprocessor.preprocess(file_path)
                batch_images.append(img)
                valid_labels.append(label_vec)
            except Exception as e:
                print(f"[Warning] Skipping file {file_path}: {e}")
                continue

        if len(batch_images) == 0:
            raise ValueError(f"No valid files found in batch {index}")

        batch_images = np.array(batch_images)
        batch_images = self.to_tensor(batch_images)

        return batch_images, np.array(valid_labels)
