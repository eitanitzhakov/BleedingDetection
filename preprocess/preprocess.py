import os
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import pandas as pd

class PreProcess:
    def __init__(self, input_dir, csv_path):
        self.input_dir = input_dir
        self.files = sorted([os.path.join(self.input_dir, f) for f in os.listdir(input_dir) if f.endswith('.dcm')])
        self.num_files = len(self.files)
        self.current_index = 0

        df = pd.read_csv(csv_path, dtype={"ID": str})
        df_any = df[df["ID"].str.endswith("_any", na=False)].copy()
        df_any["Image"] = df_any["ID"].str.split("_").str[1]
        self.labels_dict = dict(zip(df_any["Image"], df_any["Label"]))


    def load_dicom(self, filepath):
        return pydicom.dcmread(filepath)

    def to_hu(self, dicom):
        hu_img = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
        return hu_img

    def window(self, img, wc, ww):
        low = wc - ww / 2
        high = wc + ww / 2
        return np.clip(img, low, high)

    def apply_3_windows(self, hu_img):
        brain = self.window(hu_img, 40, 80)
        subdural = self.window(hu_img, 80, 200)
        bone = self.window(hu_img, 600, 2000)
        return brain, subdural, bone

    def normalize(self, img):
        img = img - img.min()
        max_v = img.max()
        if max_v > 0:
            img = img / max_v
        return img

    def channel_stack(self, brain, subdural, bone):
        brain = self.normalize(brain)
        subdural = self.normalize(subdural)
        bone = self.normalize(bone)

        stacked = np.stack([brain, subdural, bone], axis=-1)
        return stacked.astype(np.float32)

    def preprocess(self, filepath, size=(224, 224)):
        dicom = self.load_dicom(filepath)
        hu = self.to_hu(dicom)
        h, w = hu.shape[:2]
        hu = hu[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)]
        brain, subdural, bone = self.apply_3_windows(hu)
        brain = cv2.resize(brain, size)
        subdural = cv2.resize(subdural, size)
        bone = cv2.resize(bone, size)
        final_img = self.channel_stack(brain, subdural, bone)
        return final_img

    def get_label(self, dicom_path):
        dicom_id = os.path.basename(dicom_path).replace(".dcm", "")
        rows = self.df[self.df["ID"].str.startswith(dicom_id + "_")]

        if rows.empty:
            return None

        label_types = [
            "epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural", "any"
        ]

        labels = []
        for t in label_types:
            match = rows[rows["ID"].str.endswith("_" + t)]
            if not match.empty:
                labels.append(int(match["Label"].values[0]))
            else:
                labels.append(0)

        return labels

    def get_next_batch_with_labels(self, batch_size=64, size=(224, 224)):
        if self.current_index >= self.num_files:
            return None, None

        end_index = min(self.current_index + batch_size, self.num_files)
        batch_files = self.files[self.current_index:end_index]

        images = []
        labels = []

        for filepath in batch_files:
            img = self.preprocess(filepath, size=size)
            label = self.get_label(filepath)  # עכשיו תקין
            if label is None:
                continue
            images.append(img)
            labels.append(label)

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        self.current_index = end_index
        return images, labels
