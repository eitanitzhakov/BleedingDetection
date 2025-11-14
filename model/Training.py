from model import *
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import wandb
from wandb.integration.keras import WandbCallback


class Training:
    def __init__(self, model, train_data, val_data, epochs=10, fine_tune_at=40, save_path=r"C:\Users\eitan\PycharmProjects\BleedingDetection\model\model_1.0", project_name = "Hemmorage_bleeding_detection"):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.fine_tune_at = fine_tune_at
        self.save_path = save_path
        self.project_name = project_name

        wandb.init(project=self.project_name, config={"epochs": epochs, "fine_tune_at": fine_tune_at, "optimizer": "AdamW", "loss": "binary_crossentropy", "input_shape": model.input_shape})

    def get_callbacks(self):
        return [WandbCallback(save_model=False), EarlyStopping(monitor="val_auc", patience=3, mode="max", restore_best_weights=True), ModelCheckpoint(filepath=self.save_path, monitor="val_auc", save_best_only=True, mode="max")]

    def train(self):
        history1 = self.model.fit(self.train_data, self.val_data, epochs=self.epochs, callbacks=self.get_callbacks())
        self.model.unfreeze(self.fine_tune_at)
        history2 = self.model.fit(self.train_data, self.val_data, epochs=self.epochs, callbacks=self.get_callbacks())
        self.model.save(self.save_path)
        wandb.finish()
        return history1, history2

