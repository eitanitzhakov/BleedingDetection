import os
from model.model import Model
from preprocess.preprocess import PreProcess
from preprocess.DataLoader import DataLoader
from preprocess.ToTensor import ToTensorBatch
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class TrainPipeline:
    def __init__(
        self,
        data_dir,
        csv_path,
        batch_size=32,
        val_split=0.1,
        checkpoint_path="checkpoints/best_auc.h5",
        final_model_path="saved_models/final_model.h5"
    ):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.checkpoint_path = checkpoint_path
        self.final_model_path = final_model_path

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.final_model_path), exist_ok=True)

        self.preprocessor = PreProcess(input_dir=data_dir, csv_path=csv_path)
        self.to_tensor = ToTensorBatch()

        self.loader = DataLoader(
            data_dir=data_dir,
            csv_path=csv_path,
            batch_size=batch_size,
            preprocessor=self.preprocessor,
            to_tensor=self.to_tensor,
        )

        self.train_loader, self.val_loader = self.loader.split(self.val_split)

        self.model = Model()
        self.model.compile()

        self.checkpoint = ModelCheckpoint(
            filepath=self.checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False
        )

        self.early_stop = EarlyStopping(
            monitor="val_auc",
            patience=3,
            mode="max",
            restore_best_weights=True
        )

    def train_stage1(self, epochs=5):
        return self.model.fit(
            self.train_loader,
            validation_data=self.val_loader,
            epochs=epochs,
            callbacks=[self.checkpoint, self.early_stop]
        )

    def fine_tune(self, num_layers=40, epochs=10, lr=None):
        self.model.unfreeze(num_layers=num_layers, new_lr=lr)
        return self.model.fit(
            self.train_loader,
            validation_data=self.val_loader,
            epochs=epochs,
            callbacks=[self.checkpoint, self.early_stop]
        )

    def save_final(self):
        self.model.model.save(self.final_model_path)
