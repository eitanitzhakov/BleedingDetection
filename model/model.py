from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class Model:
    def __init__(self, input_shape=(224,224,3), classes=6):
        self.input_shape = input_shape
        self.classes = classes
        self.model = self.__build_model()

    def __build_model(self):
        base = EfficientNetV2B0(
            include_top=False,
            input_shape=self.input_shape,
            weights="imagenet"
        )
        base.trainable = False

        x = base.output
        x = layers.SpatialDropout2D(0.12)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.22)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.35)(x)
        out = layers.Dense(self.classes, activation="sigmoid")(x)

        return models.Model(
            inputs=base.input,
            outputs=out,
            name="ICH_EfficientNet"
        )

    def compile(self):
        self.model.compile(
            optimizer="AdamW",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(name="auc")
            ]
        )

    def summary(self):
        return self.model.summary()

    def predict(self, tensor):
        return self.model.predict(tensor)

    def unfreeze(self, num_layers=40):
        for layer in self.model.layers[-num_layers:]:
            layer.trainable = True

    def fit(self, train_data, val_data, epochs=20):
        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                patience=3,
                mode="max",
                restore_best_weights=True
            ),
            ModelCheckpoint(
                "best_model.h5",
                monitor="val_auc",
                mode="max",
                save_best_only=True
            )
        ]
        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )

