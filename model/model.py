from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import AdamW


class Model:
    def __init__(self, input_shape=(224, 224, 3), classes=6, lr=1e-4, weight_decay=1e-5):
        self.input_shape = input_shape
        self.classes = classes
        self.lr = lr
        self.weight_decay = weight_decay

        self.model, self.base_model = self.__build_model()

    def __build_model(self):
        base = EfficientNetV2B0(
            include_top=False,
            input_shape=self.input_shape,
            weights="imagenet"
        )
        base.trainable = False

        inputs = base.input
        x = base.output

        x = layers.SpatialDropout2D(0.10, name="spatial_dropout")(x)

        x = layers.GlobalAveragePooling2D(name="gap")(x)

        x = layers.Dropout(0.20, name="dense_dropout_1")(x)
        x = layers.LayerNormalization(name="ln")(x)
        x = layers.Dense(128, activation="relu", name="dense_128")(x)
        x = layers.Dropout(0.30, name="dense_dropout_2")(x)


        outputs = layers.Dense(self.classes, activation="sigmoid", name="predictions")(x)

        model = models.Model(
            inputs=inputs,
            outputs=outputs,
            name="ICH_EfficientNetV2B0"
        )

        return model, base

    def compile(self):
        optimizer = AdamW(
            learning_rate=self.lr,
            weight_decay=self.weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(
                    name="auc",
                    multi_label=True,
                    num_labels=self.classes
                )
            ]
        )

    def summary(self):
        return self.model.summary()

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def unfreeze(self, num_layers=40, new_lr=None):
        for layer in self.base_model.layers[-num_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        if new_lr is None:
            self.lr = self.lr / 10.0
        else:
            self.lr = new_lr

        self.compile()

    def fit(self, train_data, validation_data=None, epochs=20, callbacks=None):
        return self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks
        )
