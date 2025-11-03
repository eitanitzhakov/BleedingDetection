from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import AUC

class Model:
    def __init__(self, input_shape=(224,224,3), classes = 6):
        self.input_shape = input_shape
        self.classes = classes
        self.model = self.__build_model()


    def __build_model(self):
        base_model = EfficientNetV2B0(include_top=False, input_shape=self.input_shape, weights="imagenet", pooling=None)
        base_model.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(self.classes, activation='sigmoid')(x)

        return models.Model(input = base_model.input, output = output, name = 'Hemmorage_Bleeding_ClassificationModel')

    def compile(self):
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC(name = "auc")])

    def summary(self):
        return self.model.summary()

    def predict(self, tensor):
        prediction = self.model.predict(tensor)
        return prediction