import tensorflow as tf

class ToTensorBatch:
    def __call__(self, batch):
        return tf.convert_to_tensor(batch, dtype=tf.float32)
