from tensorflow.keras import layers
import tensorflow as tf


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def mlp_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)
