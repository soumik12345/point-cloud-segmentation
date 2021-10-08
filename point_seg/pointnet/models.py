from .blocks import conv_block
from .transform_block import transformation_block

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def get_baseline_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    """Filter values are from the original paper (https://arxiv.org/pdf/1612.00593.pdf)."""
    input_points = keras.Input(shape=(num_points, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(input_points, num_features=3)
    features = conv_block(transformed_inputs, filters=64)
    features = conv_block(features, filters=64)
    transformed_fetaures = transformation_block(features, num_features=64)
    features = conv_block(transformed_fetaures, filters=64)
    features = conv_block(features, filters=128)
    features = conv_block(features, filters=1024)
    global_features = layers.MaxPool1D(pool_size=num_points)(features)
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation Head.
    segmentation_input = layers.Concatenate()([transformed_fetaures, global_features])
    segmentation_features = conv_block(segmentation_input, filters=512)
    segmentation_features = conv_block(segmentation_features, filters=256)
    segmentation_features = conv_block(segmentation_features, filters=128)
    segmentation_features = conv_block(segmentation_features, filters=128)
    outputs = layers.Conv1D(num_classes, kernel_size=1, activation="softmax")(segmentation_features)
    return keras.Model(input_points, outputs)