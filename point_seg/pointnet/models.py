from .blocks import conv_block
from .transform_block import transformation_block

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def get_baseline_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    """Filter values are from the original paper (https://arxiv.org/pdf/1612.00593.pdf)."""
    input_points = keras.Input(shape=(num_points, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features = conv_block(transformed_inputs, filters=64, name="first_block")
    features = conv_block(features, filters=64, name="second_block")
    transformed_fetaures = transformation_block(
        features, num_features=64, name="feature_transformation_block"
    )
    features = conv_block(transformed_fetaures, filters=64, name="third_block")
    features = conv_block(features, filters=128, name="fourth_block")
    features = conv_block(features, filters=1024, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation Head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [transformed_fetaures, global_features]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=512, name="first_seg_block"
    )
    segmentation_features = conv_block(
        segmentation_features, filters=256, name="second_seg_block"
    )
    segmentation_features = conv_block(
        segmentation_features, filters=128, name="third_seg_block"
    )
    segmentation_features = conv_block(
        segmentation_features, filters=128, name="fourth_seg_block"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(
        input_points, outputs, name="baseline_pointnet_segmentation_model"
    )


def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(num_points, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation Head
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)
