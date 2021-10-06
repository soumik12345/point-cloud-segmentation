import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers

from .blocks import conv_block, mlp_block


class OrthogonalRegularizer(regularizers.Regularizer):
    '''Referece: https://keras.io/examples/vision/pointnet/#build-a-model'''

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))


def transformation_net(inputs, num_features):
    '''Reference: https://keras.io/examples/vision/pointnet/#build-a-model'''
    x = conv_block(inputs, filters=64)
    x = conv_block(x, filters=128)
    x = conv_block(x, filters=1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512)
    x = mlp_block(x, filters=256)
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
    )(x)


def transformation_block(inputs, num_features):
    transformed_features = transformation_net(inputs, num_features)
    transformed_features = layers.Reshape((num_features, num_features))(transformed_features)
    return layers.Dot(axes=(2, 1))([inputs, transformed_features])
