import tensorflow as tf
from typing import List, Dict


def get_float_feature(value):
    feature = lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    return feature(value)


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def split_list(given_list: List, chunk_size: int) -> List:
    return [
        given_list[offs : offs + chunk_size]
        for offs in range(0, len(given_list), chunk_size)
    ]


def create_example(point_cloud, label_cloud) -> Dict:
    return {
        # "point_cloud": get_float_feature(point_cloud.reshape(-1)),
        # "label_cloud": get_float_feature(label_cloud.reshape(-1)),
        "point_cloud": bytes_feature(tf.io.serialize_tensor(point_cloud)),
        "label_cloud": bytes_feature(tf.io.serialize_tensor(label_cloud)),
    }
