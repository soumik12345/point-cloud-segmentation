import tensorflow as tf
from typing import List, Dict


def get_float_feature(value):
    feature = lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    return feature(value)


def split_list(given_list: List, chunk_size: int) -> List:
    return [
        given_list[offs : offs + chunk_size]
        for offs in range(0, len(given_list), chunk_size)
    ]


def create_example(point_cloud, label_cloud) -> Dict:
    return {
        "point_cloud": get_float_feature(point_cloud.reshape(-1)),
        "label_cloud": get_float_feature(label_cloud.reshape(-1)),
    }
