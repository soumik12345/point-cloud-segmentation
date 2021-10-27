import os
import glob
import json
import tensorflow as tf
from configs import shapenetcore
from absl import logging


_CFG = shapenetcore.get_config()


class TFRecordLoader:
    def __init__(
        self,
        tfrecord_dir: str = "./tfrecords",
        object_category: str = "Airplane",
        jitter_minval: float = -5e-3,
        jitter_maxval: float = 5e-3,
    ) -> None:
        self.tfrecord_dir = tfrecord_dir
        self.object_category = object_category
        self.jitter_minval = jitter_minval
        self.jitter_maxval = jitter_maxval
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        metadata_file = tf.keras.utils.get_file(
            fname="metadata.json",
            origin=_CFG.metadata_url,
            cache_subdir="datasets",
            hash_algorithm="auto",
            extract=False,
            archive_format="auto",
            cache_dir="datasets",
        )
        with open(metadata_file) as json_file:
            metadata = json.load(json_file)
        return metadata

    def _parse_tfrecord_fn(self, example):
        feature_description = {
            "point_cloud": tf.io.FixedLenFeature([], tf.string),
            "label_cloud": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        point_cloud = tf.io.parse_tensor(example["point_cloud"], out_type=tf.float32)
        label_cloud = tf.io.parse_tensor(example["label_cloud"], out_type=tf.float32)
        point_cloud.set_shape((_CFG.num_points, 3))
        label_cloud.set_shape(
            (_CFG.num_points, len(self.metadata[self.object_category]["lables"]) + 1)
        )
        return point_cloud, label_cloud

    def _augment(self, point_cloud, label_cloud):
        """Jitter point clouds"""
        noise = tf.random.uniform(
            tf.shape(point_cloud),
            self.jitter_minval,
            self.jitter_maxval,
            dtype=tf.float32,
        )
        point_cloud += noise[:, :, :3]
        return point_cloud, label_cloud

    def _generate_dataset(self, split: str, batch_size: int, drop_remainder: bool):
        tfrecord_loc = os.path.join(
            self.tfrecord_dir, self.object_category, split, "*.tfrec"
        )
        tfrecord_files = (
            tf.io.gfile.glob(tfrecord_loc)
            if "gs://" in tfrecord_loc
            else glob.glob(tfrecord_loc)
        )
        logging.info(f"{split} - Total TFRecord files found: {tfrecord_files}.")
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(
            self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        dataset = (
            dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            if split == "train"
            else dataset
        )
        return dataset

    def get_datasets(self, batch_size: int = 32, drop_remainder: bool = True):
        train_dataset = self._generate_dataset(
            split="train", batch_size=batch_size, drop_remainder=drop_remainder
        )
        val_dataset = self._generate_dataset(
            split="val", batch_size=batch_size, drop_remainder=drop_remainder
        )
        return train_dataset, val_dataset
