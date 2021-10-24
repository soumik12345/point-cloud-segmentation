import os
from glob import glob
import tensorflow as tf


class TFRecordLoader:
    def __init__(
        self, tfrecord_dir: str = "./tfrecords", object_category: str = "Airplane"
    ) -> None:
        self.tfrecord_dir = tfrecord_dir
        self.object_category = object_category

    def _parse_tfrecord_fn(self, example):
        feature_description = {
            "point_cloud": tf.io.FixedLenFeature([], tf.string),
            "label_cloud": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        point_cloud = tf.io.parse_tensor(example["point_cloud"], out_type=tf.float32)
        label_cloud = tf.io.parse_tensor(example["label_cloud"], out_type=tf.float32)
        return point_cloud, label_cloud

    def _augment(self, point_cloud_batch, label_cloud_batch):
        """Jitter point and label clouds."""
        noise = tf.random.uniform(
            tf.shape(label_cloud_batch), -0.005, 0.005, dtype=tf.float32
        )
        point_cloud_batch += noise[:, :, :3]
        return point_cloud_batch, label_cloud_batch

    def _generate_dataset(self, split: str, batch_size: int):
        tfrecord_files = glob(
            os.path.join(self.tfrecord_dir, self.object_category, split, "*.tfrec")
        )
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(
            self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size=batch_size)
        dataset = (
            dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            if split == "train"
            else dataset
        )
        return dataset

    def get_datasets(self, batch_size: int = 32):
        train_dataset = self._generate_dataset(split="train", batch_size=batch_size)
        val_dataset = self._generate_dataset(split="val", batch_size=batch_size)
        return train_dataset, val_dataset
