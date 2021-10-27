import os
import json
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from absl import logging

import tensorflow as tf
from tensorflow import keras

from .utils import create_example, split_list
from point_seg.utils import make_dir


class ShapeNetCoreTFRecordWriter:
    def __init__(
        self,
        object_category: str = "Airplane",
        n_sampled_points: int = 1024,
        viz_samples=None,
    ) -> None:
        self.dataset_path = "/tmp/.keras/datasets/PartAnnotation"
        if not os.path.exists(self.dataset_path) or os.listdir(self.dataset_path) == 0:
            self._get_files()
        self.metadata = self._load_metadata()
        if object_category not in self.metadata.keys():
            raise KeyError(
                "Not a valid Shapenet Object. Must be one of "
                + str(self.metadata.keys())
            )
        else:
            self.object_category = object_category
        self.viz_samples = viz_samples
        self.n_sampled_points = n_sampled_points
        self.point_clouds, self.test_point_clouds = [], []
        self.point_cloud_labels, self.point_cloud_dataframes = [], []
        self.labels = self.metadata[self.object_category]["lables"]
        self.colors = self.metadata[self.object_category]["colors"]

    def _get_files(self):
        dataset_url = "https://github.com/soumik12345/point-cloud-segmentation/releases/download/v0.1/shapenet.zip"
        keras.utils.get_file(
            fname="shapenet.zip",
            origin=dataset_url,
            cache_subdir="datasets",
            hash_algorithm="auto",
            extract=True,
            archive_format="auto",
            cache_dir="datasets",
        )

    def _load_metadata(self):
        with open(os.path.join(self.dataset_path, "metadata.json")) as json_file:
            metadata = json.load(json_file)
        return metadata

    def _sample_point_clouds(self):
        for index in tqdm(range(len(self.point_clouds))):
            current_point_cloud = self.point_clouds[index]
            current_label_cloud = self.point_cloud_labels[index]
            n_points = len(current_point_cloud)
            # Randomly sampling respective indices
            sampled_indices = random.sample(
                list(range(n_points)), self.n_sampled_points
            )
            # Sampling points corresponding to sampled indices
            sampled_point_cloud = np.array(
                [current_point_cloud[i] for i in sampled_indices]
            )
            # Sampling corresponding one-hot encoded labels
            sampled_label_cloud = np.array(
                [current_label_cloud[i] for i in sampled_indices]
            )
            # Normalizing sampled point cloud
            norm_point_cloud = sampled_point_cloud - np.mean(
                sampled_point_cloud, axis=0
            )
            norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
            self.point_clouds[index] = sampled_point_cloud
            self.point_cloud_labels[index] = sampled_label_cloud

    def load_data(self, limit=None) -> None:
        points_dir = os.path.join(
            self.dataset_path,
            "{}/points".format(self.metadata[self.object_category]["directory"]),
        )
        labels_dir = os.path.join(
            self.dataset_path,
            "{}/points_label".format(self.metadata[self.object_category]["directory"]),
        )
        points_files = glob(os.path.join(points_dir, "*.pts"))
        if limit is not None:
            points_files = points_files[:limit]
        for point_file in tqdm(points_files):
            point_cloud = np.loadtxt(point_file)
            if point_cloud.shape[0] < self.n_sampled_points:
                continue
            file_id = point_file.split("/")[-1].split(".")[0]
            label_data, num_labels = {}, 0
            for label in self.labels:
                label_file = os.path.join(labels_dir, label, file_id + ".seg")
                if os.path.exists(label_file):
                    label_data[label] = np.loadtxt(label_file).astype("float32")
                    num_labels = len(label_data[label])
            try:
                label_map = ["none"] * num_labels
                for label in self.labels:
                    for i, data in enumerate(label_data[label]):
                        label_map[i] = label if data == 1 else label_map[i]
                label_data = [
                    self.labels.index(label) if label != "none" else len(self.labels)
                    for label in label_map
                ]
                label_data = keras.utils.to_categorical(
                    label_data, num_classes=len(self.labels) + 1
                )
                self.point_clouds.append(point_cloud)
                self.point_cloud_labels.append(label_data)
            except KeyError:
                # Use point cloud files without labels as test data
                self.test_point_clouds.append(point_cloud)
        self._sample_point_clouds()

    def _write_tfrecords_with_labels(
        self,
        point_clouds,
        label_clouds,
        samples_per_shard: int,
        tfrecord_dir: str,
        split: str,
    ):
        num_tfrecords = len(point_clouds) // samples_per_shard
        if len(point_clouds) % samples_per_shard:
            num_tfrecords += 1
        logging.info(f"num_tfrecords: {num_tfrecords}")
        point_cloud_shards = split_list(point_clouds, samples_per_shard)
        label_cloud_shards = split_list(label_clouds, samples_per_shard)
        lower_limit, upper_limit = 0, samples_per_shard
        for index in range(num_tfrecords):
            point_cloud_shard = point_cloud_shards[index]
            label_cloud_shard = label_cloud_shards[index]
            file_name = "shapenet-{}-{}-{:04d}-{:04d}.tfrec".format(
                self.n_sampled_points, split, lower_limit, upper_limit
            )
            lower_limit += samples_per_shard
            upper_limit += samples_per_shard
            logging.info(f"Writing TFRecord File {file_name}")
            with tf.io.TFRecordWriter(
                os.path.join(tfrecord_dir, self.object_category, split, file_name)
            ) as writer:
                for sample_index in tqdm(range(len(point_cloud_shard))):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=create_example(
                                point_cloud_shard[sample_index],
                                label_cloud_shard[sample_index],
                            )
                        )
                    )
                    writer.write(example.SerializeToString())

    def write_tfrecords(
        self, val_split: float, samples_per_shard: int, tfrecord_dir: str
    ):
        make_dir(tfrecord_dir)
        make_dir(os.path.join(tfrecord_dir, self.object_category))
        make_dir(os.path.join(tfrecord_dir, self.object_category, "train"))
        make_dir(os.path.join(tfrecord_dir, self.object_category, "val"))
        split_index = int(len(self.point_clouds) * (1 - val_split))
        train_point_clouds = self.point_clouds[:split_index]
        train_label_clouds = self.point_cloud_labels[:split_index]
        val_point_clouds = self.point_clouds[split_index:]
        val_label_clouds = self.point_cloud_labels[split_index:]
        logging.info("Creating Train TFRecords...")
        self._write_tfrecords_with_labels(
            train_point_clouds,
            train_label_clouds,
            samples_per_shard,
            tfrecord_dir,
            "train",
        )
        logging.info("Creating Validation TFRecords...")
        self._write_tfrecords_with_labels(
            val_point_clouds, val_label_clouds, samples_per_shard, tfrecord_dir, "val"
        )
