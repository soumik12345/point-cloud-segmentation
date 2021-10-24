import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from absl import logging

import tensorflow as tf
from tensorflow import keras

from .utils import create_example, split_list


class ShapeNetCoreTFRecordWriter:
    def __init__(
        self,
        object_category: str = "Airplane",
        n_sampled_points: int = 1024,
        viz_samples=None,
    ) -> None:
        self._get_files()
        self.dataset_path = "/tmp/.keras/datasets/PartAnnotation"
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
            sampled_indices = random.sample(list(range(n_points)), self.n_sampled_points)
            # Sampling points corresponding to sampled indices
            sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
            # Sampling corresponding one-hot encoded labels
            sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
            # Normalizing sampled point cloud
            norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
            norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
            self.point_clouds[index] = norm_point_cloud
            self.point_cloud_labels[index] = sampled_label_cloud
    
    def load_data(self) -> None:
        points_dir = os.path.join(
            self.dataset_path,
            '{}/points'.format(self.metadata[self.object_category]['directory'])
        )
        labels_dir = os.path.join(
            self.dataset_path,
            '{}/points_label'.format(self.metadata[self.object_category]['directory'])
        )
        points_files = glob(os.path.join(points_dir, '*.pts'))
        for point_file in tqdm(points_files):
            point_cloud = np.loadtxt(point_file)
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
                # label_data = label_encoder.transform(label_map)
                label_data = [self.labels.index(label) if label != 'none' else len(self.labels) for label in label_map]
                label_data = keras.utils.to_categorical(label_data, num_classes=len(self.labels) + 1)
                self.point_clouds.append(point_cloud)
                self.point_cloud_labels.append(label_data)
            except KeyError:
                # Use point cloud files without labels as test data
                self.test_point_clouds.append(point_cloud)
        self._sample_point_clouds()
    
    def write_tfrecords(self, samples_per_shard: int, tfrecord_dir: str):
        if not os.path.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir)
        if not os.path.exists(os.path.join(tfrecord_dir, self.object_category)):
            os.makedirs(os.path.join(tfrecord_dir, self.object_category))
        num_tfrecords = len(self.point_clouds) // samples_per_shard
        if len(self.point_clouds) % samples_per_shard:
            num_tfrecords += 1
        logging.info(f'num_tfrecords: {num_tfrecords}')
        point_cloud_shards = split_list(self.point_clouds, samples_per_shard)
        label_cloud_shards = split_list(self.point_cloud_labels, samples_per_shard)
        for index in range(num_tfrecords):
            point_cloud_shard = point_cloud_shards[index]
            label_cloud_shard = label_cloud_shards[index]
            file_name = f"shapenet_{index}.tfrec"
            logging.info(f'Writing TFRecord File {file_name}')
            with tf.io.TFRecordWriter(os.path.join(tfrecord_dir, self.object_category, file_name)) as writer:
                for sample_index in tqdm(range(len(point_cloud_shard))):
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=create_example(
                                point_cloud_shard[sample_index],
                                label_cloud_shard[sample_index]
                            )
                        )
                    )
                    writer.write(example.SerializeToString())
